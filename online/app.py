"""Stateless, no-cost rules API and browser UI for the public online mode.

The only retained server state is a bounded process-local load limiter. It is
not a user quota, account identity, or durable/global serverless rate limit.
"""
from __future__ import annotations

from collections import deque
from dataclasses import asdict
import ipaddress
import json
import math
import os
from pathlib import Path
import re
import threading
import time
from typing import Callable, Iterable
from urllib.parse import urlsplit

from flask import Flask, jsonify, request, send_from_directory
from werkzeug.exceptions import HTTPException

from feedback import make_feedback
from public_validation import InputError, MAX_JSON_BYTES, prepare_public_item, threshold_percent
from scoring import RULES_VERSION, compute_rule_score, decide_label

CSP = ("default-src 'none'; script-src 'self'; style-src 'self'; connect-src 'self'; "
       "img-src 'self'; font-src 'none'; object-src 'none'; base-uri 'none'; "
       "frame-ancestors 'none'; form-action 'none'")
PUBLIC_DIR = Path(__file__).resolve().parent / "public"
LOCAL_ASSETS = {"index.html", "app.js", "app.css", "storage.js", "favicon.svg"}


def canonical_origin(value: str) -> tuple[str, str]:
    """Validate an exact ASCII origin. Never accept wildcards or forwarded host."""
    if not isinstance(value, str) or value != value.strip() or not value.isascii():
        raise ValueError("Public origins must be canonical ASCII origins.")
    try:
        parsed = urlsplit(value)
        port = parsed.port
    except ValueError as exc:
        raise ValueError("Public origin has an invalid host or port.") from exc
    host = parsed.hostname or ""
    if (parsed.scheme not in {"http", "https"} or not host
            or parsed.username is not None or parsed.password is not None
            or parsed.path or parsed.query or parsed.fragment or host.endswith(".")):
        raise ValueError("Public origins contain only a scheme and host/port.")
    loopback = host == "localhost"
    try:
        loopback = loopback or ipaddress.ip_address(host).is_loopback
    except ValueError:
        labels = host.split(".")
        if len(host) > 253 or any(not re.fullmatch(r"[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?", label) for label in labels):
            raise ValueError("Public origin has an invalid DNS hostname.")
    if parsed.scheme == "http" and not loopback:
        raise ValueError("Public non-loopback origins must use HTTPS.")
    if port == 0 or (parsed.scheme, port) in {("http", 80), ("https", 443)}:
        raise ValueError("Do not include a default port in a public origin.")
    netloc = f"[{host}]" if ":" in host else host
    netloc += f":{port}" if port is not None else ""
    canonical = f"{parsed.scheme}://{netloc}"
    if canonical != value:
        raise ValueError("Public origins must use their canonical form.")
    return canonical, netloc


def configured_origins(vercel_runtime: bool) -> tuple[str, ...]:
    explicit = os.environ.get("PHISH_PUBLIC_ORIGINS", "")
    values = [part.strip() for part in explicit.split(",") if part.strip()]
    if vercel_runtime:
        # Official platform variables, available at build and runtime. Never
        # derive allowed origins from request headers or wildcard vercel.app.
        for name in ("VERCEL_URL", "VERCEL_BRANCH_URL", "VERCEL_PROJECT_PRODUCTION_URL"):
            host = os.environ.get(name, "").strip()
            if host:
                values.append("https://" + host)
    elif not values:
        values = ["http://127.0.0.1:5000", "http://localhost:5000"]
    return tuple(dict.fromkeys(canonical_origin(value)[0] for value in values))


class InstanceLimiter:
    """Thread-safe, bounded sliding window for one live Python process only."""
    def __init__(self, limit: int = 30, window_seconds: int = 60, concurrency: int = 2,
                 clock: Callable[[], float] = time.monotonic):
        self.limit = limit
        self.window_seconds = window_seconds
        self.concurrency = concurrency
        self.clock = clock
        self._lock = threading.Lock()
        self._starts: deque[float] = deque()
        self._active = 0

    def enter(self) -> int | None:
        with self._lock:
            now = self.clock()
            while self._starts and self._starts[0] <= now - self.window_seconds:
                self._starts.popleft()
            if len(self._starts) >= self.limit:
                return max(1, math.ceil(self.window_seconds - (now - self._starts[0])))
            if self._active >= self.concurrency:
                return 1
            self._starts.append(now)
            self._active += 1
            return None

    def leave(self) -> None:
        with self._lock:
            self._active -= 1


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise InputError("JSON 객체의 같은 항목을 반복해서 입력할 수 없습니다.")
        result[key] = value
    return result


def _invalid_constant(value):
    raise InputError("JSON에는 유한한 숫자만 사용할 수 있습니다.")


def read_payload() -> dict:
    if request.mimetype != "application/json" or request.headers.get("Content-Encoding", "identity") != "identity":
        raise InputError("UTF-8 application/json 형식으로 보내 주세요.")
    if request.query_string:
        raise InputError("분석 입력은 URL이 아닌 JSON 본문으로 보내 주세요.")
    raw = request.get_data(cache=False)
    if len(raw) > MAX_JSON_BYTES:
        # MAX_CONTENT_LENGTH also bounds streaming/Content-Length parsing.
        raise InputError("JSON 요청은 64 KiB 이하여야 합니다.")
    try:
        payload = json.loads(raw.decode("utf-8"), object_pairs_hook=_unique_object, parse_constant=_invalid_constant)
    except (UnicodeDecodeError, ValueError, RecursionError) as exc:
        if isinstance(exc, InputError):
            raise
        raise InputError("올바른 UTF-8 JSON 요청을 입력해 주세요.") from exc
    if not isinstance(payload, dict) or set(payload) - {"item", "threshold"} or "item" not in payload:
        raise InputError("요청에는 메일 item과 임계값 threshold만 입력할 수 있습니다.")
    return payload


def create_app(public_origins: Iterable[str] | None = None, *, vercel_runtime: bool | None = None,
               limiter: InstanceLimiter | None = None) -> Flask:
    on_vercel = os.environ.get("VERCEL") == "1" if vercel_runtime is None else vercel_runtime
    origins = configured_origins(on_vercel) if public_origins is None else tuple(public_origins)
    allowed = dict(canonical_origin(origin) for origin in origins)
    expected = {host: urlsplit(origin).scheme for origin, host in allowed.items()}
    application = Flask(__name__, static_folder=None)
    application.config.update(MAX_CONTENT_LENGTH=MAX_JSON_BYTES, PROPAGATE_EXCEPTIONS=False)
    application.json.ensure_ascii = False
    application.config["PUBLIC_ORIGINS"] = tuple(allowed)
    application.extensions["instance_limiter"] = limiter if limiter is not None else InstanceLimiter()

    @application.before_request
    def exact_origin_policy():
        if request.host not in expected:
            return jsonify(error="invalid_host", detail="설정된 앱 주소로 요청해 주세요."), 403
        # Vercel terminates TLS before WSGI. Only that platform runtime may use
        # its documented protocol header; generic local/custom WSGI ignores it.
        scheme = request.headers.get("X-Forwarded-Proto", request.scheme) if on_vercel else request.scheme
        if scheme != expected[request.host]:
            return jsonify(error="invalid_scheme", detail="설정된 연결 방식으로 요청해 주세요."), 403
        if request.path.startswith("/api/"):
            supplied_origin = request.headers.get("Origin")
            current_origin = f"{scheme}://{request.host}"
            if (supplied_origin is not None and supplied_origin != current_origin
                    or request.headers.get("Sec-Fetch-Site") == "cross-site"):
                return jsonify(error="invalid_origin", detail="같은 앱 화면에서 요청해 주세요."), 403
            if request.method not in {"GET", "HEAD", "OPTIONS"} and supplied_origin != current_origin:
                return jsonify(error="invalid_origin", detail="같은 앱 화면에서 요청해 주세요."), 403

    @application.after_request
    def response_policy(response):
        response.headers["Cache-Control"] = "no-store, max-age=0"
        response.headers["CDN-Cache-Control"] = "no-store"
        response.headers["Vercel-CDN-Cache-Control"] = "no-store"
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["Referrer-Policy"] = "no-referrer"
        response.headers["Content-Security-Policy"] = CSP
        response.headers["Permissions-Policy"] = "camera=(), microphone=(), geolocation=()"
        response.headers["X-Frame-Options"] = "DENY"
        return response

    @application.errorhandler(InputError)
    def invalid_input(exc):
        # Only our fixed field/limit messages; no submitted value is formatted.
        return jsonify(error="invalid_input", detail=str(exc)), 400

    @application.errorhandler(HTTPException)
    def invalid_request(exc):
        detail = "JSON 요청은 64 KiB 이하여야 합니다." if exc.code == 413 else "요청 경로 또는 형식을 확인해 주세요."
        return jsonify(error="request_error", detail=detail), exc.code

    @application.errorhandler(Exception)
    def unexpected_error(exc):
        # Do not log exception text, body, host, attachment, or submitted URL.
        return jsonify(error="internal_error", detail="분석을 완료하지 못했습니다. 잠시 후 다시 시도해 주세요."), 500

    @application.get("/")
    def index():
        return send_from_directory(PUBLIC_DIR, "index.html", conditional=False)

    @application.get("/<asset>")
    def local_asset(asset):
        # Only these exact UI files. No arbitrary path, README, API source,
        # data directory, or local-app template can be served by this route.
        if asset not in LOCAL_ASSETS:
            return jsonify(error="not_found", detail="요청한 화면을 찾을 수 없습니다."), 404
        return send_from_directory(PUBLIC_DIR, asset, conditional=False)

    @application.get("/api/health")
    def health():
        return jsonify(status="ok", mode="online_stateless", backend="rules", score_kind="rule_risk_index",
                       rules_version=RULES_VERSION, external_api_calls=False, persistent_storage=False,
                       gmail_enabled=False, hf_enabled=False, user_authentication=False,
                       limits={"json_bytes": MAX_JSON_BYTES, "body_chars": 16000, "links": 50,
                               "url_chars": 2048, "url_length_unit": "utf16_code_units", "attachments": 20},
                       rate_limit={"scope": "process_instance", "analyses_per_minute": 30, "concurrency": 2,
                                   "durable": False, "global_quota": False, "resets_on_restart": True})

    @application.post("/api/analyze")
    def analyze():
        started = time.perf_counter()
        load_limiter = application.extensions["instance_limiter"]
        retry_after = load_limiter.enter()
        if retry_after is not None:
            response = jsonify(error="rate_limited", detail="요청이 많습니다. 잠시 후 다시 시도해 주세요.",
                               retry_after_seconds=retry_after)
            response.headers["Retry-After"] = str(retry_after)
            return response, 429
        try:
            payload = read_payload()
            threshold = threshold_percent(payload.get("threshold", 20))
            prepared = prepare_public_item(payload["item"])
            text = "\n".join(part for part in (prepared["subject"], prepared["body"]) if part)
            # Hard-selected approved rules. No environment-selected provider,
            # Gmail, model loader, network, DNS, history, or settings import.
            score, signals = compute_rule_score(text, prepared)
            if not math.isfinite(score):
                raise RuntimeError("Invalid rule result")
            label = decide_label(score, threshold / 100)
            reasons = [signal.reason for signal in signals]
            warnings = []
            if any(prepared["auth_results"].get(key) is None for key in ("spf_pass", "dkim_pass", "dmarc_pass")):
                warnings.append("일부 인증 결과는 없거나 확인되지 않았습니다. 실패로 계산하지 않았습니다.")
            # Derived evidence can contain host/domain/attachment/keyword;
            # browser consent/storage UI explains that sensitivity explicitly.
            return jsonify(backend="rules", score_kind="rule_risk_index", rules_version=RULES_VERSION,
                           risk_score=score, label=label, threshold=threshold,
                           signals=[asdict(signal) for signal in signals], reasons=reasons, warnings=warnings,
                           phish_probability=None, persistent_storage=False, external_api_calls=False,
                           feedback=make_feedback(label, score, threshold / 100, reasons),
                           request_ms=round((time.perf_counter() - started) * 1000, 3))
        finally:
            load_limiter.leave()

    return application


# Official Flask Vercel entrypoint. Configuration uses explicit app origins or
# platform-provided domains. Unconfigured deployments reject unknown hosts.
app = create_app()

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False)
