"""One local application: Korean UI, free analysis, settings and history."""
from __future__ import annotations

import hmac
import logging
import os
import re
import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path
from urllib.parse import urlsplit

from dotenv import load_dotenv
from flask import Flask, jsonify, render_template, request
from werkzeug.exceptions import HTTPException

BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env", override=False)

from analysis_service import AnalysisService, InputError
from history_store import get_history_item, init_db, load_history, save_review
from mail_service import GmailAuthError, get_unread_emails
from server.gmail_client import GmailClientError, fetch_message_detail, fetch_recent_messages
from settings_store import SettingsConflict, SettingsStore

logger = logging.getLogger(__name__)
MESSAGE_ID = re.compile(r"^[A-Za-z0-9_-]{1,128}$")


def _json_object():
    if not request.is_json:
        raise InputError("Content-Type: application/json으로 보내 주세요.")
    value = request.get_json(silent=True)
    if not isinstance(value, dict):
        raise InputError("요청은 올바른 JSON 객체여야 합니다.")
    return value


def _integer(value, name, minimum, maximum, *, query=False):
    if query and isinstance(value, str) and value.isascii() and value.isdecimal():
        value = int(value)
    if type(value) is not int or not minimum <= value <= maximum:
        raise InputError(f"{name}은 {minimum}~{maximum} 사이의 정수여야 합니다.")
    return value


def _message_id(value):
    if not isinstance(value, str) or not MESSAGE_ID.fullmatch(value):
        raise InputError("메일 ID 형식이 올바르지 않습니다.")
    return value


def _gmail_error(exc):
    if isinstance(exc, GmailAuthError):
        return jsonify(error="gmail_configuration", detail="Gmail 연동 설정이 없습니다. 로컬 분석은 계속 사용할 수 있습니다. 기존 읽기 전용 자격 증명을 준비한 경우에만 연동을 실행해 주세요."), 400
    return jsonify(error="gmail_fetch_failed", detail="메일을 가져오지 못했습니다. 연결 상태를 확인한 뒤 다시 시도해 주세요."), 502


def create_app(data_dir: str | Path | None = None) -> Flask:
    app = Flask(__name__)
    app.config.update(MAX_CONTENT_LENGTH=1_048_576, JSON_SORT_KEYS=False)
    app.json.ensure_ascii = False
    directory = Path(data_dir or os.getenv("PHISH_DATA_DIR") or BASE_DIR / "data").resolve()
    history_path = directory / "phishguard.db"
    settings = SettingsStore(directory / "settings.json")
    init_db(history_path)
    service = AnalysisService(history_path, settings)
    app.extensions.update(analysis_service=service, settings_store=settings, history_path=history_path)
    api_token = os.getenv("PHISH_API_TOKEN", "")

    @app.before_request
    def local_request_policy():
        # Prevent hostile web pages posting to this local service and DNS rebinding.
        hostname = urlsplit(request.host_url).hostname
        if hostname not in {"127.0.0.1", "localhost", "::1"}:
            return jsonify(error="invalid_host", detail="로컬 주소로 접속해 주세요."), 403
        origin = request.headers.get("Origin")
        if request.headers.get("Sec-Fetch-Site") == "cross-site" or (origin and origin != request.host_url.rstrip("/")):
            return jsonify(error="invalid_origin", detail="이 앱 화면에서 다시 요청해 주세요."), 403
        protected = (request.path.startswith("/api/") and request.path != "/api/health") or request.path.startswith("/metrics/")
        if protected and api_token:
            supplied = request.headers.get("X-API-Key", "")
            authorization = request.headers.get("Authorization", "")
            if not supplied and authorization.startswith("Bearer "):
                supplied = authorization[7:]
            if not hmac.compare_digest(supplied.encode("utf-8"), api_token.encode("utf-8")):
                return jsonify(error="unauthorized", detail="API 토큰을 확인한 뒤 다시 시도해 주세요."), 401

    @app.after_request
    def response_headers(response):
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["Referrer-Policy"] = "no-referrer"
        response.headers["Content-Security-Policy"] = "default-src 'self'; script-src 'self'; style-src 'self'; img-src 'self' data:; connect-src 'self'; object-src 'none'; base-uri 'none'; frame-ancestors 'none'; form-action 'self'"
        if request.path.startswith(("/api/", "/metrics/")):
            response.headers["Cache-Control"] = "no-store"
        return response

    @app.errorhandler(SettingsConflict)
    def settings_conflict(exc):
        return jsonify(error="settings_conflict", detail="다른 화면에서 설정이 바뀌었습니다. 설정을 새로 불러온 뒤 저장해 주세요."), 409

    @app.errorhandler(ValueError)
    def invalid_input(exc):
        return jsonify(error="invalid_input", detail=str(exc)), 400

    @app.errorhandler(OSError)
    @app.errorhandler(sqlite3.Error)
    def storage_failure(exc):
        logger.error("Storage failure: %s", type(exc).__name__)
        return jsonify(error="storage_unavailable", detail="결과 또는 설정을 저장하지 못했습니다. 저장 위치와 여유 공간을 확인한 뒤 다시 시도해 주세요."), 503

    @app.errorhandler(RuntimeError)
    def backend_failure(exc):
        logger.error("Backend unavailable: %s", type(exc).__name__)
        return jsonify(error="analysis_unavailable", detail="분석을 완료하지 못했습니다. 분석 백엔드 설정을 확인한 뒤 다시 시도해 주세요."), 503

    @app.errorhandler(HTTPException)
    def http_failure(exc):
        detail = "요청 크기는 1 MiB 이하여야 합니다." if exc.code == 413 else "요청 경로 또는 형식을 확인해 주세요."
        return jsonify(error="request_error", detail=detail), exc.code

    @app.errorhandler(Exception)
    def unexpected_failure(exc):
        logger.error("Unexpected application failure: %s", type(exc).__name__)
        return jsonify(error="internal_error", detail="처리 중 오류가 발생했습니다. 입력을 보관한 채 다시 시도해 주세요."), 500

    @app.get("/")
    def index():
        return render_template("index.html")

    @app.get("/api/health")
    def health():
        return jsonify(status="ok", backend=os.getenv("PHISH_BACKEND", "rules"), external_api_calls=False, api_token_required=bool(api_token))

    @app.get("/api/settings")
    def get_settings():
        return jsonify(settings.load())

    @app.put("/api/settings")
    def update_settings():
        return jsonify(settings.update(_json_object()))

    @app.post("/api/analyze")
    def analyze():
        payload = _json_object()
        return jsonify(service.analyze(payload, force=payload.get("force", False)))

    @app.post("/api/analyze_batch")
    def analyze_batch():
        payload = _json_object()
        items = payload.get("items")
        if not isinstance(items, list) or not 1 <= len(items) <= 50:
            raise InputError("items는 1~50개의 메일 배열이어야 합니다.")
        force = payload.get("force", False)
        if type(force) is not bool:
            raise InputError("force는 true 또는 false여야 합니다.")
        results = []
        for position, item in enumerate(items):
            try:
                result = service.analyze(item, force=force)
            except InputError as exc:
                result = {"error": "invalid_input", "detail": str(exc)}
            results.append({**result, "index": position})
        return jsonify(results=results)

    def analyze_gmail_items(emails, force=False):
        if not isinstance(emails, list):
            raise RuntimeError("메일 공급자 응답 형식 오류")
        results = []
        for email in emails:
            if not isinstance(email, dict):
                results.append({"error": "invalid_message", "detail": "메일 내용 형식이 올바르지 않습니다."})
                continue
            gmail_id = email.get("gmail_id", email.get("id"))
            try:
                results.append(service.analyze({**email, "gmail_id": gmail_id}, force=force))
            except InputError as exc:
                results.append({"id": gmail_id, "error": "invalid_message", "detail": str(exc)})
        return results

    @app.post("/api/fetch_and_analyze")
    def fetch_and_analyze():
        payload = _json_object()
        limit = _integer(payload.get("max_results", settings.load()["max_results"]), "max_results", 1, 50)
        force = payload.get("force", False)
        if type(force) is not bool:
            raise InputError("force는 true 또는 false여야 합니다.")
        try:
            emails = get_unread_emails(max_results=limit)
        except (GmailAuthError, GmailClientError, RuntimeError) as exc:
            return _gmail_error(exc)
        return jsonify(results=analyze_gmail_items(emails, force=force))

    @app.post("/api/analyze_selected")
    def analyze_selected():
        payload = _json_object()
        ids = payload.get("message_ids")
        if not isinstance(ids, list) or not 1 <= len(ids) <= 50:
            raise InputError("message_ids는 1~50개의 메일 ID 배열이어야 합니다.")
        for message_id in ids:
            _message_id(message_id)
        limit = _integer(payload.get("max_results", len(ids)), "max_results", 1, 50)
        force = payload.get("force", False)
        if type(force) is not bool:
            raise InputError("force는 true 또는 false여야 합니다.")
        selected = list(dict.fromkeys(ids))[:limit]
        results = []
        warnings = []
        for message_id in selected:
            try:
                detail = fetch_message_detail(message_id)
            except (GmailAuthError, GmailClientError, RuntimeError) as exc:
                if isinstance(exc, GmailAuthError):
                    return _gmail_error(exc)
                results.append({"id": message_id, "error": "gmail_fetch_failed", "detail": "메일을 가져오지 못했습니다."})
                warnings.append(f"{message_id}: 가져오기 실패")
                continue
            results.extend(analyze_gmail_items([{**detail, "gmail_id": message_id}], force=force))
        return jsonify(results=results, warnings=warnings)

    @app.post("/api/emails/<message_id>/analyze")
    def analyze_gmail_email(message_id):
        _message_id(message_id)
        payload = _json_object()
        force = payload.get("force", False)
        if type(force) is not bool:
            raise InputError("force는 true 또는 false여야 합니다.")
        try:
            detail = fetch_message_detail(message_id)
        except (GmailAuthError, GmailClientError, RuntimeError) as exc:
            return _gmail_error(exc)
        result = analyze_gmail_items([{**detail, "gmail_id": message_id}], force=force)[0]
        if "error" in result:
            return jsonify(result), 400
        return jsonify(**result, messageId=message_id, isPhishing=result["label"] == "phishing", riskScore=round(result["risk_score"] * 100), summary=result["feedback"])

    @app.get("/api/list_emails")
    @app.get("/api/emails")
    def list_emails():
        limit = _integer(request.args.get("max_results", settings.load()["max_results"]), "max_results", 1, 50, query=True)
        label = request.args.get("label")
        if label and (len(label) > 100 or not re.fullmatch(r"[A-Za-z0-9_-]+", label)):
            raise InputError("메일 라벨 형식이 올바르지 않습니다.")
        try:
            messages = fetch_recent_messages(limit=limit, label=label)
        except (GmailAuthError, GmailClientError, RuntimeError) as exc:
            return _gmail_error(exc)
        return jsonify(messages=messages)

    @app.get("/api/emails/<message_id>")
    def get_email_detail(message_id):
        _message_id(message_id)
        try:
            return jsonify(fetch_message_detail(message_id))
        except (GmailAuthError, GmailClientError, RuntimeError) as exc:
            return _gmail_error(exc)

    @app.get("/api/history")
    def get_history():
        limit = _integer(request.args.get("limit", 200), "limit", 1, 1000, query=True)
        days = request.args.get("days")
        if days is not None:
            days = _integer(days, "days", 0, 3650, query=True)
        return jsonify(results=load_history(limit=limit, days=days, db_path=history_path))

    @app.get("/api/history/<record_id>")
    def get_history_detail(record_id):
        record = get_history_item(record_id, db_path=history_path)
        if record is None:
            return jsonify(error="not_found", detail="분석 이력이 없습니다."), 404
        return jsonify(record)

    @app.post("/api/history/<record_id>/review")
    def review_history(record_id):
        label = _json_object().get("label")
        if label not in ("normal", "phishing"):
            raise InputError("검토 라벨은 normal 또는 phishing이어야 합니다.")
        try:
            return jsonify(save_review(record_id, label, db_path=history_path))
        except KeyError:
            return jsonify(error="not_found", detail="분석 이력이 없습니다."), 404

    @app.get("/metrics/summary")
    def metrics_summary():
        with sqlite3.connect(history_path) as conn:
            total, phishing, normal, today, avg_latency = conn.execute(
                "SELECT COUNT(*), SUM(label='phishing'), SUM(label='normal'), SUM(date=?), AVG(latency_ms) FROM email_analysis",
                (datetime.now(timezone.utc).date().isoformat(),),
            ).fetchone()
            reviewed, disagreements = conn.execute("SELECT COUNT(*), SUM(ea.label != hl.gt_label) FROM human_labels hl JOIN email_analysis ea ON ea.id=hl.id").fetchone()
            cutoff = (datetime.now(timezone.utc).date() - timedelta(days=6)).isoformat()
            rows = conn.execute("SELECT date, COUNT(*), SUM(label='phishing') FROM email_analysis WHERE date >= ? GROUP BY date ORDER BY date", (cutoff,)).fetchall()
        return jsonify(total=total, phishing=phishing or 0, normal=normal or 0, today=today or 0, avg_latency_ms=round(avg_latency, 3) if avg_latency is not None else None, reviewed=reviewed, disagreements=disagreements or 0, trend=[dict(date=row[0], total=row[1], phishing=row[2] or 0) for row in rows], timezone="UTC")

    return app


app = create_app()

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    app.run(host="127.0.0.1", port=8080, debug=False, threaded=True)
