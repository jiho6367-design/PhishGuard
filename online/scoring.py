"""Deterministic offline phishing indicators: risk indexes, not probabilities."""
from __future__ import annotations
import ipaddress
import math
import re
from dataclasses import dataclass
from email.utils import parseaddr
from functools import lru_cache
from typing import Any, Dict, Iterable, List, Tuple
from urllib.parse import urlsplit

RULES_VERSION = "2026-10-10.rules.v2"
SUSPICIOUS_TLDS = {".xyz", ".top", ".icu", ".vip", ".click", ".link", ".pw", ".live", ".shop", ".center", ".work", ".quest"}
SHORTENER_HOSTS = {"bit.ly", "t.co", "goo.gl", "ow.ly", "tinyurl.com", "is.gd", "buff.ly", "shorte.st", "lnkd.in", "cutt.ly"}
SUSPICIOUS_EXTENSIONS = {".zip", ".exe", ".scr", ".bat", ".cmd", ".vbs", ".js", ".jar", ".docm", ".xlsm"}
TRUSTED_DOMAINS = {"google.com", "microsoft.com", "office.com", "outlook.com", "live.com", "sharepoint.com", "github.com", "dropbox.com", "box.com", "adobe.com"}
URGENT_KEYWORDS = {"otp", "2fa", "mfa", "verify", "verification", "secure", "suspend", "suspended", "reset", "urgent", "immediately", "account locked", "password", "bank", "invoice", "payment", "긴급", "즉시", "비밀번호", "계정 정지", "계정 잠김", "인증", "결제", "송금"}
ACTION_LINK_KEYWORDS = {"login", "log in", "verify", "verification", "confirm", "access", "view", "review", "claim", "wallet", "secure session", "document", "로그인", "인증", "확인", "접속", "클릭", "비밀번호", "다운로드", "송금"}
RECEIPT_KEYWORDS = {"결제", "영수증", "구독", "청구", "승인", "금액", "krw", "카드", "결제일", "주문번호", "invoice", "receipt", "payment", "subscription"}
ACTION_KEYWORDS = ACTION_LINK_KEYWORDS | {"click", "password", "update payment", "계정", "정지", "환불"}
URL_REGEX = re.compile(r"(?:(?:https?://)|(?:www\.))[^\s<>\"']+", re.IGNORECASE)
KOREAN_ENDINGS = r"(?:을|를|이|가|은|는|에|와|과|의|도|만|로|으로|에서|하세요|하십시오|해주세요|해\s*주세요|하셔야|해야|하기|되었습니다|됩니다|된|할|해|요)?"

@dataclass(frozen=True)
class RuleSignal:
    name: str
    score: float
    reason: str

def clamp_confidence(value: float) -> float:
    """Compatibility name: clamp a finite risk index to the unit interval."""
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError):
        return 0.0
    return max(0.0, min(1.0, number)) if math.isfinite(number) else 0.0

@lru_cache(maxsize=16)
def _keyword_patterns(keywords: Tuple[str, ...]):
    patterns = []
    for keyword in keywords:
        term = re.escape(keyword).replace(r"\ ", r"\s+")
        if re.search(r"[가-힣]", keyword):
            pattern = rf"(?<![가-힣A-Za-z0-9_]){term}{KOREAN_ENDINGS}(?![가-힣A-Za-z0-9_])"
        else:
            pattern = rf"(?<![A-Za-z0-9_]){term}(?![A-Za-z0-9_])"
        prefilter = keyword.casefold() if not any(char.isspace() for char in keyword) else None
        patterns.append((keyword, re.compile(pattern), prefilter))
    return tuple(patterns)

def _keyword_hits(text: str, keywords: Iterable[str]) -> List[str]:
    hits = []
    lowered = (text or "").casefold()
    for keyword, pattern, prefilter in _keyword_patterns(tuple(sorted(keywords))):
        if prefilter is not None and prefilter not in lowered:
            continue
        if pattern.search(lowered):
            hits.append(keyword)
    return hits

def _normalize_url(value: Any) -> str:
    if isinstance(value, dict):
        value = value.get("url") or value.get("href") or ""
    if not isinstance(value, str):
        return ""
    url = value.strip().strip("<>\"'").rstrip(".,;!?")
    # Trim surrounding prose punctuation without destroying IPv6 host brackets.
    for opening, closing in (("(", ")"), ("[", "]"), ("{", "}")):
        while url.endswith(closing) and url.count(closing) > url.count(opening):
            url = url[:-1].rstrip(".,;!?")
    if url.lower().startswith("www."):
        url = "https://" + url
    return url

def _extract_urls(text: str) -> List[str]:
    return list(dict.fromkeys(_normalize_url(value) for value in URL_REGEX.findall(text or "")))

def _all_urls(text: str, meta: Dict[str, Any]) -> List[str]:
    values = list(_extract_urls(text))
    links = meta.get("links") or meta.get("urls") or []
    if isinstance(links, (str, dict)):
        links = [links]
    if isinstance(links, (list, tuple)):
        values.extend(_normalize_url(link) for link in links)
    return list(dict.fromkeys(url for url in values if url))

def _url_parts(url: str):
    try:
        parsed = urlsplit(url)
        host = (parsed.hostname or "").casefold().rstrip(".")
        port = parsed.port  # Property access validates malformed/range-invalid ports.
        return parsed, host, port, None
    except (TypeError, ValueError) as exc:
        return None, "", None, str(exc)

def _is_ip(host: str) -> bool:
    try:
        ipaddress.ip_address(host)
        return True
    except ValueError:
        return False

def _score_urls(urls: Iterable[str]) -> List[RuleSignal]:
    signals: List[RuleSignal] = []
    seen: set[tuple[str, str]] = set()
    def add(name: str, score: float, reason: str, subject: str):
        key = name, subject
        if key not in seen:
            seen.add(key)
            signals.append(RuleSignal(name, score, reason))
    for url in urls:
        parsed, host, port, error = _url_parts(url)
        if error:
            add("malformed_url", 0.15, "링크 주소의 호스트 또는 포트 형식이 잘못되었습니다.", url.casefold())
            continue
        scheme = (parsed.scheme or "").casefold()
        if scheme in {"mailto", "tel", "cid"} or url.startswith("#"):
            continue
        if scheme and scheme not in {"http", "https"}:
            add("bad_scheme", 0.35, f"웹 링크가 아닌 주소 형식입니다: {scheme}", scheme)
        if not host:
            if scheme in {"http", "https"}:
                add("malformed_url", 0.15, "링크에 유효한 호스트가 없습니다.", url.casefold())
            continue
        if port is not None and port not in {80, 443}:
            add("non_standard_port", 0.2, f"일반적이지 않은 링크 포트입니다: {port}", f"{host}:{port}")
        if any(label.startswith("xn--") for label in host.split(".")) or any(ord(char) > 127 for char in host):
            add("punycode", 0.4, f"국제화 도메인은 철자와 실제 발신자를 확인해야 합니다: {host}", host)
        if _is_ip(host):
            add("ip_literal", 0.25, f"도메인 대신 IP 주소를 사용하는 링크입니다: {host}", host)
        if any(host.endswith(tld) for tld in SUSPICIOUS_TLDS):
            add("suspicious_tld", 0.3, f"추가 확인이 필요한 최상위 도메인입니다: {host}", host)
        if host.count("-") >= 2:
            add("hyphenated_host", 0.15, f"하이픈이 많은 링크 호스트입니다: {host}", host)
        if host in SHORTENER_HOSTS:
            add("url_shortener", 0.25, f"최종 목적지를 숨기는 단축 링크입니다: {host}", host)
    return signals

def _is_trusted_host(host: str) -> bool:
    return any(host == domain or host.endswith("." + domain) for domain in TRUSTED_DOMAINS)

def _score_untrusted_link_with_action(text: str, urls: Iterable[str]) -> List[RuleSignal]:
    actions = _keyword_hits(text, ACTION_LINK_KEYWORDS)
    if not actions:
        return []
    for url in urls:
        parsed, host, _, error = _url_parts(url)
        if not error and parsed.scheme.casefold() in {"http", "https"} and host and not _is_trusted_host(host):
            return [RuleSignal("untrusted_link_action", 0.35, f"확인되지 않은 도메인 {host}의 링크와 행동 요청 '{actions[0]}'이 함께 있습니다.")]
    return []

def _score_auth(meta: Dict[str, Any]) -> List[RuleSignal]:
    auth = meta.get("auth_results") or {}
    if not isinstance(auth, dict):
        return []
    return [RuleSignal("auth_fail", 0.25, f"제공된 헤더의 {protocol.upper()} 인증이 실패했습니다.")
            for protocol in ("spf", "dkim", "dmarc") if auth.get(protocol + "_pass") is False]

def _score_attachments(meta: Dict[str, Any]) -> List[RuleSignal]:
    signals, seen = [], set()
    for attachment in meta.get("attachments") or []:
        if not isinstance(attachment, dict):
            continue
        filename = str(attachment.get("filename") or "").casefold().strip()
        if filename not in seen and any(filename.endswith(ext) for ext in SUSPICIOUS_EXTENSIONS):
            seen.add(filename)
            signals.append(RuleSignal("dangerous_attachment", 0.3, f"실행 또는 매크로·압축 위험을 확인할 첨부 파일입니다: {filename}"))
    return signals

def _score_keywords(text: str) -> List[RuleSignal]:
    hits = _keyword_hits(text, URGENT_KEYWORDS)
    return [RuleSignal("urgent_language", min(0.3, 0.1 + 0.05 * len(hits)), f"긴급 행동 또는 계정·금전 관련 표현입니다: {', '.join(hits)}")] if hits else []

def _looks_like_receipt(text: str) -> bool:
    return len(_keyword_hits(text, RECEIPT_KEYWORDS)) >= 2

def _has_action_request(text: str) -> bool:
    return bool(_keyword_hits(text, ACTION_KEYWORDS))

def compute_rule_score(text: str, meta: Dict[str, Any] | None = None) -> Tuple[float, List[RuleSignal]]:
    if not isinstance(text, str):
        raise TypeError("분석할 본문은 문자열이어야 합니다.")
    if meta is not None and not isinstance(meta, dict):
        raise TypeError("메일 메타데이터는 객체여야 합니다.")
    meta = meta or {}
    urls = _all_urls(text, meta)
    signals = (_score_urls(urls) + _score_untrusted_link_with_action(text, urls)
               + _score_auth(meta) + _score_attachments(meta) + _score_keywords(text))
    sender = meta.get("from_header") or meta.get("sender") or meta.get("from")
    if sender:
        _, address = parseaddr(str(sender))
        host = address.rsplit("@", 1)[-1].casefold()
        if any(host.endswith(tld) for tld in (".ru", ".tk", ".top")):
            signals.append(RuleSignal("sender_domain", 0.1, f"발신자 도메인을 추가 확인해야 합니다: {host}"))
    score = clamp_confidence(sum(signal.score for signal in signals))
    if (_looks_like_receipt(text) and not urls and not _has_action_request(text)
            and not any(signal.name in {"auth_fail", "dangerous_attachment"} for signal in signals)):
        score = clamp_confidence(score - 0.25)
    return score, signals

def fuse_scores(model_phish_prob: float, rule_score: float, model_weight: float = 0.7) -> float:
    weight = clamp_confidence(model_weight)
    return clamp_confidence(weight * clamp_confidence(model_phish_prob) + (1.0 - weight) * clamp_confidence(rule_score))

def decide_label(fused_score: float, threshold: float = 0.5) -> str:
    number = float(threshold)
    if not math.isfinite(number) or not 0 <= number <= 1:
        raise ValueError("판정 임계값은 0~1 사이의 유한한 숫자여야 합니다.")
    return "phishing" if clamp_confidence(fused_score) + 1e-12 >= number else "normal"
