import pytest

from scoring import compute_rule_score, fuse_scores, decide_label, clamp_confidence
from scoring import _keyword_hits, _keyword_patterns, ACTION_LINK_KEYWORDS

def names(signals):
    return {signal.name for signal in signals}

def test_rule_score_punycode_and_shortener():
    score, signals = compute_rule_score("Check https://xn--phish-9o0a.com/login and http://bit.ly/abcd")
    assert score > 0.4
    assert {"punycode", "url_shortener"} <= names(signals)

def test_rule_score_ip_and_port():
    score, signals = compute_rule_score("Click http://192.168.1.5:8080/reset")
    assert score >= 0.2
    assert {"ip_literal", "non_standard_port"} <= names(signals)

def test_rule_score_attachment_extension():
    score, signals = compute_rule_score("", {"attachments": [{"filename": "invoice.zip"}]})
    assert score >= 0.3
    assert "dangerous_attachment" in names(signals)

def test_rule_score_auth_failures():
    score, signals = compute_rule_score("", {"auth_results": dict.fromkeys(["spf_pass", "dkim_pass", "dmarc_pass"], False)})
    assert score >= 0.5
    assert len(signals) == 3

def test_rule_score_urgent_keywords():
    score, signals = compute_rule_score("URGENT: Your account is suspended, reset password immediately")
    assert score >= 0.15 and "urgent_language" in names(signals)

def test_fuse_scores_weighting():
    assert fuse_scores(0.9, 0.2, 0.8) == pytest.approx(0.76)
    assert fuse_scores(0.2, 0.9, 0.8) == pytest.approx(0.34)

def test_decide_label_threshold():
    assert decide_label(0.6, 0.5) == "phishing"
    assert decide_label(0.49, 0.5) == "normal"
    assert decide_label(0.3, 0.3) == "phishing"

@pytest.mark.parametrize("value,expected", [(float("nan"), 0), (float("inf"), 0), (-1, 0), (2, 1), (None, 0)])
def test_clamp_nonfinite(value, expected):
    assert clamp_confidence(value) == expected

def test_fusion_changes_decision_with_rules():
    fused = fuse_scores(0.25, 0.9, 0.4)
    assert fused > 0.25 and decide_label(fused, 0.5) == "phishing"

def test_duplicate_urls_and_same_host_do_not_inflate_score():
    single = compute_rule_score("https://bad.xyz")[0]
    assert single == pytest.approx(0.3)
    assert compute_rule_score("https://bad.xyz " * 10)[0] == single
    assert compute_rule_score("https://BAD.XYZ/a https://bad.xyz/b https://bad.xyz/b")[0] == single

@pytest.mark.parametrize("url", ["http://example.xyz:abc/verify", "https://example.com:99999", "http://[bad-address]", "https:///missing"])
def test_malformed_urls_are_explainable_not_crashes(url):
    score, signals = compute_rule_score(url)
    assert 0 <= score <= 1
    assert "malformed_url" in names(signals)

def test_ipv6_literal_is_an_ip_indicator():
    assert "ip_literal" in names(compute_rule_score("http://[2001:db8::1]:8080/")[1])

@pytest.mark.parametrize("text", ["bankruptcy statistics", "viewpoint presentation", "invoices and paymentology", "인증서 확인란 안내"])
def test_keyword_substrings_do_not_trigger(text):
    score, signals = compute_rule_score(text)
    assert score == 0 and not signals

def test_korean_action_with_particles():
    score, signals = compute_rule_score("긴급: 비밀번호를 즉시 확인하세요 https://unknown.example/login")
    assert score >= 0.35 and "untrusted_link_action" in names(signals)

def test_unknown_authentication_does_not_mean_failure():
    score, signals = compute_rule_score("회의 일정 안내", {"auth_results": {"spf_pass": None, "dkim_pass": True, "dmarc_pass": None}})
    assert score == 0 and "auth_fail" not in names(signals)

def test_auth_string_false_is_not_trusted_as_boolean_evidence():
    assert compute_rule_score("", {"auth_results": {"spf_pass": "false"}})[0] == 0

def test_receipt_with_link_cannot_bypass_rules():
    assert compute_rule_score("Invoice payment received")[0] == 0
    assert compute_rule_score("Invoice payment received https://bad.xyz")[0] >= 0.3

def test_html_metadata_links_are_scored():
    score, signals = compute_rule_score("로그인하세요", {"links": ["https://bad.xyz", {"href": "https://bad.xyz"}]})
    assert score >= 0.35 and "untrusted_link_action" in names(signals)

def test_trusted_domain_requires_hostname_boundary():
    safe = compute_rule_score("Verify https://accounts.google.com/")[1]
    spoof = compute_rule_score("Verify https://google.com.evil.example/")[1]
    assert "untrusted_link_action" not in names(safe)
    assert "untrusted_link_action" in names(spoof)

def test_duplicate_attachments_do_not_add_weight():
    assert compute_rule_score("", {"attachments": [{"filename": "x.exe"}, {"filename": "X.EXE"}]})[0] == 0.3

def test_score_is_bounded():
    score, _ = compute_rule_score("긴급 verify https://xn--bad.xyz http://bit.ly/a", {"auth_results": {"spf_pass": False}, "attachments": [{"filename": "x.exe"}]})
    assert score == 1
@pytest.mark.parametrize("url", ["http://[2001:db8::1]", "http://[2001:db8::1]/path", "(http://[2001:db8::1])"])
def test_ipv6_brackets_are_preserved_when_trimming_prose(url):
    score, signals = compute_rule_score(url)
    assert "ip_literal" in {signal.name for signal in signals}
    assert "malformed_url" not in {signal.name for signal in signals}

def test_cached_patterns_preserve_phrase_whitespace_and_korean_boundaries():
    _keyword_patterns.cache_clear()
    first = _keyword_hits("log   in 비밀번호를 확인하세요", ACTION_LINK_KEYWORDS)
    second = _keyword_hits("log\tin 비밀번호를 확인하세요", ACTION_LINK_KEYWORDS)
    assert first == second
    assert {"log in", "비밀번호", "확인"} <= set(first)
    assert _keyword_patterns.cache_info().hits >= 1
    assert "인증" not in _keyword_hits("인증서 안내", ACTION_LINK_KEYWORDS)
