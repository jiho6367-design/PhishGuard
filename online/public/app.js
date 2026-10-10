"use strict";
(() => {
  const $ = id => document.getElementById(id);
  const store = globalThis.PhishGuardStore;
  const VIEWS = { analysis: "메일 분석", history: "분석 이력", settings: "설정", about: "이용 안내" };
  let currentView = "analysis", ready = false, storageBusy = false, storageAvailable = true;
  let model = store.blank(), analysisToken = 0, controller = null, syncToken = 0;
  let currentResult = null, draftDirty = false, draftRevision = null, draftInvalidated = false, retryUntil = 0;
  let retryTimer = null, dialogReturn = null;
  let channel = null;
  try { if (globalThis.BroadcastChannel) channel = new BroadcastChannel("phishguard-online-storage"); } catch (_) { /* focus refresh remains available */ }

  function notice(message) { $("global-status").textContent = message; $("global-status").hidden = !message; }
  function storageWarning(message) { $("storage-error").textContent = message; $("storage-error").hidden = !message; }
  function errorText(error) {
    if (error?.code === "stale") return "같은 브라우저의 다른 탭에서 저장 상태가 바뀌었습니다. 새 상태를 확인한 뒤 다시 적용해 주세요.";
    if (error?.code === "corrupt") return "브라우저 저장 데이터 형식이 올바르지 않아 불러오지 못했습니다. 기존 데이터가 남아 있을 수 있습니다. 전체 삭제로 정리하거나 브라우저의 사이트 데이터 삭제 기능을 이용해 주세요.";
    return "브라우저 저장 또는 삭제를 완료하지 못했습니다. 차단·용량 부족·비공개 모드 설정을 확인하세요. 기존 데이터가 남아 있을 수 있습니다. 분석은 계속할 수 있으며 저장 성공으로 표시하지 않습니다.";
  }
  function broadcast(type = "changed") { try { channel?.postMessage({ type }); } catch (_) { /* explicit refresh is available */ } }
  function setControls() {
    $("analyze-button").disabled = !ready || controller !== null || storageBusy || Date.now() < retryUntil;
    $("input-fields").disabled = controller !== null;
    $("cancel-analysis").hidden = controller === null;
    $("clear-input").disabled = controller !== null;
    document.querySelectorAll("[data-sample]").forEach(button => { button.disabled = controller !== null; });
    ["apply-settings", "cancel-settings", "delete-all-settings", "delete-all-history", "refresh-history"].forEach(id => { $(id).disabled = !ready || storageBusy; });
    $("threshold-input").disabled = !ready || storageBusy;
    $("save-consent").disabled = !ready || storageBusy;
    $("confirm-delete").disabled = storageBusy;
    $("active-threshold").textContent = String(model.threshold);
  }
  function refreshBadges() {
    $("storage-badge").textContent = model.consent ? "브라우저 저장 ON" : "브라우저 저장 OFF · 이 탭만";
    $("history-mode").textContent = model.consent
      ? "저장 동의한 요약만 이 브라우저 프로필에 유지합니다. 같은 프로필의 탭은 저장소를 공유합니다."
      : "현재 탭에서만 표시합니다. 탭을 닫거나 새로고침하면 이력과 설정이 사라집니다.";
    setControls();
  }
  function restoreDraft() {
    $("threshold-input").value = String(model.threshold);
    $("threshold-output").textContent = `${model.threshold} / 100`;
    $("save-consent").checked = model.consent;
    draftRevision = model.revision; draftDirty = false; draftInvalidated = false;
  }
  function renderHistory() {
    model.history = store.prune(model.history);
    const list = $("history-list"); list.replaceChildren();
    $("history-count").textContent = String(model.history.length);
    if (!model.history.length) {
      const empty = document.createElement("div"); empty.className = "panel empty-state";
      const heading = document.createElement("h3"); heading.textContent = "분석 이력이 없습니다.";
      const text = document.createElement("p"); text.textContent = "분석한 요약이 여기에 표시됩니다. 저장은 설정에서 선택할 수 있습니다.";
      empty.append(heading, text); list.append(empty); return;
    }
    for (const row of model.history) {
      const card = document.createElement("article"); card.className = "history-card";
      const detail = document.createElement("div");
      const title = document.createElement("h3"); title.textContent = row.title || "제목 없음";
      const sender = document.createElement("p"); sender.textContent = row.sender || "발신자 미입력";
      const date = document.createElement("p"); date.textContent = `${new Date(row.timestamp).toLocaleString("ko-KR")} · 임계값 ${row.threshold}/100`;
      const signals = document.createElement("p"); signals.className = "history-signals";
      signals.textContent = row.signals.length ? row.signals.map(name => store.SIGNALS[name]).join(" · ") : "확인된 규칙 신호 없음";
      detail.append(title, sender, date, signals);
      const score = document.createElement("div"); score.className = "history-score";
      const number = document.createElement("strong"); number.textContent = `${Math.round(row.risk_score * 100)}`;
      if (row.label === "phishing") number.className = "danger";
      const label = document.createElement("small"); label.textContent = row.label === "phishing" ? "검토 필요 / 100" : "임계값 미만 / 100";
      const remove = document.createElement("button"); remove.type = "button"; remove.className = "button button-small button-quiet";
      remove.textContent = "요약 삭제"; remove.setAttribute("aria-label", `${row.title || "제목 없음"} 요약 삭제`);
      remove.disabled = storageBusy; remove.addEventListener("click", () => removeSummary(row.id));
      score.append(number, label, remove); card.append(detail, score); list.append(card);
    }
  }
  function cancelAnalysis(message = "분석을 취소했습니다. 이미 전송된 요청을 되돌리는 것은 아닙니다.") {
    if (!controller) return;
    analysisToken += 1; controller.abort(); controller = null;
    clearResult();
    $("analysis-status").textContent = message; setControls();
  }
  function clearResult() {
    currentResult = null; $("result-empty").hidden = false; $("result-content").hidden = true;
    $("result-reasons").replaceChildren(); $("result-warnings").replaceChildren();
    ["result-label", "result-feedback", "result-meta", "result-storage"].forEach(id => { $(id).textContent = ""; });
    $("result-score").textContent = "0"; $("score-bar").style.width = "0%";
  }
  function route(initial = false) {
    let next = location.hash.slice(1);
    if (!Object.hasOwn(VIEWS, next)) { next = "analysis"; history.replaceState(null, "", "#analysis"); }
    if (!initial && next !== currentView) cancelAnalysis("화면을 이동해 분석 대기를 취소했습니다. 늦게 도착한 결과는 이력에 추가하지 않습니다.");
    if (next !== currentView && next === "settings") { restoreDraft(); $("settings-status").textContent = ""; }
    currentView = next;
    document.querySelectorAll("[data-view]").forEach(section => { section.hidden = section.dataset.view !== next; });
    document.querySelectorAll("[data-nav]").forEach(link => { if (link.dataset.nav === next) link.setAttribute("aria-current", "page"); else link.removeAttribute("aria-current"); });
    $("page-title").textContent = VIEWS[next]; document.title = `PhishGuard · ${VIEWS[next]}`;
    if (next === "history") { renderHistory(); if (ready && !storageBusy) void syncStorage(); }
    if (!initial) $("page-title").focus({ preventScroll: true });
  }
  async function syncStorage(forceCleared = false) {
    const token = ++syncToken;
    try {
      const fresh = await store.load();
      if (token !== syncToken || storageBusy) return;
      const changed = fresh.revision !== model.revision || forceCleared;
      if (changed) {
        cancelAnalysis("다른 탭에서 저장 상태가 바뀌어 분석 대기를 취소했습니다. 새 상태를 확인한 뒤 다시 분석해 주세요.");
        if (!fresh.consent) { model = store.blank(); clearResult(); }
        else model = fresh;
        if (currentView === "settings" && draftDirty) {
          draftInvalidated = true;
          $("settings-status").textContent = "다른 탭에서 설정이 바뀌었습니다. 현재 초안은 유지됩니다. 변경 취소로 최신 설정을 확인해 주세요.";
        }
        else restoreDraft();
        notice(forceCleared ? "같은 프로필의 다른 탭에서 이력·설정을 삭제했습니다." : "브라우저 저장 상태를 새로 불러왔습니다.");
      } else if (fresh.consent) model = fresh;
      storageAvailable = true; storageWarning(""); refreshBadges(); renderHistory();
    } catch (error) { if (token === syncToken) { storageAvailable = false; storageWarning(errorText(error)); } }
  }
  function normalizedAuth(id) { return $(id).value === "pass" ? true : $(id).value === "fail" ? false : null; }
  function getRequest() {
    const item = { title: $("email-title").value.trim(), body: $("email-body").value.trim(), sender: $("email-sender").value.trim(),
      attachments: $("email-attachments").value.split(/\r?\n/).map(value => value.trim()).filter(Boolean),
      auth_results: { spf_pass: normalizedAuth("auth-spf"), dkim_pass: normalizedAuth("auth-dkim"), dmarc_pass: normalizedAuth("auth-dmarc") } };
    if (!item.title && !item.body) { $("email-body").focus(); throw new Error("empty"); }
    if (!$("send-consent").checked) { $("send-consent").focus(); throw new Error("consent"); }
    if (item.body.length > 16000 || item.title.length > 300 || item.sender.length > 500
        || item.attachments.length > 20 || item.attachments.some(name => name.length > 260)) throw new Error("size");
    const payload = JSON.stringify({ item, threshold: model.threshold });
    if (new TextEncoder().encode(payload).byteLength > 64 * 1024) throw new Error("size");
    return { item, payload, threshold: model.threshold };
  }
  function validResult(result, threshold) {
    // Python hostname.casefold can expand each of the 2,048 input UTF-16 units
    // into up to three units. Accept the full derived reasons, without slicing
    // away evidence from a response the rules API legitimately accepted.
    return result && typeof result === "object" && result.score_kind === "rule_risk_index" && result.backend === "rules"
      && result.persistent_storage === false && result.external_api_calls === false
      && Number.isFinite(result.risk_score) && result.risk_score >= 0 && result.risk_score <= 1
      && ["phishing", "normal"].includes(result.label) && result.threshold === threshold
      && result.label === (result.risk_score + 1e-12 >= threshold / 100 ? "phishing" : "normal")
      && Array.isArray(result.reasons) && result.reasons.length <= 300 && result.reasons.every(text => typeof text === "string" && text.length <= 6400)
      && Array.isArray(result.warnings) && result.warnings.length <= 300 && result.warnings.every(text => typeof text === "string" && text.length <= 6400)
      && Array.isArray(result.signals) && result.signals.length <= 300 && result.signals.every(signal => signal && typeof signal.name === "string" && signal.name.length <= 100
        && Number.isFinite(signal.score) && signal.score >= 0 && signal.score <= 1 && typeof signal.reason === "string" && signal.reason.length <= 6400)
      && typeof result.feedback === "string" && result.feedback.length <= 20000
      && Number.isFinite(result.request_ms) && result.request_ms >= 0;
  }
  function renderList(element, values, fallback) {
    element.replaceChildren();
    for (const text of values.length ? values : [fallback]) { const li = document.createElement("li"); li.textContent = text; element.append(li); }
  }
  function renderResult(result) {
    currentResult = result; $("result-empty").hidden = true; $("result-content").hidden = false;
    const danger = result.label === "phishing";
    $("result-label").textContent = danger ? "검토 필요" : "현재 임계값 미만";
    $("result-label").parentElement.classList.toggle("danger", danger);
    $("result-score").textContent = String(Math.round(result.risk_score * 100));
    $("score-bar").style.width = `${result.risk_score * 100}%`;
    renderList($("result-reasons"), result.reasons, "입력에서 알려진 규칙 신호를 발견하지 못했습니다. 안전을 보장하지 않습니다.");
    renderList($("result-warnings"), result.warnings, "발신자와 요청을 공식 경로로 다시 확인하세요.");
    $("result-feedback").textContent = result.feedback;
    $("result-meta").textContent = `임계값 ${result.threshold}/100 · 서버 규칙 처리 ${Math.round(result.request_ms)}ms · 외부 AI API 호출 없음`;
  }
  function startRetry(seconds) {
    const wait = Number.isFinite(seconds) ? Math.min(120, Math.max(1, Math.ceil(seconds))) : 30;
    retryUntil = Date.now() + wait * 1000;
    if (retryTimer) clearInterval(retryTimer);
    const update = () => {
      const left = Math.max(0, Math.ceil((retryUntil - Date.now()) / 1000));
      $("retry-analysis").disabled = left > 0;
      $("retry-analysis").textContent = left > 0 ? `${left}초 후 다시 분석` : "다시 분석";
      setControls(); if (!left) { clearInterval(retryTimer); retryTimer = null; }
    };
    update(); retryTimer = setInterval(update, 500);
  }
  function showAnalysisError(message) { $("analysis-error-text").textContent = message; $("analysis-error").hidden = false; }
  async function analyze(event) {
    event?.preventDefault();
    if (!ready || controller || storageBusy || Date.now() < retryUntil) return;
    let request;
    $("analysis-error").hidden = true;
    try { request = getRequest(); }
    catch (error) {
      const text = error.message === "empty" ? "제목 또는 본문을 입력해 주세요." : error.message === "consent"
        ? "분석 서버로 전송되는 범위를 읽고 전송 확인을 선택해 주세요." : "입력 길이 또는 첨부 파일명 개수를 줄여 주세요. 요청은 최대 64KiB입니다.";
      showAnalysisError(text); return;
    }
    const token = ++analysisToken, revision = model.revision, consent = model.consent;
    const startedFocused = document.activeElement === $("analyze-button") || document.activeElement === $("retry-analysis");
    const abort = new AbortController(); controller = abort; let timedOut = false;
    const timeout = setTimeout(() => { timedOut = true; abort.abort(); }, 20000);
    $("analysis-status").textContent = "입력을 서버로 전송해 규칙을 확인하고 있습니다…";
    clearResult(); setControls();
    try {
      const response = await fetch("/api/analyze", { method: "POST", headers: { "Content-Type": "application/json" },
        body: request.payload, signal: abort.signal, credentials: "same-origin", mode: "same-origin", redirect: "error",
        cache: "no-store", referrerPolicy: "no-referrer" });
      if (token !== analysisToken) return;
      if (!response.ok) {
        let seconds = Number(response.headers.get("Retry-After"));
        if (response.status === 429) {
          try { const error = await response.json(); if (!seconds && Number.isFinite(error.retry_after_seconds)) seconds = error.retry_after_seconds; } catch (_) { /* generic error only */ }
          startRetry(seconds || 30); throw new Error("rate");
        }
        throw new Error(response.status === 400 || response.status === 413 || response.status === 422 ? "input" : response.status === 403 ? "origin" : "service");
      }
      const result = await response.json();
      if (token !== analysisToken) return;
      if (!validResult(result, request.threshold)) throw new Error("service");
      renderResult(result);
      const row = store.summary(request.item, result);
      if (consent && model.consent) {
        try {
          const fresh = await store.append(row, revision, abort.signal);
          if (token !== analysisToken) return;
          model = fresh; storageAvailable = true; storageWarning(""); broadcast();
          $("result-storage").textContent = "요약을 이 브라우저에 저장했습니다. 본문·전체 URL·첨부 파일명·원문 근거는 제외했습니다.";
        } catch (error) {
          if (token !== analysisToken) return;
          if (error.code === "stale") { await syncStorage(); if (token !== analysisToken) return; $("result-storage").textContent = "저장 상태가 바뀌어 이 결과를 이력에 추가하지 않았습니다."; }
          else {
            storageAvailable = false; storageWarning(errorText(error));
            model.history = store.prune([row, ...model.history]);
            $("result-storage").textContent = "브라우저 저장에 실패했습니다. 이 결과 요약은 현재 탭에서만 표시합니다. 새로고침하면 사라질 수 있습니다.";
          }
        }
      } else {
        if (token !== analysisToken) return;
        model.history = store.prune([row, ...model.history]);
        $("result-storage").textContent = "브라우저에 저장하지 않았습니다. 이 요약은 현재 탭에서만 표시합니다.";
      }
      if (token !== analysisToken) return;
      $("analysis-status").textContent = "분석을 완료했습니다. 위험지수와 근거를 확인하세요.";
      refreshBadges(); renderHistory();
      const active = document.activeElement;
      if (startedFocused && currentView === "analysis" && (active === document.body || active === $("analyze-button") || active === $("retry-analysis"))) $("result-title").focus({ preventScroll: true });
    } catch (error) {
      if (token !== analysisToken) return;
      $("analysis-status").textContent = "";
      const message = error.message === "rate" ? "요청이 많습니다. 표시된 시간만큼 기다린 뒤 다시 분석해 주세요."
        : error.message === "input" ? "서버가 입력을 허용하지 않았습니다. 길이·파일명 개수·URL 개수를 확인하세요."
        : error.message === "origin" ? "이 주소에서는 분석 요청을 허용하지 않습니다. 공식 배포 주소와 운영 설정을 확인하세요."
        : timedOut ? "분석 응답을 기다리는 시간이 지났습니다. 이미 전송된 요청이 있을 수 있습니다. 잠시 후 다시 시도해 주세요."
        : "분석 응답을 받지 못했습니다. 연결 상태를 확인하고 잠시 후 다시 시도해 주세요.";
      showAnalysisError(message);
    } finally {
      clearTimeout(timeout);
      if (token === analysisToken) { controller = null; setControls(); }
    }
  }
  async function applySettings(event) {
    event.preventDefault(); if (!ready || storageBusy) return;
    if (draftInvalidated) { $("settings-status").textContent = "다른 탭에서 저장 상태가 바뀌었습니다. 변경 취소로 최신 설정을 확인한 뒤 다시 선택해 주세요."; return; }
    const threshold = Number($("threshold-input").value), consent = $("save-consent").checked;
    if (!store.validThreshold(threshold)) { $("settings-status").textContent = "임계값은 5~95 사이의 5 단위 값이어야 합니다."; return; }
    cancelAnalysis(); storageBusy = true; syncToken += 1; setControls();
    $("settings-status").textContent = "설정을 적용하고 있습니다…";
    try {
      if (!storageAvailable && !model.consent && !consent) {
        model.threshold = threshold; restoreDraft();
        $("settings-status").textContent = "현재 탭의 임계값을 적용했습니다. 브라우저에 남은 데이터의 삭제는 확인하지 못했습니다.";
      } else {
        const wasConsented = model.consent;
        const memoryHistory = model.history;
        model = await store.apply(threshold, consent, draftRevision);
        if (!consent && !wasConsented) model.history = memoryHistory;
        storageAvailable = true; storageWarning(""); restoreDraft();
        $("settings-status").textContent = consent ? "설정을 저장했습니다. 저장을 새로 켠 경우 앞으로 분석하는 요약부터 저장합니다."
          : wasConsented ? "브라우저의 이력·설정·저장 동의 표시를 삭제했습니다. 새 임계값은 현재 탭에서만 적용됩니다." : "현재 탭의 설정을 적용했습니다. 브라우저에는 저장하지 않습니다.";
        broadcast(consent ? "changed" : wasConsented ? "cleared" : "changed");
      }
      refreshBadges(); renderHistory();
    } catch (error) {
      storageAvailable = error.code === "stale" ? storageAvailable : false;
      storageWarning(errorText(error)); $("settings-status").textContent = "설정 저장·삭제를 완료하지 못했습니다. 적용된 설정은 유지됩니다.";
    } finally { storageBusy = false; setControls(); }
  }
  async function removeSummary(id) {
    if (storageBusy) return; storageBusy = true; syncToken += 1; setControls(); renderHistory();
    try {
      if (model.consent) { model = await store.remove(id, model.revision); broadcast(); storageWarning(""); }
      else model.history = model.history.filter(row => row.id !== id);
      notice("요약 하나를 삭제했습니다."); renderHistory();
    } catch (error) { storageWarning(errorText(error)); notice("요약 삭제를 완료하지 못했습니다."); }
    finally { storageBusy = false; setControls(); renderHistory(); }
  }
  function askDelete(event) {
    if (!ready || storageBusy) return;
    dialogReturn = event.currentTarget; $("delete-dialog").returnValue = "cancel";
    $("delete-dialog").showModal(); $("dismiss-delete").focus();
  }
  async function deleteAll() {
    cancelAnalysis("전체 삭제를 시작해 분석 대기를 취소했습니다.");
    storageBusy = true; syncToken += 1; setControls(); notice("브라우저의 이력·설정 삭제를 확인하고 있습니다…");
    try {
      model = await store.clear(); storageAvailable = true; storageWarning(""); clearResult(); restoreDraft();
      notice("브라우저에 저장된 이력·설정·저장 동의 표시를 삭제했습니다. 임계값은 20, 저장은 꺼짐입니다.");
      $("settings-status").textContent = "전체 삭제를 완료했습니다."; broadcast("cleared"); refreshBadges(); renderHistory();
    } catch (error) {
      storageWarning(errorText(error)); notice("전체 삭제를 완료하지 못했습니다. 기존 데이터와 설정이 남아 있을 수 있습니다.");
    } finally { storageBusy = false; setControls(); if (dialogReturn?.isConnected) dialogReturn.focus({ preventScroll: true }); }
  }
  function clearInputs() {
    $("analysis-form").reset(); $("body-count").textContent = "0 / 16,000자";
    $("analysis-status").textContent = "입력을 비웠습니다."; $("analysis-error").hidden = true; clearResult(); $("email-title").focus();
  }
  const samples = {
    normal: { title: "다음 주 회의 안내", sender: "team@example.com", body: "다음 주 수요일 오후 2시에 회의가 있습니다. 안건을 함께 확인해 주세요.", attachments: "안내.pdf" },
    suspicious: { title: "긴급: 계정이 정지됩니다", sender: "support@example.top", body: "지금 즉시 로그인하여 비밀번호를 확인하세요. https://account-example.top/login 으로 접속하지 않으면 계정이 정지됩니다.", attachments: "계정확인.exe" },
    malformed: { title: "링크 형식 확인", sender: "notice@example.com", body: "합성 URL 예제입니다: https://example.com:99999/check", attachments: "" }
  };
  $("analysis-form").addEventListener("submit", analyze);
  $("retry-analysis").addEventListener("click", analyze);
  $("cancel-analysis").addEventListener("click", () => { cancelAnalysis(); $("analyze-button").focus(); });
  $("clear-input").addEventListener("click", clearInputs);
  $("email-body").addEventListener("input", () => { $("body-count").textContent = `${$("email-body").value.length.toLocaleString("ko-KR")} / 16,000자`; });
  document.querySelectorAll("[data-sample]").forEach(button => button.addEventListener("click", () => {
    const sample = samples[button.dataset.sample]; if (!sample || controller) return;
    $("email-title").value = sample.title; $("email-sender").value = sample.sender; $("email-body").value = sample.body; $("email-attachments").value = sample.attachments;
    ["auth-spf", "auth-dkim", "auth-dmarc"].forEach(id => { $(id).value = "unknown"; });
    $("send-consent").checked = false; $("body-count").textContent = `${sample.body.length.toLocaleString("ko-KR")} / 16,000자`;
    clearResult(); $("analysis-error").hidden = true; $("analysis-status").textContent = "합성 예제를 채웠습니다. 전송 범위를 확인한 뒤 분석해 주세요.";
  }));
  $("settings-form").addEventListener("submit", applySettings);
  $("threshold-input").addEventListener("input", () => { draftDirty = true; $("threshold-output").textContent = `${$("threshold-input").value} / 100`; });
  $("save-consent").addEventListener("change", () => { draftDirty = true; });
  $("cancel-settings").addEventListener("click", () => { restoreDraft(); $("settings-status").textContent = "변경을 취소했습니다. 적용된 설정으로 돌아왔습니다."; });
  ["delete-all-history", "delete-all-settings"].forEach(id => { $(id).addEventListener("click", askDelete); });
  $("delete-dialog").addEventListener("close", () => {
    if ($("delete-dialog").returnValue === "delete") void deleteAll();
    else if (dialogReturn?.isConnected) dialogReturn.focus({ preventScroll: true });
  });
  $("delete-dialog").addEventListener("keydown", event => {
    if (event.key !== "Tab") return;
    const first = $("dismiss-delete"), last = $("confirm-delete");
    if (event.shiftKey && document.activeElement === first) { event.preventDefault(); (last.disabled ? first : last).focus(); }
    else if (!event.shiftKey && (document.activeElement === last || last.disabled)) { event.preventDefault(); first.focus(); }
  });
  $("refresh-history").addEventListener("click", () => { void syncStorage(); });
  window.addEventListener("hashchange", () => route());
  document.querySelector(".skip-link").addEventListener("click", event => { event.preventDefault(); $("main-content").focus(); });
  window.addEventListener("focus", () => { if (ready && !storageBusy) void syncStorage(); });
  document.addEventListener("visibilitychange", () => { if (!document.hidden && ready && !storageBusy) void syncStorage(); });
  function receiveStorageChange(event) { if (["changed", "cleared"].includes(event.data?.type)) void syncStorage(event.data.type === "cleared"); }
  if (channel) channel.onmessage = receiveStorageChange;
  window.addEventListener("pagehide", () => { cancelAnalysis(); try { channel?.close(); } catch (_) { /* no persistent access */ } channel = null; });
  window.addEventListener("pageshow", event => {
    if (!event.persisted) return;
    try { if (!channel && globalThis.BroadcastChannel) { channel = new BroadcastChannel("phishguard-online-storage"); channel.onmessage = receiveStorageChange; } } catch (_) { /* focus refresh fallback */ }
    if (ready && !storageBusy) void syncStorage();
  });
  route(true); setControls();
  store.load().then(fresh => { model = fresh; storageAvailable = true; })
    .catch(error => { storageAvailable = false; storageWarning(errorText(error)); })
    .finally(() => { ready = true; restoreDraft(); refreshBadges(); renderHistory(); });
})();
