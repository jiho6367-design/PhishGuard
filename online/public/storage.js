/* Browser-only, consented summary storage. No source mail or full URL is persisted. */
"use strict";
(() => {
  const DB_NAME = "phishguard-online-v1";
  const STORE = "preferences";
  const KEY = "state";
  const MAX_HISTORY = 50;
  const RETENTION_MS = 7 * 24 * 60 * 60 * 1000;
  const SIGNALS = Object.freeze({
    malformed_url: "URL 형식 오류", bad_scheme: "웹 링크 외 주소 형식",
    non_standard_port: "일반적이지 않은 포트", punycode: "국제화 도메인",
    ip_literal: "IP 주소 링크", suspicious_tld: "추가 확인이 필요한 도메인",
    hyphenated_host: "하이픈이 많은 도메인", url_shortener: "축약 링크",
    untrusted_link_action: "확인되지 않은 링크와 행동 요청", auth_fail: "메일 인증 실패",
    dangerous_attachment: "주의가 필요한 첨부 확장자", urgent_language: "긴급 행동 요구",
    sender_domain: "발신자 도메인 확인 필요"
  });
  let dbPromise;

  class StorageFault extends Error {
    constructor(code) { super(code); this.name = "StorageFault"; this.code = code; }
  }
  const validThreshold = value => Number.isInteger(value) && value >= 5 && value <= 95 && value % 5 === 0;
  const freshRevision = () => typeof globalThis.crypto?.randomUUID === "function"
    ? globalThis.crypto.randomUUID() : `${Date.now().toString(36)}-${Math.random().toString(36).slice(2)}`;
  const blank = () => ({ key: KEY, version: 1, consent: false, threshold: 20, revision: null, history: [] });
  const scrubText = (value, limit = 160) => String(value || "")
    .replace(/[\u0000-\u0008\u000b\u000c\u000e-\u001f\u007f\u200b-\u200d\u2060\ufeff]/g, "")
    .replace(/(?:https?|ftp|file|data|javascript|mailto|tel|cid):[^\s<>"']+/gi, "[URL 제외]")
    .replace(/www\.[^\s<>"']+/gi, "[URL 제외]")
    .slice(0, limit);
  function validateRecord(row) {
    const keys = ["id", "timestamp", "title", "sender", "risk_score", "label", "threshold", "signals"];
    if (!row || typeof row !== "object" || Array.isArray(row) || Object.keys(row).some(key => !keys.includes(key))) return false;
    return typeof row.id === "string" && row.id.length > 0 && row.id.length <= 100
      && Number.isSafeInteger(row.timestamp) && row.timestamp > 0 && row.timestamp <= Date.now() + 60000
      && typeof row.title === "string" && row.title.length <= 160 && row.title === scrubText(row.title)
      && typeof row.sender === "string" && row.sender.length <= 200 && row.sender === scrubText(row.sender, 200)
      && Number.isFinite(row.risk_score) && row.risk_score >= 0 && row.risk_score <= 1
      && ["phishing", "normal"].includes(row.label) && validThreshold(row.threshold)
      && row.label === (row.risk_score + 1e-12 >= row.threshold / 100 ? "phishing" : "normal")
      && Array.isArray(row.signals) && row.signals.length <= 13
      && row.signals.every(value => typeof value === "string" && Object.hasOwn(SIGNALS, value))
      && new Set(row.signals).size === row.signals.length;
  }
  function validateState(value) {
    if (!value || typeof value !== "object" || Array.isArray(value)) throw new StorageFault("corrupt");
    const keys = ["key", "version", "consent", "threshold", "revision", "history"];
    if (Object.keys(value).some(key => !keys.includes(key)) || value.key !== KEY || value.version !== 1
        || value.consent !== true || !validThreshold(value.threshold)
        || typeof value.revision !== "string" || value.revision.length < 1 || value.revision.length > 100
        || !Array.isArray(value.history) || value.history.length > MAX_HISTORY || !value.history.every(validateRecord)
        || new Set(value.history.map(row => row.id)).size !== value.history.length) throw new StorageFault("corrupt");
    return { ...value, history: value.history.map(row => ({ ...row, signals: [...row.signals] })) };
  }
  function prune(history, now = Date.now()) {
    return history.filter(row => validateRecord(row) && row.timestamp > now - RETENTION_MS)
      .sort((a, b) => b.timestamp - a.timestamp).slice(0, MAX_HISTORY);
  }
  function summary(item, result) {
    const row = { id: freshRevision(), timestamp: Date.now(), title: scrubText(item.title),
      sender: scrubText(item.sender, 200), risk_score: result.risk_score, label: result.label,
      threshold: result.threshold,
      signals: [...new Set((result.signals || []).map(signal => signal.name).filter(name => Object.hasOwn(SIGNALS, name)))].slice(0, 13) };
    if (!validateRecord(row)) throw new StorageFault("invalid");
    return row;
  }
  function open() {
    if (dbPromise) return dbPromise;
    dbPromise = new Promise((resolve, reject) => {
      if (!globalThis.indexedDB) { reject(new StorageFault("unavailable")); return; }
      let request;
      try { request = indexedDB.open(DB_NAME, 1); } catch (_) { reject(new StorageFault("unavailable")); return; }
      let settled = false;
      const deadline = setTimeout(() => { settled = true; reject(new StorageFault("blocked")); }, 4000);
      request.onupgradeneeded = () => { if (!request.result.objectStoreNames.contains(STORE)) request.result.createObjectStore(STORE, { keyPath: "key" }); };
      request.onblocked = () => { clearTimeout(deadline); settled = true; reject(new StorageFault("blocked")); };
      request.onerror = () => { clearTimeout(deadline); settled = true; reject(new StorageFault("unavailable")); };
      request.onsuccess = () => {
        clearTimeout(deadline);
        const db = request.result;
        if (settled) { db.close(); return; }
        db.onversionchange = () => { db.close(); dbPromise = undefined; };
        resolve(db);
      };
    }).catch(error => { dbPromise = undefined; throw error; });
    return dbPromise;
  }
  async function transact(operation, signal) {
    if (signal?.aborted) throw new StorageFault("cancelled");
    const db = await open();
    if (signal?.aborted) throw new StorageFault("cancelled");
    return new Promise((resolve, reject) => {
      let tx;
      try { tx = db.transaction(STORE, "readwrite"); } catch (_) { reject(new StorageFault("unavailable")); return; }
      const store = tx.objectStore(STORE);
      let output, failure;
      const onCancel = () => { failure = new StorageFault("cancelled"); try { tx.abort(); } catch (_) { /* completed transaction has already committed */ } };
      if (signal) signal.addEventListener("abort", onCancel, { once: true });
      const deadline = setTimeout(() => { failure = new StorageFault("blocked"); try { tx.abort(); } catch (_) { reject(failure); } }, 4000);
      const cleanup = () => { clearTimeout(deadline); if (signal) signal.removeEventListener("abort", onCancel); };
      tx.oncomplete = () => { cleanup(); resolve(output); };
      tx.onabort = tx.onerror = () => { cleanup(); reject(failure || new StorageFault(tx.error?.name === "QuotaExceededError" ? "quota" : "failed")); };
      const get = store.get(KEY);
      get.onsuccess = () => {
        try {
          const current = get.result === undefined ? blank() : validateState(get.result);
          const change = operation(current);
          output = change.value;
          if (change.clear) store.clear();
          else if (change.state) store.put(change.state);
        } catch (error) { failure = error; tx.abort(); }
      };
    });
  }
  function match(current, expected) { if (current.revision !== expected) throw new StorageFault("stale"); }
  async function load() {
    return transact(current => {
      if (!current.consent) return { value: current };
      const history = prune(current.history);
      if (history.length === current.history.length) return { value: current };
      const state = { ...current, history, revision: freshRevision() };
      return { value: state, state };
    });
  }
  async function apply(threshold, consent, expected) {
    if (!validThreshold(threshold) || typeof consent !== "boolean") throw new StorageFault("invalid");
    return transact(current => {
      match(current, expected);
      if (!consent) return { value: { ...blank(), threshold }, clear: true };
      // Explicit opt-in stores only future analyses; old session-only summaries are not copied.
      const state = { key: KEY, version: 1, consent: true, threshold,
        revision: freshRevision(), history: current.consent ? prune(current.history) : [] };
      return { value: state, state };
    });
  }
  async function append(row, expected, signal) {
    if (!validateRecord(row)) throw new StorageFault("invalid");
    return transact(current => {
      match(current, expected);
      if (!current.consent) throw new StorageFault("stale");
      const state = { ...current, revision: freshRevision(), history: prune([row, ...current.history.filter(old => old.id !== row.id)]) };
      return { value: state, state };
    }, signal);
  }
  async function remove(id, expected) {
    return transact(current => {
      match(current, expected);
      if (!current.consent) throw new StorageFault("stale");
      const state = { ...current, revision: freshRevision(), history: prune(current.history.filter(row => row.id !== id)) };
      return { value: state, state };
    });
  }
  async function clear() {
    // Corrupt records must also be deletable; intentionally bypass record parsing.
    const db = await open();
    return new Promise((resolve, reject) => {
      let tx;
      try { tx = db.transaction(STORE, "readwrite"); tx.objectStore(STORE).clear(); } catch (_) { reject(new StorageFault("failed")); return; }
      const deadline = setTimeout(() => { try { tx.abort(); } catch (_) { reject(new StorageFault("blocked")); } }, 4000);
      tx.oncomplete = () => { clearTimeout(deadline); resolve(blank()); };
      tx.onabort = tx.onerror = () => { clearTimeout(deadline); reject(new StorageFault("failed")); };
    });
  }
  globalThis.PhishGuardStore = Object.freeze({ DB_NAME, STORE, KEY, MAX_HISTORY, RETENTION_MS,
    SIGNALS, StorageFault, blank, validThreshold, scrubText, validateRecord, validateState, prune,
    summary, load, apply, append, remove, clear });
})();
