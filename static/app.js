(() => {
  'use strict';

  const $ = (id) => document.getElementById(id);
  const tokenKey = 'phishguard_api_token';
  const state = {
    settings: null, history: [], metrics: null, token: '', pending: null,
    sequence: 0, historySequence: 0, detail: null, restoreFocus: null, restoreFocusId: null, restoreFocusSurface: null,
    gmailMessages: [], gmailLoading: false, settingsSaving: false,
  };
  try { state.token = sessionStorage.getItem(tokenKey) || ''; } catch (_) { /* Session-only memory remains available. */ }
  $('api-token').value = state.token;

  class ApiError extends Error {
    constructor(message, status = 0) { super(message); this.name = 'ApiError'; this.status = status; }
  }
  const errorMessage = (error) => error instanceof Error ? error.message : '요청을 처리하지 못했습니다. 다시 시도해 주세요.';
  function messageFromData(data) {
    if (typeof data?.detail === 'string') return data.detail;
    if (typeof data?.error === 'string') return data.error;
    if (typeof data?.error?.message === 'string') return data.error.message;
    if (typeof data?.message === 'string') return data.message;
    return '';
  }
  async function api(path, { method = 'GET', body, signal, timeout = 12000 } = {}) {
    const controller = new AbortController();
    let timedOut = false;
    const abort = () => controller.abort();
    if (signal?.aborted) controller.abort();
    else signal?.addEventListener('abort', abort, { once: true });
    const timer = setTimeout(() => { timedOut = true; controller.abort(); }, timeout);
    const headers = { Accept: 'application/json' };
    if (state.token) headers['X-API-Key'] = state.token;
    if (body !== undefined) headers['Content-Type'] = 'application/json';
    try {
      const response = await fetch(path, {
        method, headers, body: body === undefined ? undefined : JSON.stringify(body),
        signal: controller.signal, credentials: 'same-origin', cache: 'no-store',
      });
      let data;
      try { data = await response.json(); }
      catch (_) { throw new ApiError('서버 응답 형식을 읽을 수 없습니다. 서버 상태를 확인한 뒤 다시 시도해 주세요.', response.status); }
      if (!response.ok) {
        if (response.status === 401 || response.status === 403) {
          throw new ApiError('서버 접근이 거부되었습니다. 설정의 API 토큰을 확인하고 직접 다시 시도해 주세요.', response.status);
        }
        const detail = messageFromData(data);
        if (response.status === 409) throw new ApiError(detail || '다른 요청에서 설정이 변경되었습니다. 저장된 값을 다시 읽고 확인해 주세요.', 409);
        throw new ApiError(detail || `요청을 처리하지 못했습니다. (상태 ${response.status})`, response.status);
      }
      return data;
    } catch (error) {
      if (timedOut) throw new ApiError('응답 대기 시간이 지났습니다. 입력은 유지됩니다. 서버에서 처리·저장이 끝났을 수 있으니 이력을 확인한 뒤 다시 시도해 주세요.');
      if (signal?.aborted) throw new DOMException('응답 대기 취소', 'AbortError');
      if (error instanceof ApiError) throw error;
      if (error?.name === 'AbortError') throw error;
      throw new ApiError('서버에 연결할 수 없습니다. 로컬 서버 실행 상태를 확인한 뒤 다시 시도해 주세요. 입력은 유지됩니다.');
    } finally {
      clearTimeout(timer);
      signal?.removeEventListener('abort', abort);
    }
  }

  function node(tag, className, text) {
    const element = document.createElement(tag);
    if (className) element.className = className;
    if (text !== undefined) element.textContent = String(text);
    return element;
  }
  function icon(name) {
    const svg = document.createElementNS('http://www.w3.org/2000/svg', 'svg');
    svg.classList.add('icon'); svg.setAttribute('aria-hidden', 'true');
    const use = document.createElementNS('http://www.w3.org/2000/svg', 'use');
    use.setAttribute('href', `#icon-${name}`); svg.append(use); return svg;
  }
  function notice(id, text) { $(id).textContent = text || ''; $(id).hidden = !text; }
  function finite(value) { return typeof value === 'number' && Number.isFinite(value); }
  function risk(row) {
    const value = finite(row.risk_score) ? row.risk_score : row.confidence;
    return finite(value) ? Math.max(0, Math.min(1, value)) : null;
  }
  function riskText(row) { const value = risk(row); return value === null ? '—' : String(Math.round(value * 100)); }
  function verdict(row) { return row.label === 'phishing' ? '검토 필요' : row.label === 'normal' ? '낮은 위험' : '판정 정보 없음'; }
  function pill(row) { return node('span', `pill ${row.label === 'phishing' ? 'pill-amber' : row.label === 'normal' ? 'pill-teal' : 'pill-muted'}`, verdict(row)); }
  function dateText(value) {
    if (!value) return '날짜 정보 없음';
    const parsed = new Date(value);
    if (Number.isNaN(parsed.getTime())) return String(value);
    return new Intl.DateTimeFormat('ko-KR', { month: 'short', day: 'numeric', hour: '2-digit', minute: '2-digit' }).format(parsed);
  }
  function backendText(backend) { return backend === 'rules' ? '로컬 규칙' : backend === 'hf' ? '로컬 모델 + 규칙' : backend ? String(backend) : '분석 방식 정보 없음'; }
  function signals(row) {
    if (Array.isArray(row.signals) && row.signals.length) {
      return row.signals.map((item) => typeof item === 'string' ? item : item?.reason || item?.name || '').filter(Boolean);
    }
    return Array.isArray(row.rule_signals) ? row.rule_signals.filter((item) => typeof item === 'string') : Array.isArray(row.reasons) ? row.reasons.filter((item) => typeof item === 'string') : [];
  }
  function renderSignals(container, row) {
    const items = signals(row);
    container.replaceChildren(...(items.length ? items : ['주요 의심 신호가 확인되지 않았습니다. 발신자와 요청 내용은 별도로 확인하세요.']).map((text) => node('li', '', text)));
  }
  function feedback(row) { return row.feedback || row.gpt_feedback || '발신자와 요청 내용을 공식 경로로 확인하세요.'; }
  function warningsText(items) { return Array.isArray(items) ? items.filter((item) => typeof item === 'string').join('\n') : ''; }
  function empty(container, title, description, action = false) {
    const box = node('div', 'empty-state compact');
    if (title) box.append(node('h4', '', title));
    box.append(node('p', '', description));
    if (action) { const link = node('a', 'button button-outline', '메일 분석 시작'); link.href = '#analyze'; box.append(link); }
    container.replaceChildren(box);
  }

  const pageTitles = { overview: '개요', analyze: '메일 분석', history: '분석 이력', settings: '설정' };
  function navigate() {
    const requested = location.hash.slice(1);
    const view = Object.hasOwn(pageTitles, requested) ? requested : 'overview';
    document.querySelectorAll('[data-view]').forEach((element) => { element.hidden = element.dataset.view !== view; });
    document.querySelectorAll('[data-nav]').forEach((element) => {
      if (element.dataset.nav === view) element.setAttribute('aria-current', 'page');
      else element.removeAttribute('aria-current');
    });
    $('page-title').textContent = pageTitles[view];
    document.title = `PhishGuard · ${pageTitles[view]}`;
  }
  window.addEventListener('hashchange', navigate);
  document.querySelector('.skip-link').addEventListener('click', (event) => { event.preventDefault(); $('main-content').focus(); });
  navigate();

  async function refreshHealth() {
    try {
      const health = await api('/api/health');
      const online = health.status === 'ok' || health.status === 'healthy';
      $('connection-dot').classList.toggle('online', online);
      $('connection-dot').classList.toggle('offline', !online);
      $('header-dot').classList.toggle('online', online);
      $('header-dot').classList.toggle('offline', !online);
      $('connection-label').textContent = online ? '로컬 서버 연결됨' : '서버 상태 확인 필요';
      $('header-status').textContent = online ? '로컬 서버 연결됨' : '상태 확인 필요';
      $('backend-label').textContent = `${backendText(health.backend)} 분석 환경입니다.`;
      $('mode-description').textContent = health.backend === 'rules'
        ? '기본 모드는 링크·표현·첨부 파일명·인증 결과를 규칙으로 분석합니다. 메일을 외부 AI 서비스로 전송하지 않습니다.'
        : '서버에서 선택한 로컬 모델과 규칙을 사용합니다. 위험지표와 개별 근거를 함께 확인하세요.';
      $('cost-label').textContent = health.external_api_calls === false ? '유료 외부 API 호출 없음' : '외부 호출 정책 확인 필요';
      return health;
    } catch (error) {
      $('connection-dot').className = 'status-dot offline'; $('header-dot').className = 'status-dot offline';
      $('connection-label').textContent = '서버 연결 확인 필요'; $('header-status').textContent = '연결 확인 필요';
      $('backend-label').textContent = '로컬 서버를 실행한 뒤 새로고침하세요.';
      $('mode-description').textContent = '서버에 연결되면 분석 방식을 표시합니다.';
      $('cost-label').textContent = '호출 정책 확인 불가';
      throw error;
    }
  }
  async function refreshMetrics() {
    state.metrics = await api('/metrics/summary');
    const metric = state.metrics;
    for (const [id, key] of [['metric-total', 'total'], ['metric-phishing', 'phishing'], ['metric-today', 'today']]) {
      $(id).textContent = finite(metric[key]) ? metric[key].toLocaleString('ko-KR') : '—';
    }
    $('metric-latency').textContent = finite(metric.avg_latency_ms) ? metric.avg_latency_ms.toFixed(1) : '—';
    const trend = Array.isArray(metric.trend) ? metric.trend.slice(-7) : [];
    if (!trend.length) empty($('trend-list'), '', '메일을 분석하면 날짜별 집계가 표시됩니다.');
    else {
      const maximum = Math.max(1, ...trend.map((entry) => finite(entry.total) ? entry.total : 0));
      $('trend-list').replaceChildren(...trend.map((entry) => {
        const row = node('div', 'trend-row');
        const label = node('span', '', String(entry.date || '').slice(5));
        const bar = node('progress'); bar.max = maximum; bar.value = finite(entry.total) ? entry.total : 0;
        bar.setAttribute('aria-label', `${entry.date || '날짜 없음'} 전체 ${entry.total || 0}건, 검토 필요 ${entry.phishing || 0}건`);
        row.append(label, bar, node('span', 'trend-count', `${entry.total || 0} / ${entry.phishing || 0}`));
        return row;
      }));
    }
    $('review-summary').textContent = finite(metric.reviewed) && finite(metric.disagreements)
      ? `직접 검토 ${metric.reviewed}건 · 분석 판정과 다른 기록 ${metric.disagreements}건. 정확도 통계가 아닙니다.`
      : '직접 검토 기록은 정확도 통계가 아닙니다.';
  }
  function renderRecent() {
    if (!state.history.length) { empty($('recent-history'), '아직 분석한 메일이 없습니다.', '합성 샘플이나 메일 본문으로 시작해 보세요.', true); return; }
    $('recent-history').replaceChildren(...state.history.slice(0, 4).map((row) => {
      const button = node('button', 'recent-row'); button.type = 'button';
      button.dataset.historyId = String(row.id || '');
      const marker = node('span', 'recent-icon'); marker.append(icon('mail'));
      const copy = node('span', 'recent-copy'); copy.append(node('strong', '', row.subject || row.title || '(제목 없음)'), node('small', '', dateText(row.timestamp || row.date)));
      button.append(marker, copy, pill(row)); button.addEventListener('click', () => openDetail(row, button)); return button;
    }));
  }
  function renderHistory() {
    const query = $('history-search').value.toLocaleLowerCase('ko-KR');
    const label = $('history-filter').value;
    const rows = state.history.filter((row) => {
      const text = [row.subject, row.title, row.sender, row.from, row.body].filter(Boolean).join(' ').toLocaleLowerCase('ko-KR');
      return (!query || text.includes(query)) && (label === 'all' || row.label === label);
    });
    $('history-count').textContent = `${state.history.length}건 중 ${rows.length}건 표시 · 최근 최대 200건`;
    if (!rows.length) {
      empty($('history-list'), state.history.length ? '검색 결과가 없습니다.' : '저장된 분석이 없습니다.', state.history.length ? '검색어나 판정 필터를 바꿔 보세요.' : '메일을 분석하면 판정과 근거가 여기에 저장됩니다.', !state.history.length);
      return;
    }
    $('history-list').replaceChildren(...rows.map((row) => {
      const button = node('button', 'history-row'); button.type = 'button'; button.dataset.historyId = String(row.id || '');
      const copy = node('span', 'history-subject');
      copy.append(node('strong', '', row.subject || row.title || '(제목 없음)'), node('small', '', [dateText(row.timestamp || row.date), row.sender || row.from, humanLabel(row) ? '직접 검토 기록 있음' : ''].filter(Boolean).join(' · ')));
      const score = node('span', 'history-score', `${riskText(row)} / 100`); score.append(node('small', '', '위험지표'));
      button.append(copy, pill(row), score, node('span', 'row-arrow', '↗'));
      button.addEventListener('click', () => openDetail(row, button)); return button;
    }));
  }
  async function refreshHistory() {
    const sequence = ++state.historySequence;
    try {
      const data = await api('/api/history?limit=200');
      if (sequence !== state.historySequence) return;
      state.history = Array.isArray(data.results) ? data.results : [];
      notice('history-error', ''); renderHistory(); renderRecent();
    } catch (error) {
      if (sequence !== state.historySequence) return;
      notice('history-error', errorMessage(error));
      $('history-count').textContent = state.history.length ? '새 이력을 불러오지 못했습니다. 마지막으로 불러온 기록을 표시합니다.' : '이력을 불러오지 못했습니다. 새로고침으로 다시 시도해 주세요.';
      if (!state.history.length) {
        empty($('history-list'), '이력을 불러오지 못했습니다.', '위 안내를 확인한 뒤 새로고침해 주세요.');
        empty($('recent-history'), '', '이력 연결을 확인하지 못했습니다. 새로고침으로 다시 시도해 주세요.');
      }
      throw error;
    }
  }
  async function refreshOverview() {
    $('overview-refresh').disabled = true;
    const results = await Promise.allSettled([refreshHealth(), refreshMetrics(), refreshHistory()]);
    const failure = results.find((item) => item.status === 'rejected');
    notice('app-alert', failure ? errorMessage(failure.reason) : '');
    $('overview-refresh').disabled = false;
  }
  $('overview-refresh').addEventListener('click', refreshOverview);
  $('history-refresh').addEventListener('click', async () => {
    $('history-refresh').disabled = true;
    await Promise.allSettled([refreshHistory(), refreshMetrics()]);
    $('history-refresh').disabled = false;
  });
  $('history-search').addEventListener('input', renderHistory);
  $('history-filter').addEventListener('change', renderHistory);

  const samples = {
    safe: { title: '[합성 샘플] 다음 주 팀 회의 안내', sender: 'team@example.com', body: '안녕하세요. 다음 주 화요일 오전 10시에 팀 회의가 있습니다. 회의실은 기존과 같습니다. 별도 준비물은 없습니다. 추가할 안건이 있으면 회신해 주세요. 감사합니다.', attachments: '', auth: 'pass' },
    phishing: { title: '[합성 샘플] 계정 정지 전 즉시 확인하세요', sender: 'security@example.com', body: '긴급: 계정이 오늘 정지됩니다. 지금 로그인하고 비밀번호를 확인하세요.\nhttp://account-check.example.invalid/verify\n첨부된 보안 업데이트를 실행하면 계정 정지를 해제할 수 있습니다.', attachments: 'security-update.exe', auth: 'fail' },
    malformed: { title: '[합성 샘플] 포트가 잘못된 링크', sender: 'notice@example.com', body: '다음 링크의 형식과 위험 신호를 확인하는 테스트입니다.\nhttps://example.com:not-a-port/login\n이 주소는 실제로 열지 마세요.', attachments: '', auth: 'unknown' },
  };
  function updateBodyCount() { $('body-count').textContent = `${$('email-body').value.length.toLocaleString('ko-KR')}자`; }
  $('email-body').removeAttribute('required');
  $('email-body').addEventListener('input', updateBodyCount);
  document.querySelectorAll('[data-sample]').forEach((button) => button.addEventListener('click', () => {
    if (state.pending) return;
    const sample = samples[button.dataset.sample];
    $('email-title').value = sample.title; $('email-sender').value = sample.sender;
    $('email-body').value = sample.body; $('email-attachments').value = sample.attachments;
    ['spf', 'dkim', 'dmarc'].forEach((name) => { $(`auth-${name}`).value = sample.auth; });
    updateBodyCount(); notice('analyze-error', '');
    $('analyze-status').textContent = '합성 샘플을 입력했습니다. 분석 버튼을 눌러 확인하세요.';
  }));
  function updateBusy() {
    const busy = !!state.pending;
    $('analyze-submit').disabled = busy;
    $('analyze-submit').querySelector('span').textContent = state.pending?.kind === 'manual' ? '분석 중…' : '메일 분석하기';
    $('analyze-cancel').hidden = state.pending?.kind !== 'manual';
    $('gmail-cancel').hidden = state.pending?.kind !== 'gmail';
    $('gmail-load').disabled = busy || state.gmailLoading;
    $('gmail-analyze').disabled = busy || !document.querySelector('.gmail-row input:checked');
    $('analyze-form').querySelectorAll('input, textarea, select').forEach((field) => { field.disabled = busy; });
    document.querySelectorAll('.gmail-row input').forEach((field) => { field.disabled = busy; });
    $('token-apply').disabled = busy || state.settingsSaving;
    $('token-clear').disabled = busy || state.settingsSaving;
    document.querySelectorAll('[data-sample]').forEach((button) => { button.disabled = busy; });
    $('analyze-form').setAttribute('aria-busy', String(state.pending?.kind === 'manual'));
    $('gmail-section').setAttribute('aria-busy', String(state.pending?.kind === 'gmail' || state.gmailLoading));
  }
  function beginJob(kind) {
    if (state.pending) return null;
    const job = { id: ++state.sequence, kind, controller: new AbortController() };
    state.pending = job; updateBusy(); return job;
  }
  function currentJob(job) { return state.pending?.id === job.id; }
  function endJob(job) { if (currentJob(job)) { state.pending = null; updateBusy(); } }
  function cancelJob() {
    const job = state.pending; if (!job) return;
    state.pending = null; state.sequence += 1; job.controller.abort(); updateBusy();
    $(job.kind === 'gmail' ? 'gmail-status' : 'analyze-status').textContent = '응답 대기를 취소했습니다. 입력은 유지됩니다. 서버에서 처리가 끝나 이력에 저장될 수 있습니다.';
    $(job.kind === 'gmail' ? 'gmail-analyze' : 'analyze-submit').focus();
  }
  $('analyze-cancel').addEventListener('click', cancelJob);
  $('gmail-cancel').addEventListener('click', cancelJob);
  function renderResult(row) {
    $('result-empty').hidden = true; $('result-content').hidden = false;
    const warning = row.label === 'phishing';
    $('result-verdict').classList.toggle('is-warning', warning);
    $('result-progress').closest('.risk-block').classList.toggle('is-warning', warning);
    $('result-label').textContent = verdict(row);
    $('result-subject').textContent = row.subject || row.title || '(제목 없음)';
    $('result-caption').textContent = warning ? '의심 신호가 기준 이상입니다. 링크와 첨부 파일을 열기 전에 발신자를 공식 경로로 확인하세요.' : '주요 의심 신호가 기준 미만입니다. 이 결과만으로 메일의 안전을 보장할 수는 없습니다.';
    $('result-score').textContent = riskText(row); $('result-progress').value = risk(row) === null ? 0 : Math.round(risk(row) * 100);
    $('result-score-kind').textContent = row.score_kind === 'rule_risk_index' ? '규칙 위험지표' : '위험지표';
    $('result-threshold').textContent = finite(row.threshold) ? `검토 기준 ${Math.round(row.threshold * 100)} / 100 이상` : '검토 기준 정보 없음';
    renderSignals($('result-signals'), row); $('result-feedback').textContent = feedback(row);
    notice('result-warnings', warningsText(row.warnings));
    $('result-cache').hidden = !row.from_cache;
    $('result-timing').textContent = finite(row.latency_ms) ? `처리 시간 ${row.latency_ms.toFixed(1)} ms${row.from_cache ? ' · 저장된 결과' : ''}` : '처리 시간 정보 없음';
    $('result-backend').textContent = backendText(row.backend);
  }
  async function refreshAfterAnalysis() {
    const responses = await Promise.allSettled([refreshHistory(), refreshMetrics()]);
    const failed = responses.find((item) => item.status === 'rejected');
    if (failed) notice('app-alert', `분석은 완료했지만 화면 집계를 갱신하지 못했습니다. ${errorMessage(failed.reason)}`);
  }
  $('analyze-form').addEventListener('submit', async (event) => {
    event.preventDefault(); if (state.pending) return;
    notice('analyze-error', ''); $('analyze-status').textContent = '';
    const title = $('email-title').value.trim(), body = $('email-body').value.trim();
    if (!title && !body) { notice('analyze-error', '메일 제목이나 본문을 입력해 주세요.'); $('email-body').focus(); return; }
    const attachments = $('email-attachments').value.split(',').map((name) => name.trim()).filter(Boolean);
    if (attachments.length > 50 || attachments.some((name) => name.length > 260)) {
      notice('analyze-error', '첨부 파일명은 최대 50개, 각각 260자 이내로 입력해 주세요.'); $('email-attachments').focus(); return;
    }
    const auth_results = {};
    ['spf', 'dkim', 'dmarc'].forEach((name) => { const value = $(`auth-${name}`).value; auth_results[`${name}_pass`] = value === 'unknown' ? null : value === 'pass'; });
    const payload = { title, body, sender: $('email-sender').value.trim(), attachments, auth_results };
    const job = beginJob('manual'); if (!job) return;
    $('analyze-status').textContent = '의심 신호를 확인하고 있습니다…';
    try {
      const row = await api('/api/analyze', { method: 'POST', body: payload, signal: job.controller.signal });
      if (!currentJob(job)) return;
      renderResult(row); $('analyze-status').textContent = row.from_cache ? '같은 입력과 분석 기준의 저장된 결과를 재사용했습니다.' : '분석을 완료하고 이력에 저장했습니다.';
      endJob(job); void refreshAfterAnalysis();
    } catch (error) {
      if (!currentJob(job)) return;
      $('analyze-status').textContent = ''; notice('analyze-error', errorMessage(error));
    } finally { endJob(job); }
  });

  function humanLabel(row) { return row.human_label || row.review_label || null; }
  function setReviewLabel(row) {
    const label = humanLabel(row);
    $('detail-review-label').textContent = label ? `최근 기록: ${verdict({ label })}. 사용자 판단 기록이며 정확도 검증을 뜻하지 않습니다.` : '아직 직접 검토한 기록이 없습니다.';
  }
  function openDetail(row, trigger, preview = false) {
    state.detail = preview ? null : row; state.restoreFocus = trigger || document.activeElement;
    state.restoreFocusId = state.restoreFocus?.dataset?.historyId || null;
    state.restoreFocusSurface = state.restoreFocus?.classList?.contains('history-row') ? 'history' : state.restoreFocus?.classList?.contains('recent-row') ? 'recent' : null;
    $('detail-eyebrow').textContent = preview ? 'GMAIL 메일 미리보기' : '분석 상세';
    $('detail-title').textContent = row.subject || row.title || '(제목 없음)';
    $('detail-meta').textContent = [row.sender || row.from ? `발신자: ${row.sender || row.from}` : '', dateText(row.timestamp || row.date), row.id ? `기록 ID: ${row.id}` : ''].filter(Boolean).join('\n');
    $('detail-analysis').hidden = preview;
    $('detail-review').hidden = preview || !row.id;
    $('detail-label').textContent = verdict(row); $('detail-label').className = `pill ${row.label === 'phishing' ? 'pill-amber' : 'pill-teal'}`;
    $('detail-score').textContent = `위험지표 ${riskText(row)} / 100`;
    $('detail-feedback').textContent = feedback(row); renderSignals($('detail-signals'), row);
    $('detail-body').textContent = row.body || row.raw_body || '보관된 본문이 없습니다. 본문 보관 설정과 해당 기록을 확인해 주세요.';
    notice('review-error', ''); $('review-status').textContent = ''; setReviewLabel(row);
    $('review-normal').disabled = false; $('review-phishing').disabled = false;
    if (!$('detail-dialog').open) $('detail-dialog').showModal();
    $('detail-close').focus();
  }
  $('detail-close').addEventListener('click', () => $('detail-dialog').close());
  $('detail-dialog').addEventListener('click', (event) => { if (event.target === $('detail-dialog')) $('detail-dialog').close(); });
  $('detail-dialog').addEventListener('close', () => {
    if (state.restoreFocus?.isConnected) { state.restoreFocus.focus(); return; }
    const selector = state.restoreFocusSurface === 'history' ? '.history-row' : state.restoreFocusSurface === 'recent' ? '.recent-row' : null;
    if (selector && state.restoreFocusId) {
      const replacement = [...document.querySelectorAll(selector)].find((row) => row.dataset.historyId === state.restoreFocusId);
      if (replacement) { replacement.focus(); return; }
    }
    if (state.restoreFocusSurface === 'history') $('history-search').focus();
    else if (state.restoreFocusSurface === 'recent') $('overview-refresh').focus();
    else $('analyze-submit').focus();
  });
  async function review(label) {
    const row = state.detail; if (!row?.id || $('review-normal').disabled) return;
    $('review-normal').disabled = true; $('review-phishing').disabled = true; notice('review-error', '');
    $('review-status').textContent = '검토 기록을 저장하고 있습니다…';
    try {
      await api(`/api/history/${encodeURIComponent(row.id)}/review`, { method: 'POST', body: { label } });
      row.human_label = label;
      if (state.detail?.id === row.id) { setReviewLabel(row); $('review-status').textContent = '직접 검토 기록을 저장했습니다. 분석 기준은 변경하지 않았습니다.'; }
      await Promise.allSettled([refreshHistory(), refreshMetrics()]);
    } catch (error) {
      if (state.detail?.id === row.id) { $('review-status').textContent = ''; notice('review-error', errorMessage(error)); }
    } finally {
      if (state.detail?.id === row.id) { $('review-normal').disabled = false; $('review-phishing').disabled = false; }
    }
  }
  $('review-normal').addEventListener('click', () => review('normal'));
  $('review-phishing').addEventListener('click', () => review('phishing'));

  function settingsValues() { return { threshold: Number($('setting-threshold').value), max_results: Number($('setting-max-results').value), retain_body: $('setting-retain-body').checked }; }
  function updateDirty() {
    const draft = settingsValues();
    const dirty = !!state.settings && ['threshold', 'max_results', 'retain_body'].some((key) => draft[key] !== state.settings[key]);
    $('settings-dirty').hidden = !dirty; $('settings-save').disabled = !dirty || state.settingsSaving;
    $('settings-cancel').disabled = !dirty || state.settingsSaving;
    $('settings-fields').disabled = !state.settings || state.settingsSaving;
    $('token-apply').disabled = !!state.pending || state.settingsSaving;
    $('token-clear').disabled = !!state.pending || state.settingsSaving;
  }
  function renderSettings(preserveDraft = false) {
    const settings = state.settings; if (!settings) return;
    if (!preserveDraft) {
      $('setting-threshold').value = settings.threshold; $('setting-max-results').value = settings.max_results; $('setting-retain-body').checked = settings.retain_body;
    }
    $('settings-fields').disabled = false; $('settings-version').textContent = `저장 버전: ${settings.version || '정보 없음'} · 서버 재실행 후에도 저장된 설정을 사용합니다.`;
    $('gmail-limit').textContent = `목록 제한: 최대 ${settings.max_results}개`;
    $('analysis-privacy').textContent = settings.retain_body
      ? '제목·발신자·본문·판정·근거가 로컬 이력에 저장됩니다.'
      : '본문은 새 분석 이력에 보관하지 않습니다. 제목·발신자·판정·근거는 로컬에 저장됩니다.';
    notice('settings-warning', warningsText(settings.warnings)); updateDirty();
  }
  async function loadSettings(preserveDraft = false) {
    $('settings-reload').disabled = true;
    try { state.settings = await api('/api/settings'); renderSettings(preserveDraft); notice('settings-error', ''); }
    catch (error) { notice('settings-error', errorMessage(error)); throw error; }
    finally { $('settings-reload').disabled = false; }
  }
  $('settings-form').addEventListener('input', () => { updateDirty(); $('settings-status').textContent = ''; });
  $('settings-cancel').addEventListener('click', () => { renderSettings(); notice('settings-error', ''); $('settings-status').textContent = '변경을 취소했습니다. 저장된 설정을 유지합니다.'; });
  $('settings-reload').addEventListener('click', async () => { try { await loadSettings(); $('settings-status').textContent = '서버에 저장된 설정을 다시 읽었습니다.'; } catch (_) { $('settings-status').textContent = ''; } });
  $('settings-form').addEventListener('submit', async (event) => {
    event.preventDefault(); if (!state.settings || state.settingsSaving) return;
    const draft = settingsValues();
    if (!Number.isFinite(draft.threshold) || draft.threshold < .05 || draft.threshold > .95 || !Number.isInteger(draft.max_results) || draft.max_results < 1 || draft.max_results > 50) {
      notice('settings-error', '임계값은 0.05–0.95, 목록 개수는 1–50 사이의 정수로 입력해 주세요.'); return;
    }
    state.settingsSaving = true; updateDirty(); $('settings-reload').disabled = true; notice('settings-error', '');
    $('settings-status').textContent = '설정을 저장하고 있습니다…';
    try {
      state.settings = await api('/api/settings', { method: 'PUT', body: { ...draft, version: state.settings.version } });
      renderSettings(); $('settings-status').textContent = '설정을 저장했습니다. 새 분석에 적용됩니다.';
    } catch (error) { notice('settings-error', errorMessage(error)); $('settings-status').textContent = ''; }
    finally { state.settingsSaving = false; $('settings-reload').disabled = false; updateDirty(); }
  });
  function applyToken(clear = false) {
    const preserveDraft = !$('settings-dirty').hidden;
    state.token = clear ? '' : $('api-token').value.trim(); $('api-token').value = state.token;
    let stored = true;
    try { if (state.token) sessionStorage.setItem(tokenKey, state.token); else sessionStorage.removeItem(tokenKey); }
    catch (_) { stored = false; }
    $('token-status').textContent = clear ? '이 탭의 토큰을 지웠습니다.' : stored ? '이 탭의 세션에 적용했습니다. 연결을 다시 확인합니다.' : '현재 화면의 메모리에 적용했습니다. 새로고침하면 다시 입력해야 합니다.';
    void refreshOverview(); void loadSettings(preserveDraft).catch(() => {});
  }
  $('token-apply').addEventListener('click', () => applyToken());
  $('token-clear').addEventListener('click', () => applyToken(true));

  $('gmail-load').addEventListener('click', async () => {
    if (state.gmailLoading || state.pending) return;
    state.gmailLoading = true; updateBusy(); notice('gmail-error', '');
    $('gmail-status').textContent = '기존 Gmail 연결에서 목록을 읽고 있습니다…';
    try {
      const max = state.settings?.max_results;
      const data = await api(`/api/list_emails${max ? `?max_results=${encodeURIComponent(max)}` : ''}`, { timeout: 20000 });
      state.gmailMessages = Array.isArray(data.messages) ? data.messages : [];
      $('gmail-results').replaceChildren();
      $('gmail-list').replaceChildren(...state.gmailMessages.map((message, index) => {
        const row = node('div', 'gmail-row');
        const check = node('input'); check.type = 'checkbox'; check.id = `gmail-select-${index}`; check.value = message.id;
        check.addEventListener('change', updateBusy);
        const label = node('label'); label.htmlFor = check.id;
        label.append(node('strong', '', message.subject || '(제목 없음)'), node('small', '', [message.from || message.sender, dateText(message.date)].filter(Boolean).join(' · ')), node('p', '', message.snippet || '미리보기 내용 없음'));
        const preview = node('button', 'button button-subtle', '본문 보기'); preview.type = 'button';
        preview.addEventListener('click', async () => {
          preview.disabled = true; notice('gmail-error', '');
          try { const detail = await api(`/api/emails/${encodeURIComponent(message.id)}`, { timeout: 20000 }); openDetail(detail, preview, true); }
          catch (error) { notice('gmail-error', errorMessage(error)); }
          finally { preview.disabled = false; }
        });
        row.append(check, label, preview); return row;
      }));
      if (!state.gmailMessages.length) empty($('gmail-list'), '메일 목록이 비어 있습니다.', '기존 연결의 메일과 목록 조건을 확인해 주세요.');
      $('gmail-status').textContent = `${state.gmailMessages.length}개의 메일을 불러왔습니다. 분석할 메일을 선택하세요.`;
    } catch (error) { $('gmail-status').textContent = ''; notice('gmail-error', `${errorMessage(error)}\n기존 Gmail 자격 증명이 없으면 직접 메일 입력으로 분석할 수 있습니다.`); }
    finally { state.gmailLoading = false; updateBusy(); }
  });
  $('gmail-analyze').addEventListener('click', async () => {
    if (state.pending) return;
    const message_ids = [...document.querySelectorAll('.gmail-row input:checked')].map((input) => input.value);
    if (!message_ids.length) { notice('gmail-error', '분석할 메일을 선택해 주세요.'); return; }
    const job = beginJob('gmail'); if (!job) return;
    notice('gmail-error', ''); $('gmail-status').textContent = `선택한 ${message_ids.length}개의 메일을 분석하고 있습니다…`;
    try {
      const data = await api('/api/analyze_selected', { method: 'POST', body: { message_ids }, signal: job.controller.signal, timeout: 30000 });
      if (!currentJob(job)) return;
      const results = Array.isArray(data.results) ? data.results : [];
      $('gmail-results').replaceChildren(...results.map((row) => {
        const element = node('div', 'gmail-result-row');
        element.append(node('strong', '', row.subject || row.title || row.id || '(제목 없음)'));
        if (row.error) element.append(node('span', 'muted', typeof row.error === 'string' ? row.error : '이 메일의 분석에 실패했습니다.'));
        else {
          element.append(pill(row)); const show = node('button', 'button button-subtle', '근거 보기'); show.type = 'button';
          show.addEventListener('click', () => openDetail(row, show)); element.append(show);
        }
        return element;
      }));
      const failures = results.filter((row) => row.error).length;
      $('gmail-status').textContent = `분석 완료 ${results.length - failures}건${failures ? ` · 실패 ${failures}건. 선택을 유지했습니다. 확인 후 직접 다시 시도하세요.` : ''}`;
      notice('gmail-error', warningsText(data.warnings));
      endJob(job); void refreshAfterAnalysis();
    } catch (error) { if (currentJob(job)) { $('gmail-status').textContent = ''; notice('gmail-error', errorMessage(error)); } }
    finally { endJob(job); }
  });

  void Promise.allSettled([refreshOverview(), loadSettings()]);
})();
