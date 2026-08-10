  
    function switchTab(tabId) {
      document.querySelectorAll('.nav-tab').forEach(t => {
        t.classList.remove('active');
        t.setAttribute('aria-selected', 'false');
      });
      document.querySelectorAll('.tab-content').forEach(c => c.classList.remove('active'));

      event.currentTarget.classList.add('active');
      event.currentTarget.setAttribute('aria-selected', 'true');
      document.getElementById('tab-' + tabId).classList.add('active');
    }

    function showToast(msg) {
      const toast = document.getElementById('toast');
      document.getElementById('toast-msg').innerText = msg;
      toast.classList.add('show');
      setTimeout(() => toast.classList.remove('show'), 3000);
    }

    // ── Button loading state ──
    const _btnOriginals = new Map();
    function btnLoading(selector, loading) {
      const btns = document.querySelectorAll(selector);
      btns.forEach(btn => {
        if (loading) {
          if (!_btnOriginals.has(btn)) {
            _btnOriginals.set(btn, { html: btn.innerHTML, disabled: btn.disabled });
          }
          btn.disabled = true;
          btn.innerHTML = '<span class="btn-spinner"></span> ' + (btn.getAttribute('data-loading') || 'Working…');
        } else {
          const orig = _btnOriginals.get(btn);
          if (orig) {
            btn.disabled = orig.disabled;
            btn.innerHTML = orig.html;
            _btnOriginals.delete(btn);
          }
        }
      });
    }

    // ── Form dirty tracking ──
    let _formSnapshots = null;
    const _TRACKED_IDS = [
      'cfg-base-url', 'cfg-cooldown', 'cfg-rpm', 'cfg-nim-proxy',
      'cfg-default-model', 'cfg-model-sonnet', 'cfg-model-opus', 'cfg-model-haiku', 'cfg-model-fable',
      'cfg-reasoning-pref', 'cfg-reasoning-sonnet', 'cfg-reasoning-opus', 'cfg-reasoning-haiku', 'cfg-reasoning-fable',
      'cfg-reasoning-budget',
      'cfg-web-search', 'cfg-web-fetch', 'cfg-search-provider', 'cfg-ddg-max', 'cfg-tavily-key', 'cfg-doubao-key',
      'cfg-quota-mock', 'cfg-prefix-detection', 'cfg-title-skip', 'cfg-suggestion-skip', 'cfg-filepath-mock',
      'cfg-port', 'cfg-host', 'cfg-proxy-token', 'cfg-timeout',
      'cfg-log-level', 'cfg-native-tool-clean',
    ];

    function captureFormSnapshot() {
      const snap = {};
      _TRACKED_IDS.forEach(id => {
        const el = document.getElementById(id);
        if (el) snap[id] = el.value;
      });
      _formSnapshots = snap;
    }

    function isFormDirty() {
      if (!_formSnapshots) return false;
      for (const id of _TRACKED_IDS) {
        const el = document.getElementById(id);
        if (el) {
          const current = _formSnapshots[id];
          if (current === undefined) continue;
          if (el.value !== current) return true;
        }
      }
      return false;
    }

    window.addEventListener('beforeunload', (e) => {
      if (isFormDirty()) {
        e.preventDefault();
        e.returnValue = '';
      }
    });

    async function restartService() {
      if (!confirm("Are you sure you want to restart the proxy service?")) {
        return;
      }
      btnLoading('.btn-restart', true);
      try {
        const res = await apiFetch('/admin/api/restart', { method: 'POST' });
        const data = await res.json();
        showToast(data.message || 'Restarting...');
        setTimeout(() => location.reload(), 2000);
      } catch (err) {
        showToast('Failed to send restart request');
        btnLoading('.btn-restart', false);
      }
    }

    function fillFromDefault(targetId) {
      const defaultModel = document.getElementById('cfg-default-model').value.trim();
      if (!defaultModel) { showToast('Default Model is currently empty'); return; }
      const el = document.getElementById(targetId);
      if (el) el.value = defaultModel;
    }

    function fillAllFromDefault() {
      const defaultModel = document.getElementById('cfg-default-model').value.trim();
      if (!defaultModel) {
        showToast('Default Model is currently empty');
        return;
      }
      ['cfg-model-sonnet', 'cfg-model-opus', 'cfg-model-haiku', 'cfg-model-fable'].forEach(id => {
        const el = document.getElementById(id);
        if (el) {
          el.value = defaultModel;
        }
      });
      updateAllRouteStatuses();
      showToast('Applied Default Model to all routes. Click "Save Changes" to save.');
    }

    function updateRouteStatus(routeId) {
      const input = document.getElementById('cfg-model-' + routeId);
      const dot = document.getElementById('status-' + routeId);
      if (!input || !dot) return;
      if (input.value.trim()) {
        dot.classList.add('configured');
        dot.classList.remove('inherited');
        dot.title = 'Route model configured';
      } else {
        dot.classList.add('inherited');
        dot.classList.remove('configured');
        dot.title = 'Inheriting Default Model';
      }
    }

    function updateAllRouteStatuses() {
      ['sonnet', 'opus', 'haiku', 'fable'].forEach(updateRouteStatus);
    }

    function getApiToken() {
      const field = document.getElementById('cfg-proxy-token');
      const fromField = field && field.value ? field.value.trim() : '';
      return fromField || (localStorage.getItem('nim_proxy_token') || '');
    }

    async function apiFetch(url, options = {}) {
      const token = getApiToken();
      const headers = Object.assign({}, options.headers || {});
      if (token && !headers['Authorization']) {
        headers['Authorization'] = 'Bearer ' + token;
      }
      const opts = Object.assign({}, options, { headers });
      const res = await fetch(url, opts);
      if (res.status === 401) {
        showToast('Authentication required — enter PROXY_API_TOKEN in Server Security');
      }
      return res;
    }

    async function fetchModels(notify = false) {
      try {
        const res = await apiFetch('/admin/api/models');
        const data = await res.json();
        const models = data.models || [];
        
        const datalist = document.getElementById('nim-models-list');
        datalist.innerHTML = '';

        models.forEach(m => {
          const opt = document.createElement('option');
          opt.value = m;
          datalist.appendChild(opt);
        });

        if (notify) {
          showToast(`Fetched ${models.length} models (${data.source})`);
        }
      } catch (err) {
        console.error('Failed to fetch models:', err);
        if (notify) showToast('Failed to refresh model list — using cached or fallback');
      }
    }

    async function loadConfig(fillForms = false) {
      try {
        const res = await apiFetch('/admin/api/config');
        if (res.status === 401) return;
        const data = await res.json();
        
        const stats = data.key_pool_stats || {};
        document.getElementById('stat-total').innerText = stats.total || 0;
        document.getElementById('stat-active').innerText = stats.active || 0;
        document.getElementById('stat-cooldown').innerText = stats.cooldown || 0;
        document.getElementById('stat-disabled').innerText = stats.disabled || 0;

        const tbody = document.getElementById('key-table-body');
        tbody.innerHTML = '';
        const now = Date.now() / 1000;

        (data.key_details || []).forEach(k => {
          let badge = '<span class="badge badge-success">Healthy / Active</span>';
          let cdText = '-';

          if (k.is_disabled) {
            badge = '<span class="badge badge-danger">Disabled (401)</span>';
          } else if (k.cooldown_until > now) {
            badge = '<span class="badge badge-warning">Cooldown</span>';
            cdText = Math.ceil(k.cooldown_until - now) + 's';
          }

          tbody.innerHTML += `
            <tr>
              <td>${k.key}</td>
              <td>${badge}</td>
              <td>${k.failure_count}</td>
              <td>${cdText}</td>
            </tr>
          `;
        });

        renderKeyList(data.key_details || []);
        submitPollingSnapshot(data);

        if (fillForms) {
          const s = data.settings || {};
          document.getElementById('cfg-base-url').value = s.NVIDIA_NIM_BASE_URL || '';
          document.getElementById('cfg-cooldown').value = s.KEY_COOLDOWN_SECONDS || 60;
          document.getElementById('cfg-rpm').value = s.RATE_LIMIT_RPM || 0;
          document.getElementById('cfg-nim-proxy').value = s.NVIDIA_NIM_PROXY || '';
          
          document.getElementById('cfg-default-model').value = s.NIM_DEFAULT_MODEL || '';
          document.getElementById('cfg-model-sonnet').value = s.MODEL_SONNET || '';
          document.getElementById('cfg-model-opus').value = s.MODEL_OPUS || '';
          document.getElementById('cfg-model-haiku').value = s.MODEL_HAIKU || '';
          document.getElementById('cfg-model-fable').value = s.MODEL_FABLE || '';

          document.getElementById('cfg-reasoning-pref').value = s.REASONING_PREFERENCE || 'client';
          document.getElementById('cfg-reasoning-sonnet').value = s.REASONING_SONNET || 'inherit';
          document.getElementById('cfg-reasoning-opus').value = s.REASONING_OPUS || 'inherit';
          document.getElementById('cfg-reasoning-haiku').value = s.REASONING_HAIKU || 'inherit';
          document.getElementById('cfg-reasoning-fable').value = s.REASONING_FABLE || 'inherit';
          document.getElementById('cfg-reasoning-budget').value = s.DEFAULT_REASONING_BUDGET || 4096;

          updateAllRouteStatuses();

          document.getElementById('cfg-web-search').value = s.ENABLE_WEB_SEARCH ? 'true' : 'false';
          document.getElementById('cfg-web-fetch').value = s.ENABLE_WEB_FETCH ? 'true' : 'false';
          document.getElementById('cfg-search-provider').value = s.WEB_SEARCH_PROVIDER || 'duckduckgo';
          document.getElementById('cfg-ddg-max').value = s.DUCKDUCKGO_MAX_RESULTS || 5;
          document.getElementById('cfg-tavily-key').value = s.TAVILY_API_KEY || '';
          document.getElementById('cfg-doubao-key').value = s.DOUBAO_SEARCH_API_KEY || '';

          document.getElementById('cfg-quota-mock').value = s.ENABLE_NETWORK_PROBE_MOCK ? 'true' : 'false';
          document.getElementById('cfg-prefix-detection').value = s.FAST_PREFIX_DETECTION ? 'true' : 'false';
          document.getElementById('cfg-title-skip').value = s.ENABLE_TITLE_GENERATION_SKIP ? 'true' : 'false';
          document.getElementById('cfg-suggestion-skip').value = s.ENABLE_SUGGESTION_MODE_SKIP ? 'true' : 'false';
          document.getElementById('cfg-filepath-mock').value = s.ENABLE_FILEPATH_EXTRACTION_MOCK ? 'true' : 'false';

          document.getElementById('cfg-port').value = s.PORT || 8000;
          document.getElementById('cfg-host').value = s.HOST || '0.0.0.0';
          document.getElementById('cfg-proxy-token').value = s.PROXY_API_TOKEN || '';
          document.getElementById('cfg-timeout').value = s.HTTP_TIMEOUT || 120.0;

          document.getElementById('cfg-log-level').value = s.LOG_LEVEL || 'INFO';
          document.getElementById('cfg-native-tool-clean').value = s.ENABLE_NATIVE_TOOL_CLEANING ? 'true' : 'false';

          if (s.NIM_DEFAULT_MODEL) {
            document.getElementById('play-model').value = s.NIM_DEFAULT_MODEL;
          }
          captureFormSnapshot();
        }

      } catch (err) {
        console.error('Failed to load config:', err);
        showToast('Failed to load configuration from server');
      }
    }

    async function saveConfig() {
      btnLoading('.btn-save-config', true);
      try {
      const payload = {
        nvidia_nim_base_url: document.getElementById('cfg-base-url').value,
        key_cooldown_seconds: parseFloat(document.getElementById('cfg-cooldown').value),
        rate_limit_rpm: parseInt(document.getElementById('cfg-rpm').value) || 0,
        nvidia_nim_proxy: document.getElementById('cfg-nim-proxy').value,
        nim_default_model: document.getElementById('cfg-default-model').value,
        model_sonnet: document.getElementById('cfg-model-sonnet').value,
        model_opus: document.getElementById('cfg-model-opus').value,
        model_haiku: document.getElementById('cfg-model-haiku').value,
        model_fable: document.getElementById('cfg-model-fable').value,
        reasoning_preference: document.getElementById('cfg-reasoning-pref').value,
        reasoning_sonnet: document.getElementById('cfg-reasoning-sonnet').value,
        reasoning_opus: document.getElementById('cfg-reasoning-opus').value,
        reasoning_haiku: document.getElementById('cfg-reasoning-haiku').value,
        reasoning_fable: document.getElementById('cfg-reasoning-fable').value,
        default_reasoning_budget: parseInt(document.getElementById('cfg-reasoning-budget').value),
        enable_web_search: document.getElementById('cfg-web-search').value === 'true',
        enable_web_fetch: document.getElementById('cfg-web-fetch').value === 'true',
        web_search_provider: document.getElementById('cfg-search-provider').value,
        duckduckgo_max_results: parseInt(document.getElementById('cfg-ddg-max').value),
        tavily_api_key: document.getElementById('cfg-tavily-key').value,
        doubao_search_api_key: document.getElementById('cfg-doubao-key').value,
        fast_prefix_detection: document.getElementById('cfg-prefix-detection').value === 'true',
        enable_network_probe_mock: document.getElementById('cfg-quota-mock').value === 'true',
        enable_title_generation_skip: document.getElementById('cfg-title-skip').value === 'true',
        enable_suggestion_mode_skip: document.getElementById('cfg-suggestion-skip').value === 'true',
        enable_filepath_extraction_mock: document.getElementById('cfg-filepath-mock').value === 'true',
        port: parseInt(document.getElementById('cfg-port').value),
        host: document.getElementById('cfg-host').value,
        proxy_api_token: document.getElementById('cfg-proxy-token').value,
        http_timeout: parseFloat(document.getElementById('cfg-timeout').value),
        log_level: document.getElementById('cfg-log-level').value,
        enable_native_tool_cleaning: document.getElementById('cfg-native-tool-clean').value === 'true',
      };

      localStorage.setItem('nim_proxy_token', payload.proxy_api_token || '');
      const res = await apiFetch('/admin/api/config', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload)
      });
      if (res.status === 401) return;
      const data = await res.json();
      showToast(data.message || 'Settings updated successfully');
      loadConfig(false);
      fetchModels();
      captureFormSnapshot(); // reset dirty marker after successful save
      } catch (err) {
        showToast('Save failed: ' + (err instanceof Error ? err.message : String(err)));
      } finally {
        btnLoading('.btn-save-config', false);
      }
    }

    async function resetKeyPool() {
      btnLoading('.btn-reset', true);
      try {
      const res = await apiFetch('/admin/api/keypool/reset', { method: 'POST' });
      if (res.status === 401) return;
      const data = await res.json();
      showToast(data.message);
      loadConfig(false);
      } finally {
        btnLoading('.btn-reset', false);
      }
    }

    // -------- Key List CRUD --------

    function renderKeyList(keyDetails) {
      const container = document.getElementById('key-list');
      if (!container) return;
      const now = Date.now() / 1000;
      let html = '';
      (keyDetails || []).forEach((k, idx) => {
        let statusCls = 'healthy';
        let statusText = 'Active';
        if (k.is_disabled) { statusCls = 'disabled-badge'; statusText = 'Disabled'; }
        else {
          const cdRem = k.cooldown_remaining !== undefined ? k.cooldown_remaining : (k.cooldown_until > now ? Math.ceil(k.cooldown_until - now) : 0);
          if (cdRem > 0) { statusCls = 'cooldown-badge'; statusText = 'Cooldown ' + cdRem + 's'; }
        }

        html += `
          <div class="key-item" id="key-row-${idx}">
            <span class="key-masked" id="key-display-${idx}">${k.key_masked || k.key}</span>
            <input class="key-edit-input" id="key-input-${idx}" style="display:none" value="">
            <span class="key-status-badge ${statusCls}">${statusText}</span>
            <div class="key-actions" id="key-actions-${idx}">
              <button class="btn-icon" onclick="startEditKey(${idx})" title="Edit">
                <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                  <path d="M11 4H4a2 2 0 0 0-2 2v14a2 2 0 0 0 2 2h14a2 2 0 0 0 2-2v-7"/>
                  <path d="M18.5 2.5a2.121 2.121 0 0 1 3 3L12 15l-4 1 1-4 9.5-9.5z"/>
                </svg>
              </button>
              <button class="btn-icon danger" onclick="deleteKey(${idx})" title="Delete">
                <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                  <polyline points="3 6 5 6 21 6"/><path d="M19 6v14a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2V6m3 0V4a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2"/><line x1="10" y1="11" x2="10" y2="17"/><line x1="14" y1="11" x2="14" y2="17"/>
                </svg>
              </button>
            </div>
            <div class="key-actions" id="key-edit-actions-${idx}" style="display:none">
              <button class="btn-icon confirm" onclick="saveEditKey(${idx})" title="Save">
                <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5">
                  <polyline points="20 6 9 17 4 12"/>
                </svg>
              </button>
              <button class="btn-icon danger" onclick="cancelEditKey(${idx})" title="Cancel">
                <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5">
                  <line x1="18" y1="6" x2="6" y2="18"/><line x1="6" y1="6" x2="18" y2="18"/>
                </svg>
              </button>
            </div>
          </div>`;
      });
      container.innerHTML = html;
    }

    let _keyCache = [];

    // Keep a snapshot of key_details so edit/delete can look up the raw key by index.
    function submitPollingSnapshot(data) {
      _keyCache = data.key_details || [];
    }

    async function addKey() {
      const input = document.getElementById('key-add-input');
      const val = input.value.trim();
      if (!val) return;
      btnLoading('.btn-add-key', true);
      try {
      const res = await apiFetch('/admin/api/keys', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({api_key: val})
      });
      if (res.status === 401) return;
      if (res.ok) {
        const data = await res.json();
        showToast(data.message);
        input.value = '';
        loadConfig(false);
      } else {
        const err = await res.json();
        showToast(err.detail || 'Failed to add key.');
      }
      } finally {
        btnLoading('.btn-add-key', false);
      }
    }

    function startEditKey(idx) {
      if (!_keyCache || !_keyCache[idx]) return;
      document.getElementById('key-display-' + idx).style.display = 'none';
      document.getElementById('key-actions-' + idx).style.display = 'none';
      const inp = document.getElementById('key-input-' + idx);
      inp.value = _keyCache[idx].raw_key || '';
      inp.style.display = '';
      inp.focus();
      document.getElementById('key-edit-actions-' + idx).style.display = '';
    }

    async function saveEditKey(idx) {
      if (!_keyCache || !_keyCache[idx]) return;
      const newVal = document.getElementById('key-input-' + idx).value.trim();
      if (!newVal) return cancelEditKey(idx);
      const oldVal = _keyCache[idx].raw_key;
      const res = await apiFetch('/admin/api/keys', {
        method: 'PUT',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({old_key: oldVal, new_key: newVal})
      });
      if (res.status === 401) return;
      if (res.ok) {
        const data = await res.json();
        showToast(data.message);
        loadConfig(false);
      } else {
        const err = await res.json();
        showToast(err.detail || 'Failed to save key.');
        cancelEditKey(idx);
      }
    }

    function cancelEditKey(idx) {
      document.getElementById('key-display-' + idx).style.display = '';
      document.getElementById('key-actions-' + idx).style.display = '';
      document.getElementById('key-input-' + idx).style.display = 'none';
      document.getElementById('key-edit-actions-' + idx).style.display = 'none';
    }

    async function deleteKey(idx) {
      if (!_keyCache || !_keyCache[idx]) return;
      const entry = _keyCache[idx];
      const apiKey = entry.raw_key;
      if (!confirm('Delete this key?\n\n' + (entry.key_masked || apiKey))) return;
      const res = await apiFetch('/admin/api/keys', {
        method: 'DELETE',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({api_key: apiKey})
      });
      if (res.status === 401) return;
      if (res.ok) {
        const data = await res.json();
        showToast(data.message);
        loadConfig(false);
      } else {
        const err = await res.json();
        showToast(err.detail || 'Failed to delete key.');
      }
    }

    async function runTestChat() {
      const prompt = document.getElementById('play-prompt').value;
      const model = document.getElementById('play-model').value;
      const mode = document.getElementById('play-mode').value;
      const out = document.getElementById('play-output');
      const token = getApiToken();

      out.innerText = mode === 'streaming'
        ? 'Streaming response from NVIDIA NIM…\n'
        : 'Sending test request to NVIDIA NIM…';
      btnLoading('.btn-test-chat', true);

      try {
        if (mode === 'streaming') {
          const url = '/admin/api/test_chat_stream';
          const headers = { 'Content-Type': 'application/json' };
          if (token) headers['Authorization'] = 'Bearer ' + token;
          const res = await fetch(url, {
            method: 'POST',
            headers,
            body: JSON.stringify({ prompt, model })
          });

          if (res.status === 401) {
            showToast('Authentication required — enter PROXY_API_TOKEN in Server Security');
            return;
          }
          if (!res.ok) {
            out.innerText = `[ERROR ${res.status}]: ${res.statusText}`;
            return;
          }

          out.innerText = 'Streaming response from NVIDIA NIM…\n';
          const reader = res.body.getReader();
          const decoder = new TextDecoder();
          let buffer = '';

          while (true) {
            const { done, value } = await reader.read();
            if (done) break;
            buffer += decoder.decode(value, { stream: true });
            out.innerText += buffer;
            buffer = '';
            // Auto-scroll
            out.scrollTop = out.scrollHeight;
          }
        } else {
          const res = await apiFetch('/admin/api/test_chat', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ prompt, model })
          });
          if (res.status === 401) return;
          const data = await res.json();
          if (!res.ok) {
            out.innerText = `[ERROR ${res.status}]: ${data.detail || data.message || res.statusText}`;
            return;
          }
          out.innerText = data.status === 'success'
            ? `[SUCCESS] (${data.model_used}):\n${data.response}`
            : `[ERROR]: ${data.message}`;
        }
      } catch (err) {
        out.innerText = `[NETWORK ERROR]: ${err instanceof Error ? err.message : String(err)}`;
      } finally {
        btnLoading('.btn-test-chat', false);
      }
    }

    const tokEl = document.getElementById('cfg-proxy-token');
    if (tokEl) {
      const saved = localStorage.getItem('nim_proxy_token');
      if (saved && !tokEl.value) tokEl.value = saved;
      tokEl.addEventListener('input', e => localStorage.setItem('nim_proxy_token', e.target.value.trim()));
    }

    async function loadStats() {
      try {
        const res = await apiFetch('/admin/api/stats');
        if (res.status === 401) return;
        const data = await res.json();

        const stats = data.key_pool_stats || {};
        document.getElementById('stat-total').innerText = stats.total || 0;
        document.getElementById('stat-active').innerText = stats.active || 0;
        document.getElementById('stat-cooldown').innerText = stats.cooldown || 0;
        document.getElementById('stat-disabled').innerText = stats.disabled || 0;

        const tbody = document.getElementById('key-table-body');
        tbody.innerHTML = '';
        (data.key_details || []).forEach(k => {
          let badge = '<span class="badge badge-success">Healthy / Active</span>';
          let cdText = '-';

          if (k.is_disabled) {
            badge = '<span class="badge badge-danger">Disabled (401)</span>';
          } else if (k.cooldown_remaining > 0) {
            badge = '<span class="badge badge-warning">Cooldown</span>';
            cdText = k.cooldown_remaining + 's';
          }

          tbody.innerHTML += `
            <tr>
              <td>${k.key}</td>
              <td>${badge}</td>
              <td>${k.failure_count}</td>
              <td>${cdText}</td>
            </tr>
          `;
        });

        renderKeyList(data.key_details || []);
        // NOTE: do NOT call submitPollingSnapshot from stats — it lacks raw_key needed for CRUD
      } catch (err) {
        // silently ignore polling failures to avoid toast spam
        console.error('Polling stats failed:', err);
      }
    }

    loadConfig(true);
    fetchModels();
    let _pollAbort = null;
    setInterval(() => {
      if (_pollAbort) _pollAbort.abort();
      _pollAbort = new AbortController();
      // TODO: pass signal to loadStats for clean cancellation
      loadStats();
    }, 5000);
  
