/* eslint-disable no-unused-vars */
const $ = (selector, scope = document) => scope.querySelector(selector);
const $$ = (selector, scope = document) => Array.from(scope.querySelectorAll(selector));

const numberFormatter = new Intl.NumberFormat('zh-CN');
const formatNumber = (value) => numberFormatter.format(Math.max(0, Number(value || 0)));

const reliabilityData = window.__RELIABILITY_DATA__ || {};

/* ---------- DOM references ---------- */
const importLogList = $('#import-log-list');
const dailyImportsEl = $('#daily-imports');
const latestSourceEl = $('#latest-source');

const manualForm = $('#manual-import-form');
const fileForm = $('#file-import-form');
const dbForm = $('#db-import-form');

const filePreview = $('#file-preview');
const dbSampleInline = $('#db-sample-inline');
const dbInlineStatus = $('#db-inline-status');
const dbStatus = $('#db-status');

const saveParamBtn = $('#save-param-btn');
const paramToast = $('#param-toast');
const userForm = $('#user-form');
const userTableBody = $('#user-table tbody');

const fileModal = $('#file-analysis-modal');
const fileModalStatus = $('#file-modal-status');
const fileModalPreview = $('#file-modal-preview');
const fileModalRunBtn = $('#file-modal-run');
const fileModalApplyBtn = $('#file-modal-apply');
const fileModalConfirmBtn = $('#file-modal-confirm');
const fileModalSelects = {
    module: $('#file-modal-map-module'),
    failures: $('#file-modal-map-failures'),
    mtbf: $('#file-modal-map-mtbf'),
    runtime: $('#file-modal-map-runtime')
};
const fileAiResult = $('#file-ai-result');
const fileAiResultStatus = $('#file-ai-result-status');
const fileAiResultList = $('#file-ai-result-list');

const dbModal = $('#db-analysis-modal');
const dbModalStatus = $('#db-modal-status');
const dbModalPreview = $('#db-modal-preview');
const dbModalRunBtn = $('#db-modal-run');
const dbModalApplyBtn = $('#db-modal-apply');
const dbModalConfirmBtn = $('#db-modal-confirm');
const dbModalInputs = {
    module: $('#db-modal-map-module'),
    failures: $('#db-modal-map-failures'),
    mtbf: $('#db-modal-map-mtbf'),
    runtime: $('#db-modal-map-runtime')
};
const dbAiResult = $('#db-ai-result');
const dbAiResultStatus = $('#db-ai-result-status');
const dbAiResultList = $('#db-ai-result-list');

const aiFieldLabels = {
    module: '模块',
    failures: '失效次数',
    mtbf: 'MTBF',
    runtime: '时长'
};

const contentSections = $$('.content-section');
const navButtons = $$('[data-section-target]');

/* ---------- Global state ---------- */
const chartRegistry = {};
const sectionRefs = {};
const state = {
    file: {
        file: null,
        tag: '',
        sample: { headers: [], rows: [] },
        aiMapping: null
    },
    db: {
        payload: null,
        sample: { headers: [], rows: [] },
        aiMapping: null
    }
};

/* ---------- Helpers ---------- */
const showToast = (message) => {
    if (!paramToast || !message) return;
    paramToast.textContent = message;
    paramToast.classList.add('show');
    setTimeout(() => paramToast.classList.remove('show'), 2200);
};

const setStatusText = (element, text) => {
    if (element) element.textContent = text;
};

const openModal = (modal) => {
    if (!modal) return;
    modal.removeAttribute('hidden');
    document.body.classList.add('modal-open');
};

const closeModal = (modal) => {
    if (!modal) return;
    modal.setAttribute('hidden', 'hidden');
    if (!document.querySelector('.modal-backdrop:not([hidden])')) {
        document.body.classList.remove('modal-open');
    }
};

document.addEventListener('click', (event) => {
    const trigger = event.target.closest('[data-close]');
    if (!trigger) return;
    const targetModal = document.querySelector(trigger.getAttribute('data-close'));
    closeModal(targetModal);
});

/* ---------- Navigation ---------- */
const getSectionId = (section) => section.dataset.section || section.id;

const showSection = (targetId) => {
    contentSections.forEach((section) => {
        const visible = getSectionId(section) === targetId;
        section.classList.toggle('section-visible', visible);
    });
    navButtons.forEach((button) => {
        button.classList.toggle('active', button.dataset.sectionTarget === targetId);
    });
};

navButtons.forEach((button) => {
    button.addEventListener('click', (event) => {
        if (button.tagName === 'A') event.preventDefault();
        const targetId = button.dataset.sectionTarget;
        if (!targetId) return;
        showSection(targetId);
        document.querySelector(`[data-section="${targetId}"]`)?.scrollIntoView({
            behavior: 'smooth',
            block: 'start'
        });
    });
});

showSection('data-import');

/* ---------- import stats & log ---------- */
const defaultLog = (reliabilityData.recent_records || []).map((record) => ({
    source: record.source || '导入',
    detail: `${record.module} · 失效 ${record.failures} 次`,
    time: new Date(record.timestamp || Date.now()).toLocaleString('zh-CN')
}));

let dailyImports = Number(reliabilityData.import_stats?.daily ?? 0);
let importLog = defaultLog.slice(-6).reverse();

const renderLog = () => {
    if (!importLog.length) {
        importLogList.innerHTML = '<li>暂无记录，完成一次导入即可驱动模型</li>';
        return;
    }
    importLogList.innerHTML = importLog
        .map((item) => `<li><strong>${item.source}</strong> · ${item.detail}<br/><span>${item.time}</span></li>`)
        .join('');
};

const pushLog = (entry) => {
    importLog.unshift({ ...entry, time: new Date().toLocaleString('zh-CN') });
    importLog = importLog.slice(0, 6);
    renderLog();
};

const renderUserRow = (profile) => {
    if (!userTableBody) return;
    const row = document.createElement('tr');
    const status = profile.status || '启用';
    row.innerHTML = `
        <td>${profile.name || ''}</td>
        <td>${profile.role || ''}</td>
        <td>${profile.email || ''}</td>
        <td><span class="status-pill ${status === '启用' ? 'success' : 'warning'}">${status}</span></td>
    `;
    const hintRow = userTableBody.querySelector('.empty-hint');
    if (hintRow) hintRow.remove();
    userTableBody.appendChild(row);
};

const refreshImportStats = () => {
    dailyImportsEl.textContent = formatNumber(dailyImports);
    latestSourceEl.textContent = importLog[0]?.source || '-';
};

renderLog();
refreshImportStats();

/* ---------- Table preview & mapping helpers ---------- */
const renderTablePreview = (element, rows, emptyText = '暂无数据') => {
    if (!element) return;
    if (!rows.length) {
        element.innerHTML = emptyText;
        return;
    }
    const headers = Object.keys(rows[0]);
    const head = `<tr>${headers.map((header) => `<th>${header}</th>`).join('')}</tr>`;
    const body = rows
        .map((row) => `<tr>${headers.map((header) => `<td>${row[header] ?? ''}</td>`).join('')}</tr>`)
        .join('');
    element.innerHTML = `<table class="preview-table"><thead>${head}</thead><tbody>${body}</tbody></table>`;
};

const populateSelect = (select, headers = []) => {
    if (!select) return;
    const options = ['<option value="">自动识别</option>']
        .concat(headers.map((header) => `<option value="${header}">${header}</option>`))
        .join('');
    select.innerHTML = options;
};

const populateSelectGroup = (group, headers = []) => {
    Object.values(group).forEach((select) => populateSelect(select, headers));
};

const collectMappingFromSelects = (selects) => {
    const mapping = {};
    Object.entries(selects).forEach(([key, select]) => {
        if (select?.value) mapping[key] = select.value;
    });
    return mapping;
};

const collectMappingFromInputs = (inputs) => {
    const mapping = {};
    Object.entries(inputs).forEach(([key, input]) => {
        if (input?.value) mapping[key] = input.value.trim();
    });
    return mapping;
};

const applyMappingToSelects = (mapping, selects) => {
    Object.entries(mapping || {}).forEach(([key, value]) => {
        if (selects[key]) selects[key].value = value;
    });
};

const applyMappingToInputs = (mapping, inputs) => {
    Object.entries(mapping || {}).forEach(([key, value]) => {
        if (inputs[key]) inputs[key].value = value;
    });
};

const resetAiResult = (wrapper, statusEl, listEl) => {
    if (statusEl) statusEl.textContent = '待运行';
    if (listEl) listEl.innerHTML = '';
    if (wrapper) wrapper.hidden = true;
};

const renderAiResult = (wrapper, statusEl, listEl, mapping) => {
    if (!wrapper || !statusEl || !listEl) return;
    const entries = Object.entries(mapping || {});
    if (!entries.length) {
        statusEl.textContent = '未得到有效映射，请重试或手动选择';
        listEl.innerHTML = '<li>暂无推荐映射</li>';
        wrapper.hidden = false;
        return;
    }
    statusEl.textContent = '已生成推荐，请确认是否应用';
    listEl.innerHTML = entries
        .map(([key, value]) => `<li><span>${aiFieldLabels[key] || key}</span><code>${value}</code></li>`)
        .join('');
    wrapper.hidden = false;
};

/* ---------- HTTP helpers ---------- */
const postJSON = async (url, payload) => {
    const response = await fetch(url, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload)
    });
    const body = await response.json();
    if (!response.ok) throw new Error(body.message || '请求失败');
    return body;
};

const fetchFileSample = async (file, tag) => {
    const formData = new FormData();
    formData.append('file', file);
    if (tag) formData.append('tag', tag);
    const response = await fetch('/api/import/file/sample', { method: 'POST', body: formData });
    const body = await response.json();
    if (!response.ok) throw new Error(body.message || '样本获取失败');
    return { headers: body.headers || [], rows: body.rows || [] };
};

const runDeepseekMapping = async (sample, statusEl) => {
    if (!sample.headers.length) throw new Error('没有可解析的表头');
    if (!sample.rows.length) throw new Error('没有样本数据');
    setStatusText(statusEl, 'DeepSeek 正在分析...');
    const body = await postJSON('/api/deepseek/mapping', sample);
    setStatusText(statusEl, 'DeepSeek 分析完成，可应用推荐映射');
    return body.mapping || {};
};

const refreshAfterSuccess = (message, logEntry) => {
    showToast(message);
    dailyImports += 1;
    refreshImportStats();
    if (logEntry) pushLog(logEntry);
};

/* ---------- File import ---------- */
const openFileModal = (sample) => {
    state.file.aiMapping = null;
    populateSelectGroup(fileModalSelects, sample.headers);
    renderTablePreview(fileModalPreview, sample.rows, '暂无样本');
    renderTablePreview(filePreview, sample.rows, '暂无数据');
    fileModalApplyBtn.disabled = true;
    setStatusText(fileModalStatus, '样本已获取，请选择映射策略');
    resetAiResult(fileAiResult, fileAiResultStatus, fileAiResultList);
    openModal(fileModal);
};

fileForm?.addEventListener('submit', async (event) => {
    event.preventDefault();
    const fileInput = fileForm.querySelector('input[type="file"]');
    const tagInput = fileForm.querySelector('input[name="tag"]');
    const file = fileInput?.files?.[0];
    if (!file) {
        showToast('请先选择文件');
        return;
    }
    try {
        setStatusText(fileModalStatus, '正在解析样本 ...');
        state.file.file = file;
        state.file.tag = tagInput?.value?.trim() || '';
        const sample = await fetchFileSample(file, state.file.tag);
        state.file.sample = sample;
        openFileModal(sample);
    } catch (error) {
        showToast(error.message);
        setStatusText(fileModalStatus, error.message);
    }
});

fileModalRunBtn?.addEventListener('click', async () => {
    try {
        const mapping = await runDeepseekMapping(state.file.sample, fileModalStatus);
        state.file.aiMapping = mapping;
        fileModalApplyBtn.disabled = false;
        renderAiResult(fileAiResult, fileAiResultStatus, fileAiResultList, mapping);
    } catch (error) {
        showToast(error.message);
        setStatusText(fileModalStatus, error.message);
    }
});

fileModalApplyBtn?.addEventListener('click', () => {
    if (!state.file.aiMapping) {
        showToast('暂无可用的推荐映射');
        return;
    }
    applyMappingToSelects(state.file.aiMapping, fileModalSelects);
    showToast('已应用 DeepSeek 建议');
});

fileModalConfirmBtn?.addEventListener('click', async () => {
    if (!state.file.file) {
        showToast('请重新上传文件');
        return;
    }
    const mode = document.querySelector('input[name="file-modal-mode"]:checked')?.value || 'preprocess';
    const mapping = collectMappingFromSelects(fileModalSelects);
    const formData = new FormData();
    formData.append('file', state.file.file);
    if (state.file.tag) formData.append('tag', state.file.tag);
    formData.append('analysis-mode', mode);
    Object.entries(mapping).forEach(([key, value]) => {
        if (value) formData.append(`map-${key}`, value);
    });
    try {
        const response = await fetch('/api/import/file', { method: 'POST', body: formData });
        const body = await response.json();
        if (!response.ok) throw new Error(body.message || '文件导入失败');
        closeModal(fileModal);
        refreshAfterSuccess(body.message || '文件导入完成', {
            source: '文件导入',
            detail: state.file.file.name
        });
        showSection('data-import');
    } catch (error) {
        showToast(error.message);
    }
});

/* ---------- Database import ---------- */
const dbSamplePayload = (formData) => ({
    connection: formData.get('connection'),
    table: formData.get('table'),
    increment: formData.get('increment') || undefined,
    limit: Number(formData.get('limit') || 200)
});

dbForm?.addEventListener('submit', async (event) => {
    event.preventDefault();
    const formData = new FormData(dbForm);
    const payload = dbSamplePayload(formData);
    if (!payload.connection || !payload.table) {
        showToast('请填写连接串和数据表');
        return;
    }
    try {
        setStatusText(dbInlineStatus, '正在连接数据库...');
        const body = await postJSON('/api/import/mysql/sample', payload);
        state.db.payload = payload;
        state.db.sample = { headers: body.headers || [], rows: body.rows || [] };
        state.db.aiMapping = null;
        renderTablePreview(dbSampleInline, state.db.sample.rows, '暂无数据');
        renderTablePreview(dbModalPreview, state.db.sample.rows, '暂无数据');
        Object.values(dbModalInputs).forEach((input) => {
            if (input) input.value = '';
        });
        dbModalApplyBtn.disabled = true;
        setStatusText(dbModalStatus, '样本已获取，请选择映射策略');
        resetAiResult(dbAiResult, dbAiResultStatus, dbAiResultList);
        openModal(dbModal);
    } catch (error) {
        setStatusText(dbInlineStatus, error.message);
        showToast(error.message);
    }
});

dbModalRunBtn?.addEventListener('click', async () => {
    try {
        const mapping = await runDeepseekMapping(state.db.sample, dbModalStatus);
        state.db.aiMapping = mapping;
        dbModalApplyBtn.disabled = false;
        renderAiResult(dbAiResult, dbAiResultStatus, dbAiResultList, mapping);
    } catch (error) {
        setStatusText(dbModalStatus, error.message);
        showToast(error.message);
    }
});

dbModalApplyBtn?.addEventListener('click', () => {
    if (!state.db.aiMapping) {
        showToast('暂无可用的推荐映射');
        return;
    }
    applyMappingToInputs(state.db.aiMapping, dbModalInputs);
    showToast('已应用 DeepSeek 建议');
});

dbModalConfirmBtn?.addEventListener('click', async () => {
    if (!state.db.payload) {
        showToast('请先连接数据库并获取样本');
        return;
    }
    const mode = document.querySelector('input[name="db-modal-mode"]:checked')?.value || 'preprocess';
    const mapping = collectMappingFromInputs(dbModalInputs);
    const payload = { ...state.db.payload, analysis_mode: mode, mapping };
    try {
        const body = await postJSON('/api/import/mysql', payload);
        dbStatus.textContent = '同步完成';
        dbStatus.classList.add('success');
        closeModal(dbModal);
        refreshAfterSuccess(body.message || '数据库导入完成', {
            source: 'MySQL 导入',
            detail: payload.table
        });
        showSection('data-import');
    } catch (error) {
        showToast(error.message);
    }
});

/* ---------- Manual import ---------- */
manualForm?.addEventListener('submit', async (event) => {
    event.preventDefault();
    const formData = new FormData(manualForm);
    const payload = {
        module: formData.get('module'),
        failures: Number(formData.get('failures') || 0),
        mtbf: Number(formData.get('mtbf') || 1),
        runtime: Number(formData.get('runtime') || formData.get('mtbf') || 1)
    };
    if (!payload.module) {
        showToast('模块名称不可为空');
        return;
    }
    try {
        const body = await postJSON('/api/import/manual', payload);
        manualForm.reset();
        refreshAfterSuccess(body.message || '手动导入完成', {
            source: '手动导入',
            detail: `${payload.module} · 失效 ${payload.failures}`
        });
    } catch (error) {
        showToast(error.message);
    }
});

/* ---------- Analyze sections ---------- */
document.querySelectorAll('[data-section-container]').forEach((container) => {
    const section = container.dataset.sectionContainer;
    sectionRefs[section] = {
        container,
        button: document.querySelector(`[data-section-button="${section}"]`),
        status: document.querySelector(`[data-section-status="${section}"]`),
        formulas: document.querySelector(`[data-formula-container="${section}"]`),
        loading: false
    };
});

const commonChartOptions = {
    responsive: true,
    maintainAspectRatio: false,
    plugins: {
        legend: {
            labels: {
                color: '#cbd5f5',
                usePointStyle: true
            }
        }
    },
    scales: {
        x: {
            ticks: { color: '#94a3b8' },
            grid: { color: 'rgba(148, 163, 184, 0.1)' }
        },
        y: {
            ticks: { color: '#94a3b8' },
            grid: { color: 'rgba(148, 163, 184, 0.08)' }
        }
    }
};

const renderCharts = (section, charts) => {
    const ref = sectionRefs[section];
    if (!ref) return;
    if (!chartRegistry[section]) chartRegistry[section] = [];
    chartRegistry[section].forEach((chart) => chart.destroy());
    chartRegistry[section] = [];
    ref.container.innerHTML = '';
    if (!charts.length) {
        ref.container.innerHTML = '<p class="doc-note">暂无可视化，请先导入数据</p>';
        return;
    }
    charts.forEach((definition) => {
        const card = document.createElement('div');
        card.className = 'chart-card';
        const canvas = document.createElement('canvas');
        card.appendChild(canvas);
        ref.container.appendChild(card);
        const instance = new Chart(canvas.getContext('2d'), {
            type: definition.type,
            data: {
                labels: definition.labels,
                datasets: definition.datasets
            },
            options: { ...commonChartOptions, ...(definition.options || {}) }
        });
        chartRegistry[section].push(instance);
    });
};

const renderFormulas = (element, formulas) => {
    if (!element) return;
    element.innerHTML = '';
    if (!formulas?.length) return;
    element.innerHTML = formulas
        .map(
            (formula) => `
        <article class="formula-card">
            <h5>${formula.title}</h5>
            <p class="formula-latex">\\(${formula.latex}\\)</p>
            <p class="formula-desc">${formula.description}</p>
        </article>`
        )
        .join('');
    if (window.MathJax) {
        window.MathJax.typesetPromise?.([element]);
    }
};

const analyzeSection = async (section) => {
    const ref = sectionRefs[section];
    if (!ref || ref.loading) return;
    ref.loading = true;
    ref.button && (ref.button.disabled = true);
    setStatusText(ref.status, '分析中...');
    try {
        const response = await fetch(`/api/analyze/${section}`);
        const body = await response.json();
        if (!response.ok) throw new Error(body.message || '分析失败');
        renderCharts(section, body.charts || []);
        renderFormulas(ref.formulas, body.formulas || []);
        setStatusText(ref.status, '分析完成');
        if (ref.button) ref.button.textContent = '重新分析';
    } catch (error) {
        setStatusText(ref.status, error.message);
        showToast(error.message);
    } finally {
        ref.loading = false;
        ref.button && (ref.button.disabled = false);
    }
};

Object.entries(sectionRefs).forEach(([section, ref]) => {
    ref.button?.addEventListener('click', () => analyzeSection(section));
});

/* ---------- Parameter persistence ---------- */
saveParamBtn?.addEventListener('click', () => {
    const rows = $$('#param-table-body tr');
    const payload = rows.map((row) => {
        const [alpha, beta, extra] = row.querySelectorAll('input');
        return {
            model: row.dataset.model,
            alpha: alpha?.value,
            beta: beta?.value,
            extra: extra?.value
        };
    });
    localStorage.setItem('reliability-params', JSON.stringify(payload));
    showToast('参数已保存');
});

const savedParams = localStorage.getItem('reliability-params');
if (savedParams) {
    try {
        JSON.parse(savedParams).forEach((item) => {
            const row = document.querySelector(`[data-model="${item.model}"]`);
            if (!row) return;
            const [alpha, beta, extra] = row.querySelectorAll('input');
            if (alpha) alpha.value = item.alpha;
            if (beta) beta.value = item.beta;
            if (extra) extra.value = item.extra;
        });
    } catch (error) {
        console.warn('参数还原失败', error);
    }
}

/* ---------- User table ---------- */
userForm?.addEventListener('submit', async (event) => {
    event.preventDefault();
    const formData = new FormData(userForm);
    const payload = {
        name: formData.get('name'),
        role: formData.get('role'),
        email: formData.get('email'),
        status: formData.get('status') || '启用'
    };
    const submitBtn = userForm.querySelector('button[type=\"submit\"]');
    if (submitBtn) submitBtn.disabled = true;
    try {
        const resp = await fetch('/api/users', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(payload)
        });
        const result = await resp.json();
        if (!resp.ok || result.status !== 'success') {
            showToast(result.message || '同步云端失败');
            return;
        }
        renderUserRow(result.profile || payload);
        userForm.reset();
        showToast('已同步到云端');
    } catch (error) {
        console.error(error);
        showToast('同步云端失败');
    } finally {
        if (submitBtn) submitBtn.disabled = false;
    }
});

const loadUsersFromCloud = async () => {
    try {
        const resp = await fetch('/api/users');
        const result = await resp.json();
        if (!resp.ok || result.status !== 'success') {
            console.warn('加载云端用户失败', result.message);
            return;
        }
        const users = result.data || [];
        users.forEach(renderUserRow);
    } catch (error) {
        console.warn('加载云端用户失败', error);
    }
};

loadUsersFromCloud();

// 全部板块刷新（手动触发）
const refreshAllSections = () => {
    Object.keys(sectionRefs).forEach((section) => analyzeSection(section));
};

// 导出 HTML
const exportBtn = $('#export-html-btn');
exportBtn?.addEventListener('click', async () => {
    try {
        const resp = await fetch('/api/export/html');
        if (!resp.ok) throw new Error('导出失败');
        const blob = await resp.blob();
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = 'reliability_export.html';
        a.click();
        URL.revokeObjectURL(url);
        showToast('导出完成');
    } catch (error) {
        showToast(error.message);
    }
});

