import { useCallback, useEffect, useMemo, useRef, useState } from 'react'

const API = '/api'

const TOOL_META = {
  search_dataset:         { icon: '🔍', label: '跨平台搜索' },
  search_hf_hub:          { icon: '🤗', label: 'HF搜索' },
  search_semantic_scholar: { icon: '📚', label: 'S2论文搜索' },
  search_pwc_dataset:     { icon: '📈', label: 'GitHub数据集' },
  get_hf_metadata:        { icon: '📊', label: 'HF元数据' },
  get_hf_dataset_card:    { icon: '📋', label: 'HF卡片' },
  get_hf_dataset_files:   { icon: '🗂️', label: 'HF文件' },
  get_hf_dataset_configs: { icon: '⚙️', label: 'HF配置' },
  web_search:             { icon: '🌍', label: '网页搜索' },
  tavily_search:          { icon: '✨', label: 'Tavily' },
  get_github_readme:      { icon: '📄', label: 'README' },
  get_github_dir:         { icon: '📁', label: 'GitHub目录' },
  get_github_repo_info:   { icon: '⭐', label: 'GitHub信息' },
  get_zenodo_record:      { icon: '🎓', label: 'Zenodo' },
  get_gdrive_folder:      { icon: '📂', label: 'GDrive' },
  fetch_webpage_text:     { icon: '🌐', label: '读取网页' },
  search_zenodo:          { icon: '🔬', label: 'Zenodo搜索' },
  search_opendatalab:     { icon: '🏛️', label: 'OpenDataLab' },
  compare_datasets:       { icon: '⚖️', label: '批量对比' },
  get_paper_code_repos:   { icon: '🔗', label: '论文代码仓库' },
  finish:                 { icon: '✅', label: '完成' },
}

const ALL_KEY_FIELDS = [
  { id: 'dashscope', header: 'X-API-Key', label: 'DashScope (百炼)', storage: 'dashscope_api_key',
    placeholder: 'sk-…', hint: 'LLM 推理引擎（必需）', required: true },
  { id: 'github', header: 'X-GitHub-Token', label: 'GitHub Token', storage: 'github_token',
    placeholder: 'ghp_…', hint: '解除 60 次/小时限额' },
  { id: 'huggingface', header: 'X-HF-Token', label: 'HuggingFace Token', storage: 'hf_token',
    placeholder: 'hf_…', hint: '访问 gated 数据集 / 提升限额' },
  { id: 'semantic_scholar', header: 'X-S2-Key', label: 'Semantic Scholar', storage: 's2_api_key',
    placeholder: '…', hint: '解除 S2 论文搜索限流' },
  { id: 'tavily', header: 'X-Tavily-Key', label: 'Tavily Search', storage: 'tavily_api_key',
    placeholder: 'tvly-…', hint: '高质量搜索（推荐）' },
]

function Spinner({ size = 10 }) {
  return <span className="spin" style={{ width: size, height: size }} />
}

// ── Settings Modal ──────────────────────────────────────────────────────────

function SettingsModal({ open, onClose, keys, onChange, serverKeys, onTest, testStatus }) {
  if (!open) return null
  return (
    <div className="modal-overlay" onClick={onClose}>
      <div className="modal" onClick={e => e.stopPropagation()}>
        <div className="modal-head">
          <h2>API 配置</h2>
          <button className="modal-close" onClick={onClose}>×</button>
        </div>
        <div className="modal-body">
          <p className="modal-desc">配置后自动保存在浏览器本地，不会上传到服务端存储。</p>
          {ALL_KEY_FIELDS.map(f => {
            const val = (keys[f.id] || '').trim()
            const fromServer = !val && serverKeys[f.id]
            return (
              <div key={f.id} className="cfg-row">
                <div className="cfg-label">
                  <span>{f.label}</span>
                  {f.required && <span className="cfg-req">必需</span>}
                  {val && <span className="cfg-ok">✓</span>}
                  {fromServer && <span className="cfg-env">服务器已配</span>}
                </div>
                <div className="cfg-input-row">
                  <input className="cfg-input" type="password" autoComplete="off"
                    placeholder={f.placeholder} value={keys[f.id] || ''}
                    onChange={e => onChange(f.id, e.target.value)} />
                  {f.id === 'dashscope' && val && (
                    <button className="cfg-test" onClick={onTest}
                      disabled={testStatus === 'testing'}>
                      {testStatus === 'testing' ? '…' : testStatus === 'ok' ? '✓' : '测试'}
                    </button>
                  )}
                </div>
                <div className="cfg-hint">{f.hint}</div>
              </div>
            )
          })}
        </div>
      </div>
    </div>
  )
}

// ── Progress Strip ──────────────────────────────────────────────────────────

function ProgressStrip({ run, running }) {
  const [showDetail, setShowDetail] = useState(false)
  const scrollRef = useRef(null)
  const activity = run?.activity || []
  const tools = activity.filter(a => a.type === 'tool')
  const current = [...tools].reverse().find(t => !t.result)
  const done = run?.done
  const meta = current ? (TOOL_META[current.tool] || { icon: '⚙️', label: current.tool }) : null

  useEffect(() => {
    if (!showDetail) return
    const el = scrollRef.current
    if (el) el.scrollTop = el.scrollHeight
  }, [activity.length, showDetail])

  if (!running && !done) return null

  return (
    <div className="progress-strip">
      <div className="progress-bar">
        <div className="progress-line">
          {running && !done && <Spinner size={12} />}
          {done && <span className="progress-done-icon">✓</span>}
          <span className="progress-text">
            {done ? `检索完成 · ${tools.length} 步 · ${(run.results || []).length} 个数据集`
              : meta ? `${meta.icon} ${meta.label} · 第 ${tools.length} 步`
              : '搜索中'}
          </span>
          {/* mini tool icon trail */}
          <span className="progress-trail">
            {tools.slice(-6).map((t, i) => {
              const m = TOOL_META[t.tool] || { icon: '·' }
              return <span key={i} className={`trail-dot ${t.result ? 'trail-done' : ''}`}
                title={m.label}>{m.icon}</span>
            })}
          </span>
        </div>
        <button className="progress-toggle" onClick={() => setShowDetail(v => !v)}>
          {showDetail ? '收起详情' : '查看详情'}
        </button>
      </div>
      {showDetail && (
        <div className="progress-detail" ref={scrollRef}>
          {activity.map((item, i) => {
            if (item.type === 'thought') {
              const preview = (item.text || '').replace(/\s+/g, ' ').trim().slice(0, 120)
              return <div key={i} className="pd-row pd-thought">◈ {preview}{item.text?.length > 120 ? '…' : ''}</div>
            }
            const m = TOOL_META[item.tool] || { icon: '·', label: item.tool }
            const sub = summarizeTool(item.tool, item.result)
            return (
              <div key={i} className="pd-row pd-tool">
                <span>{m.icon}</span>
                <span className="pd-name">{m.label}</span>
                {!item.result && <Spinner size={8} />}
                <span className="pd-sub">{sub}</span>
              </div>
            )
          })}
        </div>
      )}
    </div>
  )
}

function summarizeTool(tool, result) {
  if (!result) return ''
  if (tool === 'search_dataset') return result.found ? `${result.live_count} 链接` : '无结果'
  if (tool === 'search_hf_hub') return result.datasets?.length ? `${result.datasets.length} 个 · ${result.datasets[0]?.id || ''}` : '无'
  if (tool === 'search_semantic_scholar') return result.count ? `${result.count} 篇` : (result.error || '无')
  if (tool === 'get_hf_metadata') return result.splits?.length ? result.splits.join(', ') : 'splits 为空'
  if (tool === 'web_search' || tool === 'tavily_search') return result.count ? `${result.count} 条` : (result.error || '无')
  if (tool === 'get_github_readme') {
    const c = result.data_links?.cloud_links?.length
    return c ? `${c} 个下载链接` : (result.has_content ? '已读取' : '无')
  }
  if (tool === 'finish') return `${result.datasets?.length ?? 0} 个数据集`
  if (result.error) return result.error.slice(0, 40)
  return '✓'
}

// ── Dataset Card ────────────────────────────────────────────────────────────

function DatasetCard({ ds }) {
  const [expanded, setExpanded] = useState(false)
  const links = ds.links || []
  const primary = links[0]
  const rest = expanded ? links.slice(1) : []
  const splits = ds.splits || []
  const conf = ds.confidence != null ? Math.round(ds.confidence * 100) : null

  return (
    <article className="ds-card">
      <div className="ds-top">
        <h3 className="ds-name">{ds.name}</h3>
        <div className="ds-badges">
          {ds.hf_id && <a href={`https://huggingface.co/datasets/${ds.hf_id}`} target="_blank"
            rel="noopener noreferrer" className="ds-badge ds-hf">🤗 {ds.hf_id}</a>}
          {ds.license && <span className="ds-badge">{ds.license}</span>}
          {conf != null && <span className={`ds-badge ${conf >= 80 ? 'ds-conf-hi' : conf >= 50 ? 'ds-conf-mid' : 'ds-conf-lo'}`}>{conf}%</span>}
          {ds.downloads != null && <span className="ds-badge">↓{ds.downloads.toLocaleString()}</span>}
        </div>
      </div>
      {splits.length > 0 && (
        <div className="ds-splits">
          {splits.map(s => <span key={s} className={`ds-split ${s === 'train' ? 'ds-split-train' : ''}`}>{s}</span>)}
        </div>
      )}
      {ds.reason && <p className="ds-reason">{ds.reason}</p>}
      {primary && (
        <a className="ds-link-primary" href={primary.url} target="_blank" rel="noopener noreferrer">
          <div><div className="ds-lp-title">{primary.label || primary.url}</div>
          <div className="ds-lp-sub">{primary.source}</div></div>
          <span>→</span>
        </a>
      )}
      {links.length > 1 && <>
        {rest.map((l, i) => (
          <a key={i} className="ds-link-minor" href={l.url} target="_blank" rel="noopener noreferrer">
            <span className="ds-lm-src">{l.source}</span>
            <span className="ds-lm-label">{l.label || l.url}</span>
            <span>↗</span>
          </a>
        ))}
        <button className="ds-more" onClick={() => setExpanded(v => !v)}>
          {expanded ? '收起' : `+${links.length - 1} 个来源`}
        </button>
      </>}
    </article>
  )
}

// ── Paper Result Block ──────────────────────────────────────────────────────

function PaperBlock({ index, run, multi }) {
  const [open, setOpen] = useState(true)
  const results = run.results || []
  if (!multi) {
    return <>
      {run.answer && <div className="answer-bar">{run.answer}</div>}
      {run.error && <div className="notice">{run.error}</div>}
      {results.length === 0 && !run.error
        ? <div className="empty-hint">未找到匹配数据集</div>
        : results.map((ds, i) => <DatasetCard key={`${ds.name}-${i}`} ds={ds} />)}
    </>
  }
  return (
    <div className="paper-block">
      <button className="pb-head" onClick={() => setOpen(v => !v)}>
        <span className="pb-idx">{index + 1}</span>
        <span className="pb-name">{run.name}</span>
        <span className="pb-count">{results.length > 0 ? `${results.length} 个数据集` : (run.error ? '出错' : '无结果')}</span>
        <span className="pb-arr">{open ? '▾' : '▸'}</span>
      </button>
      {open && <div className="pb-body">
        {run.answer && <div className="answer-bar">{run.answer}</div>}
        {run.error && <div className="notice">{run.error}</div>}
        {results.map((ds, i) => <DatasetCard key={`${index}-${ds.name}-${i}`} ds={ds} />)}
      </div>}
    </div>
  )
}

// ── App ─────────────────────────────────────────────────────────────────────

export default function App() {
  const [text, setText] = useState('')
  const [pdfFiles, setPdfFiles] = useState([])
  const [running, setRunning] = useState(false)
  const [paperRuns, setPaperRuns] = useState({})
  const [error, setError] = useState('')
  const [showSettings, setShowSettings] = useState(false)
  const [keys, setKeys] = useState(() =>
    Object.fromEntries(ALL_KEY_FIELDS.map(f => [f.id, '']))
  )
  const [serverKeys, setServerKeys] = useState({})
  const [testStatus, setTestStatus] = useState(null)

  useEffect(() => {
    const loaded = {}
    for (const f of ALL_KEY_FIELDS) {
      const v = localStorage.getItem(f.storage)
      if (v) loaded[f.id] = v
    }
    if (Object.keys(loaded).length) setKeys(prev => ({ ...prev, ...loaded }))
    fetch(`${API}/config`).then(r => r.ok ? r.json() : null)
      .then(d => { if (d?.server_keys) setServerKeys(d.server_keys) }).catch(() => {})
  }, [])

  const handleKeyChange = (id, v) => {
    setKeys(prev => ({ ...prev, [id]: v }))
    const f = ALL_KEY_FIELDS.find(x => x.id === id)
    if (f) localStorage.setItem(f.storage, v || '')
    if (id === 'dashscope') setTestStatus(null)
  }

  const testKey = useCallback(async () => {
    const k = (keys.dashscope || '').trim()
    if (!k) return
    setTestStatus('testing')
    try {
      const r = await fetch(`${API}/test-key`, { method: 'POST', headers: { 'X-API-Key': k } })
      const d = await r.json()
      setTestStatus(d.ok ? 'ok' : 'fail')
    } catch { setTestStatus('fail') }
  }, [keys.dashscope])

  const apiKey = (keys.dashscope || '').trim()
  const hasKey = !!apiKey || !!serverKeys.dashscope
  const configuredCount = ALL_KEY_FIELDS.filter(f => (keys[f.id] || '').trim() || serverKeys[f.id]).length

  const authHeaders = useMemo(() => {
    const h = {}
    for (const f of ALL_KEY_FIELDS) {
      const v = (keys[f.id] || '').trim()
      if (v) h[f.header] = v
    }
    return h
  }, [keys])

  const canRun = useMemo(
    () => (text.trim().length > 0 || pdfFiles.length > 0) && !running && hasKey,
    [text, pdfFiles, running, hasKey]
  )

  const handleStream = useCallback(async (res) => {
    const reader = res.body.getReader()
    const dec = new TextDecoder()
    let buf = ''
    while (true) {
      const { done, value } = await reader.read()
      if (done) break
      buf += dec.decode(value, { stream: true })
      const lines = buf.split('\n')
      buf = lines.pop() || ''
      for (const line of lines) {
        if (!line.startsWith('data: ')) continue
        const raw = line.slice(6).trim()
        if (!raw) continue
        let ev
        try { ev = JSON.parse(raw) } catch { continue }
        if (ev.event === 'stream_end' || ev.event === 'heartbeat' || ev.event === 'multi_done') continue
        const pi = ev.paper_index ?? 0
        const pn = ev.paper_name || pdfFiles[pi]?.name || `查询`
        const ensure = (d = {}) => ({ name: pn, activity: [], results: null, answer: '', error: '', done: false, ...d })

        if (ev.event === 'agent_thought') {
          setPaperRuns(p => { const c = ensure(p[pi]); return { ...p, [pi]: { ...c, name: pn, activity: [...c.activity, { type: 'thought', text: ev.text }] } } })
        }
        if (ev.event === 'tool_call') {
          setPaperRuns(p => { const c = ensure(p[pi]); return { ...p, [pi]: { ...c, name: pn, activity: [...c.activity, { type: 'tool', tool: ev.tool, args: ev.args, call_num: ev.call_num, result: null }] } } })
        }
        if (ev.event === 'tool_result') {
          setPaperRuns(p => { const c = ensure(p[pi]); return { ...p, [pi]: { ...c, name: pn, activity: c.activity.map(it => it.type === 'tool' && it.call_num === ev.call_num ? { ...it, result: ev.result } : it) } } })
        }
        if (ev.event === 'done') {
          setPaperRuns(p => { const c = ensure(p[pi]); return { ...p, [pi]: { ...c, name: pn, results: ev.results || [], answer: ev.reason || '', done: true } } })
        }
        if (ev.event === 'error') {
          const msg = ev.message || '发生错误'
          setError(msg)
          setPaperRuns(p => { const c = ensure(p[pi]); return { ...p, [pi]: { ...c, name: pn, error: msg } } })
        }
      }
    }
  }, [pdfFiles])

  const runSearch = useCallback(async () => {
    if (!canRun) return
    setRunning(true); setError('')
    const init = {}
    if (pdfFiles.length >= 1) pdfFiles.forEach((f, i) => { init[i] = { name: f.name, activity: [], results: null, answer: '', error: '', done: false } })
    else init[0] = { name: '查询', activity: [], results: null, answer: '', error: '', done: false }
    setPaperRuns(init)
    try {
      let res
      if (pdfFiles.length > 1) {
        const fd = new FormData()
        for (const f of pdfFiles) fd.append('files', f)
        fd.append('question', text)
        res = await fetch(`${API}/search/pdfs`, { method: 'POST', headers: authHeaders, body: fd })
      } else if (pdfFiles.length === 1) {
        const fd = new FormData()
        fd.append('file', pdfFiles[0]); fd.append('question', text)
        res = await fetch(`${API}/search/pdf`, { method: 'POST', headers: authHeaders, body: fd })
      } else {
        res = await fetch(`${API}/search`, { method: 'POST', headers: { 'Content-Type': 'application/json', ...authHeaders }, body: JSON.stringify({ text, options: {} }) })
      }
      if (!res.ok) throw new Error(await res.text().catch(() => `HTTP ${res.status}`))
      await handleStream(res)
    } catch (e) { setError(e.message) }
    finally { setRunning(false) }
  }, [canRun, handleStream, pdfFiles, text, authHeaders])

  const entries = Object.entries(paperRuns).sort((a, b) => Number(a[0]) - Number(b[0]))
  const hasResults = entries.some(([, r]) => r.results !== null)
  const multi = entries.length > 1

  return (
    <div className="app">
      <style>{`
*,*::before,*::after{box-sizing:border-box;margin:0;padding:0}
:root{--bg:#f8f9fb;--surface:#fff;--ink:#0f1923;--muted:#64748b;--line:#e5e7eb;--brand:#2563eb;--brand-soft:#eff6ff;--ok:#059669;--ok-soft:#ecfdf5;--bad:#dc2626;--radius:12px}
body{font-family:"Inter","Noto Sans SC","PingFang SC",system-ui,sans-serif;background:var(--bg);color:var(--ink);min-height:100vh}
.app{min-height:100vh;display:flex;flex-direction:column}

/* Top bar */
.topbar{display:flex;align-items:center;padding:12px 24px;background:var(--surface);border-bottom:1px solid var(--line)}
.topbar-brand{font-size:16px;font-weight:700;letter-spacing:-.02em;background:linear-gradient(135deg,#2563eb,#0ea5e9);-webkit-background-clip:text;-webkit-text-fill-color:transparent;background-clip:text;display:flex;align-items:center;gap:8px}
.topbar-brand::before{content:'◆';font-size:14px}
.topbar-sub{font-size:12px;color:var(--muted);margin-left:6px;font-weight:400}
.topbar-spacer{flex:1}
.topbar-keys{font-size:11px;color:var(--muted);margin-right:8px}
.settings-btn{width:32px;height:32px;border-radius:8px;border:1px solid var(--line);background:var(--surface);cursor:pointer;font-size:16px;display:flex;align-items:center;justify-content:center;color:var(--muted);transition:all 120ms}
.settings-btn:hover{border-color:#93c5fd;color:var(--ink);background:#f0f9ff}

/* Modal */
.modal-overlay{position:fixed;inset:0;background:rgba(0,0,0,.35);z-index:100;display:flex;align-items:center;justify-content:center;backdrop-filter:blur(2px)}
.modal{background:var(--surface);border-radius:16px;width:min(480px,92vw);max-height:85vh;overflow-y:auto;box-shadow:0 24px 48px rgba(0,0,0,.18)}
.modal-head{display:flex;justify-content:space-between;align-items:center;padding:18px 24px 12px;border-bottom:1px solid var(--line)}
.modal-head h2{font-size:17px;font-weight:700}
.modal-close{background:none;border:none;font-size:22px;cursor:pointer;color:var(--muted);padding:4px 8px;border-radius:6px}
.modal-close:hover{background:#f1f5f9;color:var(--ink)}
.modal-body{padding:16px 24px 24px;display:grid;gap:14px}
.modal-desc{font-size:12px;color:var(--muted);line-height:1.5}
.cfg-row{display:grid;gap:4px}
.cfg-label{display:flex;align-items:center;gap:6px;font-size:12px;font-weight:600;color:#334155}
.cfg-req{font-size:10px;color:var(--brand);background:var(--brand-soft);padding:1px 6px;border-radius:99px}
.cfg-ok{color:var(--ok);font-size:11px}
.cfg-env{font-size:10px;color:#6366f1;background:#eef2ff;padding:1px 6px;border-radius:99px}
.cfg-input-row{display:flex;gap:6px}
.cfg-input{flex:1;border:1px solid var(--line);border-radius:8px;padding:7px 10px;font:inherit;font-size:13px;background:#fafbfc;color:var(--ink);outline:none;min-width:0}
.cfg-input:focus{border-color:#93c5fd;background:#fff}
.cfg-test{border:1px solid var(--line);border-radius:6px;padding:4px 10px;font:inherit;font-size:11px;background:#fafbfc;color:var(--muted);cursor:pointer;flex-shrink:0}
.cfg-test:hover:not(:disabled){border-color:var(--brand);color:var(--brand)}
.cfg-hint{font-size:11px;color:var(--muted)}

/* Main area */
.main{flex:1;max-width:880px;width:100%;margin:0 auto;padding:32px 20px 60px;display:grid;gap:20px;align-content:start}
.main-empty{display:flex;flex-direction:column;align-items:center;justify-content:center;padding:60px 20px;text-align:center}
.main-empty h2{font-size:24px;font-weight:700;margin-bottom:8px}
.main-empty p{color:var(--muted);font-size:14px;max-width:400px;line-height:1.6}

/* Input panel */
.input-panel{background:var(--surface);border-radius:var(--radius);padding:16px;box-shadow:0 1px 3px rgba(0,0,0,.05)}
.input-panel textarea{width:100%;border:1px solid var(--line);border-radius:10px;background:#fafbfc;color:var(--ink);resize:vertical;min-height:88px;padding:10px 14px;line-height:1.6;font:inherit;font-size:14px;outline:none}
.input-panel textarea:focus{border-color:#93c5fd;background:#fff}
.input-bar{display:flex;align-items:center;gap:8px;margin-top:10px;flex-wrap:wrap}
.upload-btn{display:flex;align-items:center;gap:5px;border:1px solid var(--line);border-radius:8px;padding:6px 12px;font-size:12.5px;color:var(--muted);cursor:pointer;background:#fafbfc;white-space:nowrap}
.upload-btn:hover{border-color:#93c5fd;color:var(--ink)}
.upload-btn input{display:none}
.file-chip{padding:3px 8px;border-radius:5px;font-size:11px;background:var(--brand-soft);color:var(--brand);border:1px solid #bfdbfe;display:flex;align-items:center;gap:4px;max-width:180px;overflow:hidden}
.file-chip span{overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
.file-chip button{background:none;border:none;cursor:pointer;padding:0;color:var(--muted);font-size:12px;line-height:1;flex-shrink:0}
.file-chip button:hover{color:var(--bad)}
.spacer{flex:1}
.go-btn{border:none;border-radius:10px;padding:8px 24px;background:var(--brand);color:#fff;font:inherit;font-size:13px;font-weight:600;cursor:pointer;white-space:nowrap;transition:opacity 120ms,transform 120ms}
.go-btn:hover:not(:disabled){opacity:.88;transform:translateY(-1px)}
.go-btn:disabled{opacity:.35;cursor:not-allowed;transform:none}
.no-key-hint{font-size:11px;color:var(--muted)}
.no-key-hint button{background:none;border:none;color:var(--brand);cursor:pointer;font:inherit;font-size:11px;text-decoration:underline;padding:0}

/* Spinner */
@keyframes spin{to{transform:rotate(360deg)}}
.spin{display:inline-block;border-radius:50%;border:1.5px solid #e2e8f0;border-top-color:var(--brand);animation:spin .7s linear infinite;flex-shrink:0}

/* Progress strip */
.progress-strip{background:var(--surface);border:1px solid var(--line);border-radius:var(--radius);overflow:hidden}
.progress-bar{display:flex;align-items:center;padding:10px 14px;gap:8px}
.progress-line{display:flex;align-items:center;gap:8px;flex:1;min-width:0}
.progress-done-icon{color:var(--ok);font-size:14px;font-weight:700}
.progress-text{font-size:12.5px;font-weight:500;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.progress-trail{display:flex;gap:2px;font-size:12px;margin-left:auto;flex-shrink:0}
.trail-dot{opacity:.3;transition:opacity 200ms}
.trail-dot.trail-done{opacity:.7}
.progress-toggle{background:none;border:none;font:inherit;font-size:11px;color:var(--brand);cursor:pointer;white-space:nowrap;flex-shrink:0;padding:2px 6px}
.progress-toggle:hover{text-decoration:underline}
.progress-detail{border-top:1px solid var(--line);max-height:260px;overflow-y:auto;padding:6px 10px;display:grid;gap:2px;background:#fafbfc}
.pd-row{font-size:11.5px;padding:3px 6px;border-radius:4px;display:flex;align-items:center;gap:6px;min-height:22px}
.pd-thought{color:#64748b;font-style:italic}
.pd-tool{color:var(--ink)}
.pd-name{font-weight:600;font-size:11px;flex-shrink:0}
.pd-sub{flex:1;color:var(--muted);overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
.progress-detail::-webkit-scrollbar{width:5px}
.progress-detail::-webkit-scrollbar-thumb{background:#cbd5e1;border-radius:3px}

/* Notice */
.notice{border-radius:var(--radius);padding:10px 14px;font-size:13px;border:1px solid #fca5a5;background:#fef2f2;color:var(--bad)}
.empty-hint{text-align:center;padding:20px;color:var(--muted);font-size:13px}

/* Results */
.results-section{display:grid;gap:14px}
.results-head{display:flex;justify-content:space-between;align-items:baseline}
.results-head h2{font-size:18px;font-weight:700}
.results-head span{font-size:12px;color:var(--muted)}
.answer-bar{padding:10px 14px;background:#f0f9ff;border-left:4px solid #0ea5e9;border-radius:6px;font-size:13px;line-height:1.6;color:#0c4a6e}

/* Paper block (multi) */
.paper-block{border:1px solid var(--line);border-radius:var(--radius);overflow:hidden;background:var(--surface)}
.pb-head{width:100%;display:flex;align-items:center;gap:8px;padding:10px 14px;background:transparent;border:none;cursor:pointer;font:inherit;font-size:13px;text-align:left;color:var(--ink)}
.pb-head:hover{background:#f8fafc}
.pb-idx{font-size:10px;font-weight:700;color:#fff;background:#94a3b8;width:18px;height:18px;border-radius:50%;display:inline-flex;align-items:center;justify-content:center;flex-shrink:0}
.pb-name{font-weight:600;flex:1;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
.pb-count{font-size:12px;color:var(--muted);flex-shrink:0}
.pb-arr{color:#94a3b8;font-size:10px;flex-shrink:0}
.pb-body{padding:12px 14px;border-top:1px solid var(--line);display:grid;gap:12px}

/* Dataset card */
@keyframes fadeUp{from{opacity:0;transform:translateY(4px)}to{opacity:1;transform:none}}
.ds-card{background:var(--surface);border:1px solid var(--line);border-radius:var(--radius);padding:14px 16px;animation:fadeUp 200ms ease both}
.ds-top{display:flex;justify-content:space-between;align-items:flex-start;gap:10px;margin-bottom:6px;flex-wrap:wrap}
.ds-name{font-size:17px;font-weight:700;letter-spacing:-.01em}
.ds-badges{display:flex;gap:5px;align-items:center;flex-wrap:wrap}
.ds-badge{font-size:10.5px;padding:2px 7px;border-radius:99px;background:#f1f5f9;color:var(--muted);white-space:nowrap}
.ds-hf{color:var(--brand);background:var(--brand-soft);text-decoration:none;border:1px solid #bfdbfe}
.ds-hf:hover{text-decoration:underline}
.ds-conf-hi{background:#dcfce7;color:#166534}
.ds-conf-mid{background:#fef9c3;color:#854d0e}
.ds-conf-lo{background:#fee2e2;color:#991b1b}
.ds-splits{display:flex;gap:4px;flex-wrap:wrap;margin-bottom:6px}
.ds-split{border-radius:4px;padding:2px 7px;font-size:11px;font-weight:600;background:#f1f5f9;color:var(--muted)}
.ds-split-train{background:var(--ok-soft);color:var(--ok)}
.ds-reason{font-size:12.5px;color:#475569;margin-bottom:8px;line-height:1.5}
.ds-link-primary{display:flex;justify-content:space-between;align-items:center;gap:8px;text-decoration:none;color:inherit;border:1px solid #bfdbfe;border-radius:8px;padding:10px 12px;background:var(--brand-soft);margin-bottom:6px;font-size:13px}
.ds-link-primary:hover{border-color:var(--brand)}
.ds-lp-title{font-weight:600;color:var(--brand);font-size:13px}
.ds-lp-sub{font-size:11px;color:var(--muted);margin-top:1px}
.ds-link-minor{display:flex;align-items:center;gap:6px;text-decoration:none;padding:5px 0;border-bottom:1px solid var(--line);color:inherit;font-size:12px}
.ds-link-minor:last-of-type{border-bottom:none}
.ds-link-minor:hover .ds-lm-label{color:var(--brand)}
.ds-lm-src{font-size:10px;background:#f1f5f9;color:var(--muted);padding:1px 5px;border-radius:3px;flex-shrink:0}
.ds-lm-label{flex:1;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
.ds-more{font:inherit;font-size:11px;color:var(--brand);background:none;border:none;cursor:pointer;padding:3px 0}
.ds-more:hover{text-decoration:underline}

@media(max-width:600px){
  .topbar{padding:10px 14px}
  .main{padding:16px 12px 40px}
  .input-bar{flex-direction:column;align-items:stretch}
  .go-btn{width:100%}
}
      `}</style>

      {/* Top bar */}
      <header className="topbar">
        <div className="topbar-brand">
          Dataset Retrieval Agent
          <span className="topbar-sub">Paper → Dataset</span>
        </div>
        <span className="topbar-spacer" />
        <span className="topbar-keys">{configuredCount}/{ALL_KEY_FIELDS.length} API</span>
        <button className="settings-btn" onClick={() => setShowSettings(true)} title="API 配置">⚙</button>
      </header>

      <SettingsModal
        open={showSettings}
        onClose={() => setShowSettings(false)}
        keys={keys}
        onChange={handleKeyChange}
        serverKeys={serverKeys}
        onTest={testKey}
        testStatus={testStatus}
      />

      <div className="main">
        {/* Input */}
        <div className="input-panel">
          <textarea value={text} onChange={e => setText(e.target.value)}
            placeholder="描述你需要查找的数据集，或上传论文 PDF 让 Agent 自动提取…" />
          <div className="input-bar">
            <label className="upload-btn">
              <span>📎 上传 PDF（可多选）</span>
              <input type="file" accept=".pdf" multiple
                onChange={e => setPdfFiles(Array.from(e.target.files || []))} />
            </label>
            {pdfFiles.map((f, i) => (
              <div key={`${f.name}-${i}`} className="file-chip">
                <span>📄 {f.name}</span>
                <button onClick={() => setPdfFiles(p => p.filter((_, j) => j !== i))}>×</button>
              </div>
            ))}
            <span className="spacer" />
            {!hasKey && (
              <span className="no-key-hint">
                需先 <button onClick={() => setShowSettings(true)}>配置 API Key</button>
              </span>
            )}
            <button className="go-btn" disabled={!canRun} onClick={runSearch}>
              {running ? '检索中…' : '开始检索'}
            </button>
          </div>
        </div>

        {/* Progress — one strip per paper */}
        {entries.length > 0 && entries.map(([idx, run]) => (
          <ProgressStrip key={`p-${idx}`} run={run} running={running && !run.done} />
        ))}

        {error && <div className="notice">{error}</div>}

        {/* Results */}
        {hasResults && (
          <div className="results-section">
            <div className="results-head">
              <h2>检索结果</h2>
              {multi && <span>{entries.length} 篇论文</span>}
            </div>
            {entries.map(([idx, run]) =>
              run.results === null ? null : (
                <PaperBlock key={`r-${idx}`} index={Number(idx)} run={run} multi={multi} />
              )
            )}
          </div>
        )}

        {/* Empty state */}
        {entries.length === 0 && !running && (
          <div className="main-empty">
            <h2>Paper → Dataset</h2>
            <p>上传论文 PDF 或输入数据集相关问题，Agent 将自动从 HuggingFace、GitHub、Zenodo、PapersWithCode 等 7+ 个数据源并行检索。</p>
          </div>
        )}
      </div>
    </div>
  )
}
