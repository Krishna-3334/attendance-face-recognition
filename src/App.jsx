import { useState, useEffect, useRef, useCallback } from 'react'
import {
  Users,
  Camera,
  History,
  UserPlus,
  CheckCircle2,
  AlertCircle,
  Loader2,
  Trash2,
  ShieldCheck,
  ShieldAlert,
  Fingerprint,
  LayoutDashboard,
  Bell,
  Cpu,
  Activity
} from 'lucide-react'
import { motion, AnimatePresence } from 'framer-motion'
import { format } from 'date-fns'
import api, { captureFrame } from './api'

const TABS = {
  DASHBOARD: 'dashboard',
  ATTENDANCE: 'attendance',
  REGISTER: 'register',
  HISTORY: 'history',
  SECURITY: 'security'
}

const PROBE_INTERVAL_MS = 1500
const ENROL_FRAMES = 3
const ENROL_FRAME_GAP_MS = 700

export default function App() {
  const [activeTab, setActiveTab] = useState(TABS.DASHBOARD)
  const [config, setConfig] = useState(null)
  const [stats, setStats] = useState(null)
  const [users, setUsers] = useState([])
  const [attendance, setAttendance] = useState([])
  const [spoofEvents, setSpoofEvents] = useState([])
  const [loading, setLoading] = useState(true)
  const [backendError, setBackendError] = useState(null)
  const [notifications, setNotifications] = useState([])

  const addNotification = useCallback((title, message, type = 'success') => {
    const id = Date.now() + Math.random()
    setNotifications((prev) => [{ id, title, message, type }, ...prev].slice(0, 5))
    setTimeout(() => setNotifications((prev) => prev.filter((n) => n.id !== id)), 4500)
  }, [])

  const refresh = useCallback(async () => {
    const [s, u, a, e] = await Promise.all([
      api.stats(),
      api.users(),
      api.attendance(),
      api.spoofEvents()
    ])
    setStats(s)
    setUsers(u)
    setAttendance(a)
    setSpoofEvents(e)
  }, [])

  useEffect(() => {
    let cancelled = false
    ;(async () => {
      try {
        const cfg = await api.config()
        if (cancelled) return
        setConfig(cfg)
        await refresh()
        setBackendError(null)
      } catch (err) {
        if (!cancelled) setBackendError(err.message)
      } finally {
        if (!cancelled) setLoading(false)
      }
    })()
    return () => {
      cancelled = true
    }
  }, [refresh])

  if (loading) {
    return (
      <div className="flex flex-col items-center justify-center min-h-screen bg-black gap-4">
        <Loader2 className="w-12 h-12 text-blue-500 animate-spin" />
        <h2 className="text-2xl font-bold tracking-tight text-white">Connecting to recognition service</h2>
        <p className="text-white/60">Loading SCRFD detector and ArcFace embedding model…</p>
      </div>
    )
  }

  if (backendError) {
    return (
      <div className="flex flex-col items-center justify-center min-h-screen bg-black gap-4 p-6">
        <ShieldAlert className="w-12 h-12 text-red-500" />
        <h2 className="text-2xl font-bold tracking-tight text-white">Backend unreachable</h2>
        <p className="text-white/60 text-sm">{backendError}</p>
        <div className="card mt-4" style={{ maxWidth: 560 }}>
          <p className="text-sm text-white/70">
            Start the API, then reload:
          </p>
          <pre className="text-xs font-mono text-blue-400 mt-3">
{`cd backend
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python scripts/download_models.py
uvicorn app.main:app --port 8000`}
          </pre>
        </div>
      </div>
    )
  }

  return (
    <div className="main-container">
      <aside className="sidebar">
        <div className="flex items-center gap-3 mb-8">
          <div className="w-10 h-10 rounded-xl bg-blue-600 flex items-center justify-center shadow-lg">
            <ShieldCheck className="text-white" size={24} />
          </div>
          <div>
            <h1 className="font-bold text-xl tracking-tight">BioAccess</h1>
            <p className="text-xs text-white/40 uppercase tracking-widest">ArcFace v2.0</p>
          </div>
        </div>

        <nav className="flex flex-col gap-2">
          <SidebarLink icon={<LayoutDashboard size={20} />} label="Dashboard"
            active={activeTab === TABS.DASHBOARD} onClick={() => setActiveTab(TABS.DASHBOARD)} />
          <SidebarLink icon={<Camera size={20} />} label="Mark Attendance"
            active={activeTab === TABS.ATTENDANCE} onClick={() => setActiveTab(TABS.ATTENDANCE)} />
          <SidebarLink icon={<UserPlus size={20} />} label="Enrolment"
            active={activeTab === TABS.REGISTER} onClick={() => setActiveTab(TABS.REGISTER)} />
          <SidebarLink icon={<History size={20} />} label="Attendance Log"
            active={activeTab === TABS.HISTORY} onClick={() => setActiveTab(TABS.HISTORY)} />
          <SidebarLink icon={<ShieldAlert size={20} />} label="Rejected Attempts"
            active={activeTab === TABS.SECURITY} onClick={() => setActiveTab(TABS.SECURITY)} />
        </nav>

        <div className="mt-auto pt-6 border-t border-white/10">
          <div className="p-3 bg-white/5 rounded-2xl">
            <p className="text-xs text-white/40 uppercase tracking-widest mb-2">Operating point</p>
            <p className="text-sm font-mono text-blue-400">cos ≥ {config.match_threshold}</p>
            <p className="text-xs text-white/40 mt-1">
              liveness {config.liveness_enabled ? `≥ ${config.liveness_threshold}` : 'disabled'}
            </p>
          </div>
        </div>
      </aside>

      <main className="content-area">
        <header className="flex justify-between items-center bg-white/5 p-4 py-3 rounded-2xl backdrop-blur-md border border-white/10">
          <div className="flex items-center gap-3">
            <div className="px-3 py-1 rounded-full bg-green-500/10 border border-green-500/20 flex items-center gap-2">
              <div className="w-2 h-2 rounded-full bg-green-500 animate-pulse" />
              <span className="text-xs font-medium text-green-500">Service online</span>
            </div>
            <span className="text-xs text-white/40 font-mono">{config.detector} · {config.embedding_dim}-D ArcFace</span>
          </div>
          <div className="flex items-center gap-3">
            <div className="w-10 h-10 rounded-xl bg-white/5 border border-white/10 flex items-center justify-center relative">
              <Bell size={18} />
              {notifications.length > 0 && (
                <div className="absolute" style={{ top: 8, right: 8, width: 8, height: 8, borderRadius: 9999, background: '#ef4444' }} />
              )}
            </div>
            <p className="text-sm font-medium">{format(new Date(), 'EEEE, MMMM do')}</p>
          </div>
        </header>

        <AnimatePresence mode="wait">
          {activeTab === TABS.DASHBOARD && <DashboardView key="dash" stats={stats} config={config} />}
          {activeTab === TABS.ATTENDANCE && (
            <AttendanceView key="att" config={config} onEvent={refresh} addNotification={addNotification} />
          )}
          {activeTab === TABS.REGISTER && (
            <RegisterView key="reg" users={users} onChanged={refresh} addNotification={addNotification} />
          )}
          {activeTab === TABS.HISTORY && <HistoryView key="hist" attendance={attendance} />}
          {activeTab === TABS.SECURITY && <SecurityView key="sec" events={spoofEvents} config={config} />}
        </AnimatePresence>
      </main>

      <div className="fixed-toasts">
        <AnimatePresence>
          {notifications.map((n) => (
            <motion.div
              key={n.id}
              initial={{ opacity: 0, x: 50 }}
              animate={{ opacity: 1, x: 0 }}
              exit={{ opacity: 0, scale: 0.95 }}
              className={`toast ${n.type}`}
            >
              <div className={n.type === 'error' ? 'text-red-500' : 'text-blue-500'}>
                {n.type === 'error' ? <AlertCircle size={18} /> : <CheckCircle2 size={18} />}
              </div>
              <div>
                <h4 className="font-bold text-sm">{n.title}</h4>
                <p className="text-sm text-white/70">{n.message}</p>
              </div>
            </motion.div>
          ))}
        </AnimatePresence>
      </div>
    </div>
  )
}

function SidebarLink({ icon, label, active, onClick }) {
  return (
    <button onClick={onClick} className={`sidebar-link ${active ? 'active' : ''}`}>
      {icon}
      <span>{label}</span>
      {active && <motion.div layoutId="activeInd" className="sidebar-link-active" />}
    </button>
  )
}

// --- Dashboard ---------------------------------------------------------------

function DashboardView({ stats, config }) {
  return (
    <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} className="flex flex-col gap-6">
      <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
        <StatCard title="Enrolled identities" value={stats.registered_users}
          icon={<Users className="text-blue-500" />} subtitle="gallery templates" />
        <StatCard title="Present today" value={stats.attendance_today}
          icon={<CheckCircle2 className="text-green-500" />} subtitle="unique check-ins" />
        <StatCard title="Attacks blocked" value={stats.spoof_attempts_blocked}
          icon={<ShieldAlert className="text-red-500" />} subtitle="failed liveness" />
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        <div className="card">
          <div className="flex justify-between items-center mb-6">
            <h3 className="font-bold text-lg">Model card</h3>
            <Cpu size={18} className="text-blue-500" />
          </div>
          <dl className="flex flex-col gap-3">
            <Row label="Detector" value={config.detector} />
            <Row label="Embedding" value={config.embedding_model} />
            <Row label="Dimensionality" value={`${config.embedding_dim}-D, L2-normalised`} />
            <Row label="Match rule" value={`cosine ≥ ${config.match_threshold}, margin ≥ ${config.match_margin}`} />
            <Row label="Liveness" value={config.liveness_enabled
              ? `MiniFASNet, P(live) ≥ ${config.liveness_threshold}`
              : 'disabled'} />
            <Row label="Duplicate window" value={`${config.cooldown_seconds}s`} />
          </dl>
        </div>

        <div className="card">
          <div className="flex justify-between items-center mb-6">
            <h3 className="font-bold text-lg">Today's matching</h3>
            <Activity size={18} className="text-blue-500" />
          </div>
          <div className="flex flex-col gap-4">
            <Row label="Check-in events" value={stats.total_events} />
            <Row label="Mean similarity today"
              value={stats.avg_similarity_today === null ? '—' : stats.avg_similarity_today.toFixed(3)} />
            <div className="p-4 rounded-2xl bg-white/5 border border-white/10">
              <p className="text-sm text-white/70">
                The threshold above is not a default — it is the operating point selected from the
                FAR/FRR curves in <span className="font-mono text-blue-400">backend/eval/reports</span>.
                Re-run the evaluation after changing the gallery.
              </p>
            </div>
          </div>
        </div>
      </div>
    </motion.div>
  )
}

function Row({ label, value }) {
  return (
    <div className="flex justify-between items-center gap-4">
      <span className="text-sm text-white/40">{label}</span>
      <span className="text-sm font-medium" style={{ textAlign: 'right' }}>{value}</span>
    </div>
  )
}

function StatCard({ title, value, icon, subtitle }) {
  return (
    <div className="card flex flex-col gap-1">
      <div className="flex justify-between items-start mb-4">
        <div className="p-3 rounded-2xl bg-white/5 border border-white/10">{icon}</div>
      </div>
      <p className="text-sm text-white/40 font-medium uppercase tracking-wider">{title}</p>
      <h3 className="text-3xl font-bold tracking-tight">{value}</h3>
      {subtitle && <p className="text-xs text-white/40">{subtitle}</p>}
    </div>
  )
}

// --- Attendance --------------------------------------------------------------

const OUTCOME_STYLE = {
  match: { colour: '#22c55e', title: 'Access granted' },
  spoof: { colour: '#ef4444', title: 'Presentation attack blocked' },
  unknown: { colour: '#f59e0b', title: 'Not enrolled' },
  ambiguous: { colour: '#f59e0b', title: 'Ambiguous match — rejected' },
  cooldown: { colour: '#3b82f6', title: 'Already checked in' },
  no_face: { colour: '#64748b', title: 'No face in frame' }
}

function AttendanceView({ config, onEvent, addNotification }) {
  const videoRef = useRef(null)
  const timerRef = useRef(null)
  const inFlight = useRef(false)
  const [result, setResult] = useState(null)
  const [cameraError, setCameraError] = useState(null)

  useEffect(() => {
    let stream
    ;(async () => {
      try {
        stream = await navigator.mediaDevices.getUserMedia({ video: { width: 1280, height: 720 } })
        if (videoRef.current) videoRef.current.srcObject = stream
      } catch (err) {
        setCameraError(err.message)
      }
    })()
    return () => {
      if (timerRef.current) clearInterval(timerRef.current)
      if (stream) stream.getTracks().forEach((t) => t.stop())
    }
  }, [])

  const probe = useCallback(async () => {
    const video = videoRef.current
    if (!video || video.readyState < 2 || inFlight.current) return
    inFlight.current = true
    try {
      const blob = await captureFrame(video)
      const res = await api.recognise(blob)
      setResult(res)
      if (res.outcome === 'match') {
        addNotification('Check-in', `${res.name} recognised (cos ${res.similarity.toFixed(3)})`)
        onEvent()
      } else if (res.outcome === 'spoof') {
        addNotification('Blocked', `${res.liveness.attack_type} attack rejected`, 'error')
        onEvent()
      }
    } catch (err) {
      addNotification('Error', err.message, 'error')
    } finally {
      inFlight.current = false
    }
  }, [addNotification, onEvent])

  const handlePlay = () => {
    if (timerRef.current) return
    timerRef.current = setInterval(probe, PROBE_INTERVAL_MS)
  }

  const style = result ? OUTCOME_STYLE[result.outcome] ?? OUTCOME_STYLE.no_face : null

  return (
    <motion.div initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} className="flex flex-col items-center gap-6">
      <div className="video-frame">
        {cameraError ? (
          <div className="flex flex-col items-center justify-center h-full gap-3 p-6">
            <AlertCircle className="text-red-500" size={32} />
            <p className="text-white/60 text-sm">Camera unavailable: {cameraError}</p>
          </div>
        ) : (
          <video ref={videoRef} autoPlay muted playsInline onPlay={handlePlay} className="w-full h-full" style={{ objectFit: 'cover' }} />
        )}

        <div className="scan-overlay">
          <div className="scan-box">
            <motion.div
              animate={{ top: ['8%', '92%', '8%'] }}
              transition={{ duration: 4, repeat: Infinity, ease: 'linear' }}
              className="scan-line"
            />
          </div>
        </div>

        <AnimatePresence>
          {style && result.outcome !== 'no_face' && (
            <motion.div
              key={`${result.outcome}-${result.latency_ms}`}
              initial={{ opacity: 0, scale: 0.9 }}
              animate={{ opacity: 1, scale: 1 }}
              exit={{ opacity: 0, scale: 0.9 }}
              className="result-banner"
              style={{ background: `${style.colour}22`, borderColor: style.colour }}
            >
              <p className="font-bold text-lg" style={{ color: style.colour }}>{style.title}</p>
              {result.name && <p className="text-2xl font-bold tracking-tight">{result.name}</p>}
              {result.detail && <p className="text-xs text-white/60">{result.detail}</p>}
            </motion.div>
          )}
        </AnimatePresence>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-3 gap-6 w-full">
        <MetricCard icon={<Fingerprint size={22} className="text-blue-500" />} label="Cosine similarity"
          value={result ? result.similarity.toFixed(3) : '—'}
          sub={result ? `runner-up ${result.runner_up.toFixed(3)} · threshold ${config.match_threshold}` : 'awaiting probe'} />
        <MetricCard icon={<ShieldCheck size={22} className="text-purple-500" />} label="Liveness P(live)"
          value={result?.liveness ? result.liveness.live_score.toFixed(3) : '—'}
          sub={result?.liveness ? `${result.liveness.attack_type} · threshold ${config.liveness_threshold}` : 'MiniFASNet'} />
        <MetricCard icon={<Activity size={22} className="text-green-500" />} label="Round-trip latency"
          value={result ? `${result.latency_ms} ms` : '—'} sub="detect + liveness + match" />
      </div>
    </motion.div>
  )
}

function MetricCard({ icon, label, value, sub }) {
  return (
    <div className="card p-4 flex items-center gap-4">
      <div className="p-3 rounded-xl bg-white/5">{icon}</div>
      <div>
        <p className="text-xs text-white/40 font-bold uppercase tracking-widest">{label}</p>
        <p className="text-lg font-bold font-mono">{value}</p>
        <p className="text-xs text-white/40">{sub}</p>
      </div>
    </div>
  )
}

// --- Enrolment ---------------------------------------------------------------

function RegisterView({ users, onChanged, addNotification }) {
  const [name, setName] = useState('')
  const [capturing, setCapturing] = useState(false)
  const [busy, setBusy] = useState(false)
  const [progress, setProgress] = useState(0)
  const [lastEnrolment, setLastEnrolment] = useState(null)
  const videoRef = useRef(null)
  const streamRef = useRef(null)

  const stopCamera = useCallback(() => {
    if (streamRef.current) {
      streamRef.current.getTracks().forEach((t) => t.stop())
      streamRef.current = null
    }
    setCapturing(false)
  }, [])

  useEffect(() => stopCamera, [stopCamera])

  const startCapture = async () => {
    if (!name.trim()) {
      addNotification('Error', 'Enter a name first.', 'error')
      return
    }
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ video: { width: 1280, height: 720 } })
      streamRef.current = stream
      setCapturing(true)
      requestAnimationFrame(() => {
        if (videoRef.current) videoRef.current.srcObject = stream
      })
    } catch (err) {
      addNotification('Error', `Camera unavailable: ${err.message}`, 'error')
    }
  }

  const enrol = async () => {
    const video = videoRef.current
    if (!video) return
    setBusy(true)
    setProgress(0)
    try {
      const blobs = []
      for (let i = 0; i < ENROL_FRAMES; i += 1) {
        blobs.push(await captureFrame(video))
        setProgress(Math.round(((i + 1) / (ENROL_FRAMES + 1)) * 100))
        if (i < ENROL_FRAMES - 1) await new Promise((r) => setTimeout(r, ENROL_FRAME_GAP_MS))
      }
      const res = await api.enrol(name.trim(), blobs)
      setProgress(100)
      if (!res.ok) {
        addNotification('Enrolment failed', res.detail ?? 'no usable frames', 'error')
        setLastEnrolment(res)
      } else {
        addNotification('Enrolled', `${res.user.name} — ${res.accepted_samples} frames accepted`)
        setLastEnrolment(res)
        setName('')
        stopCamera()
        onChanged()
      }
    } catch (err) {
      addNotification('Error', err.message, 'error')
    } finally {
      setBusy(false)
      setTimeout(() => setProgress(0), 600)
    }
  }

  const removeUser = async (user) => {
    if (!window.confirm(`Remove ${user.name} from the gallery?`)) return
    try {
      await api.deleteUser(user.id)
      addNotification('Removed', `${user.name} deleted.`, 'info')
      onChanged()
    } catch (err) {
      addNotification('Error', err.message, 'error')
    }
  }

  return (
    <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} className="flex flex-col gap-6" style={{ maxWidth: 720, margin: '0 auto', width: '100%' }}>
      <div className="card">
        <div className="flex items-center gap-3 mb-6">
          <UserPlus className="text-blue-500" size={24} />
          <h3 className="font-bold text-xl">Enrol identity</h3>
        </div>

        <label className="text-xs font-bold uppercase text-white/40 tracking-widest">Name / ID</label>
        <input
          type="text"
          className="text-input"
          placeholder="e.g. Krishna Kaushik"
          value={name}
          onChange={(e) => setName(e.target.value)}
        />

        {!capturing ? (
          <button className="btn btn-primary w-full" onClick={startCapture}>Start capture</button>
        ) : (
          <div className="flex flex-col gap-4">
            <div className="enrol-preview">
              <video ref={videoRef} autoPlay muted playsInline className="w-full h-full" style={{ objectFit: 'cover' }} />
            </div>
            <p className="text-xs text-white/40">
              {ENROL_FRAMES} frames are captured {ENROL_FRAME_GAP_MS} ms apart and averaged into one template —
              vary your pose slightly between them.
            </p>
            <div className="flex gap-4">
              <button onClick={stopCamera} className="btn btn-outline flex-1" disabled={busy}>Cancel</button>
              <button onClick={enrol} className="btn btn-primary flex-1" disabled={busy}>
                {busy ? <Loader2 size={18} className="animate-spin" /> : `Capture ${ENROL_FRAMES} frames`}
              </button>
            </div>
            {progress > 0 && (
              <div className="progress-track"><div className="progress-fill" style={{ width: `${progress}%` }} /></div>
            )}
          </div>
        )}

        {lastEnrolment && (
          <div className="p-4 rounded-2xl bg-white/5 border border-white/10 mt-4">
            <p className="text-sm font-bold mb-2">Last enrolment</p>
            <Row label="Frames accepted" value={lastEnrolment.accepted_samples} />
            {lastEnrolment.intra_class_similarity !== null && lastEnrolment.intra_class_similarity !== undefined && (
              <Row label="Intra-class similarity" value={lastEnrolment.intra_class_similarity.toFixed(3)} />
            )}
            {lastEnrolment.rejected_samples?.length > 0 && (
              <ul className="text-xs text-white/40 mt-2">
                {lastEnrolment.rejected_samples.map((r) => <li key={r}>· {r}</li>)}
              </ul>
            )}
          </div>
        )}
      </div>

      <div className="card">
        <h3 className="font-bold mb-6 flex items-center gap-2">
          <ShieldCheck size={18} className="text-blue-500" />
          Gallery ({users.length})
        </h3>
        <div className="flex flex-col gap-3">
          {users.length === 0 ? (
            <div className="empty-state">No identities enrolled yet.</div>
          ) : (
            users.map((user) => (
              <div key={user.id} className="list-row">
                <div className="flex items-center gap-4">
                  <div className="w-10 h-10 rounded-xl bg-blue-500/10 flex items-center justify-center text-blue-500">
                    <Users size={18} />
                  </div>
                  <div>
                    <p className="font-bold">{user.name}</p>
                    <p className="text-xs text-white/40 font-mono">
                      {user.n_samples} frame template · enrolled {format(new Date(user.created_at), 'MMM d, HH:mm')}
                    </p>
                  </div>
                </div>
                <button onClick={() => removeUser(user)} className="icon-btn"><Trash2 size={16} /></button>
              </div>
            ))
          )}
        </div>
      </div>
    </motion.div>
  )
}

// --- Logs --------------------------------------------------------------------

function HistoryView({ attendance }) {
  return (
    <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} className="card">
      <div className="flex items-center gap-3 mb-6">
        <History className="text-blue-500" size={24} />
        <h3 className="font-bold text-xl">Attendance log</h3>
      </div>
      {attendance.length === 0 ? (
        <div className="empty-state">No check-ins recorded.</div>
      ) : (
        <div style={{ overflowX: 'auto' }}>
          <table className="data-table">
            <thead>
              <tr>
                <th>Subject</th><th>Timestamp</th><th>Cosine</th><th>P(live)</th><th style={{ textAlign: 'right' }}>Status</th>
              </tr>
            </thead>
            <tbody>
              {attendance.map((log) => (
                <tr key={log.id}>
                  <td className="font-bold text-sm">{log.name}</td>
                  <td className="text-sm text-white/60">{format(new Date(log.timestamp), 'MMM d, HH:mm:ss')}</td>
                  <td className="text-sm font-mono text-blue-400">{log.similarity.toFixed(3)}</td>
                  <td className="text-sm font-mono text-purple-500">{log.liveness === null ? '—' : log.liveness.toFixed(3)}</td>
                  <td style={{ textAlign: 'right' }}><span className="pill pill-green">{log.status}</span></td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </motion.div>
  )
}

function SecurityView({ events, config }) {
  return (
    <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} className="flex flex-col gap-6">
      <div className="card">
        <div className="flex items-center gap-3 mb-6">
          <ShieldAlert className="text-red-500" size={24} />
          <h3 className="font-bold text-xl">Rejected presentation attempts</h3>
        </div>
        <p className="text-sm text-white/60 mb-6">
          Every probe is scored for liveness <em>before</em> it is compared with the gallery, so a photo or a
          phone screen can never produce an attendance record. Rejections below are logged with the identity
          they would otherwise have matched. Current threshold: P(live) ≥ {config.liveness_threshold}.
        </p>
        {events.length === 0 ? (
          <div className="empty-state">No presentation attacks recorded.</div>
        ) : (
          <div style={{ overflowX: 'auto' }}>
            <table className="data-table">
              <thead>
                <tr><th>Timestamp</th><th>Attack type</th><th>P(live)</th><th>Would have matched</th></tr>
              </thead>
              <tbody>
                {events.map((e) => (
                  <tr key={e.id}>
                    <td className="text-sm text-white/60">{format(new Date(e.timestamp), 'MMM d, HH:mm:ss')}</td>
                    <td><span className="pill pill-red">{e.attack_type}</span></td>
                    <td className="text-sm font-mono">{e.live_score.toFixed(3)}</td>
                    <td className="text-sm">{e.matched_name ?? '—'}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </div>
    </motion.div>
  )
}
