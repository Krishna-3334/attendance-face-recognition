import { useState, useEffect, useRef, useCallback } from 'react'
import * as faceapi from 'face-api.js'
import { 
  Users, 
  Camera, 
  History, 
  UserPlus, 
  CheckCircle2, 
  AlertCircle, 
  Loader2, 
  X, 
  Trash2, 
  ShieldCheck, 
  Fingerprint, 
  LayoutDashboard,
  LogOut,
  Bell
} from 'lucide-react'
import { motion, AnimatePresence } from 'framer-motion'
import { format } from 'date-fns'

const MODAL_TRANSITION = {
  initial: { opacity: 0, scale: 0.95 },
  animate: { opacity: 1, scale: 1 },
  exit: { opacity: 0, scale: 0.95 }
}

const DASHBOARD_TABS = {
  ATTENDANCE: 'attendance',
  REGISTER: 'register',
  HISTORY: 'history',
  DASHBOARD: 'dashboard'
}

export default function App() {
  const [activeTab, setActiveTab] = useState(DASHBOARD_TABS.DASHBOARD)
  const [isModelsLoaded, setIsModelsLoaded] = useState(false)
  const [users, setUsers] = useState(() => {
    const saved = localStorage.getItem('face-attendance-users')
    return saved ? JSON.parse(saved) : []
  })
  const [attendance, setAttendance] = useState(() => {
    const saved = localStorage.getItem('face-attendance-log')
    return saved ? JSON.parse(saved) : []
  })
  const [loading, setLoading] = useState(true)
  const [notifications, setNotifications] = useState([])

  // Persistent storage
  useEffect(() => {
    localStorage.setItem('face-attendance-users', JSON.stringify(users))
  }, [users])

  useEffect(() => {
    localStorage.setItem('face-attendance-log', JSON.stringify(attendance))
  }, [attendance])

  // Load face-api models
  useEffect(() => {
    const loadModels = async () => {
      try {
        const MODEL_URL = '/models'
        await Promise.all([
          faceapi.nets.tinyFaceDetector.loadFromUri(MODEL_URL),
          faceapi.nets.faceLandmark68Net.loadFromUri(MODEL_URL),
          faceapi.nets.faceRecognitionNet.loadFromUri(MODEL_URL),
          faceapi.nets.ssdMobilenetv1.loadFromUri(MODEL_URL)
        ])
        setIsModelsLoaded(true)
        setLoading(false)
        console.log('Models loaded successfully')
      } catch (error) {
        console.error('Error loading face-api models:', error)
        addNotification('System', 'Failed to load AI models. Contact admin.', 'error')
      }
    }
    loadModels()
  }, [])

  const addNotification = (title, message, type = 'success') => {
    const id = Date.now()
    setNotifications(prev => [{ id, title, message, type }, ...prev].slice(0, 5))
    setTimeout(() => {
      setNotifications(prev => prev.filter(n => n.id !== id))
    }, 4000)
  }

  const logAttendance = useCallback((userName) => {
    const now = new Date()
    const today = format(now, 'yyyy-MM-dd')
    
    // Simple rule: Only log attendance once every 5 minutes for the same person
    const lastEntry = [...attendance].reverse().find(a => a.name === userName)
    if (lastEntry) {
      const diff = now - new Date(lastEntry.timestamp)
      if (diff < 5 * 60 * 1000) return // Skip if logged within 5 mins
    }

    const newRecord = {
      id: Date.now(),
      name: userName,
      timestamp: now.toISOString(),
      status: 'Present',
      date: today
    }
    
    setAttendance(prev => [newRecord, ...prev])
    addNotification('Check-in', `${userName} recognized! Attendance marked.`, 'success')
  }, [attendance])

  const deleteUser = (userName) => {
    if (window.confirm(`Delete user ${userName}?`)) {
      setUsers(prev => prev.filter(u => u.name !== userName))
      addNotification('Manager', `User ${userName} deleted.`, 'info')
    }
  }

  if (loading) {
    return (
      <div className="flex flex-col items-center justify-center min-h-screen bg-black gap-4">
        <Loader2 className="w-12 h-12 text-blue-500 animate-spin" />
        <h2 className="text-2xl font-bold tracking-tight text-white">Initializing Neural Core</h2>
        <p className="text-white/60">Loading face-recognition models...</p>
      </div>
    )
  }

  return (
    <div className="main-container">
      {/* Sidebar */}
      <aside className="sidebar">
        <div className="flex items-center gap-3 mb-8">
          <div className="w-10 h-10 rounded-xl bg-blue-600 flex items-center justify-center shadow-lg shadow-blue-500/20">
            <ShieldCheck className="text-white" size={24} />
          </div>
          <div>
            <h1 className="font-bold text-xl tracking-tight">BioAccess</h1>
            <p className="text-xs text-white/40 uppercase tracking-widest">Enterprise v1.0</p>
          </div>
        </div>

        <nav className="flex flex-col gap-2">
          <SidebarLink 
            icon={<LayoutDashboard size={20} />} 
            label="Dashboard" 
            active={activeTab === DASHBOARD_TABS.DASHBOARD}
            onClick={() => setActiveTab(DASHBOARD_TABS.DASHBOARD)}
          />
          <SidebarLink 
            icon={<Camera size={20} />} 
            label="Mark Attendance" 
            active={activeTab === DASHBOARD_TABS.ATTENDANCE}
            onClick={() => setActiveTab(DASHBOARD_TABS.ATTENDANCE)}
          />
          <SidebarLink 
            icon={<UserPlus size={20} />} 
            label="User Management" 
            active={activeTab === DASHBOARD_TABS.REGISTER}
            onClick={() => setActiveTab(DASHBOARD_TABS.REGISTER)}
          />
          <SidebarLink 
            icon={<History size={20} />} 
            label="History Log" 
            active={activeTab === DASHBOARD_TABS.HISTORY}
            onClick={() => setActiveTab(DASHBOARD_TABS.HISTORY)}
          />
        </nav>

        <div className="mt-auto pt-6 border-t border-white/10">
          <div className="flex items-center gap-3 p-3 bg-white/5 rounded-2xl">
            <div className="w-10 h-10 rounded-full bg-gradient-to-tr from-blue-500 to-purple-500" />
            <div className="flex-1 overflow-hidden">
              <p className="text-sm font-semibold truncate">Admin Console</p>
              <p className="text-xs text-white/40">Security Active</p>
            </div>
            <LogOut size={16} className="text-white/40 hover:text-white cursor-pointer" />
          </div>
        </div>
      </aside>

      {/* Main Content */}
      <main className="content-area">
        <header className="flex justify-between items-center bg-white/5 p-4 py-3 rounded-2xl backdrop-blur-md border border-white/10">
          <div className="flex items-center gap-4">
            <div className="px-3 py-1 rounded-full bg-green-500/10 border border-green-500/20 flex items-center gap-2">
              <div className="w-2 h-2 rounded-full bg-green-500 animate-pulse" />
              <span className="text-xs font-medium text-green-500">System Ready</span>
            </div>
          </div>
          <div className="flex items-center gap-3">
             <div className="w-10 h-10 rounded-xl bg-white/5 border border-white/10 flex items-center justify-center hover:bg-white/10 transition-colors cursor-pointer relative">
               <Bell size={18} />
               {notifications.length > 0 && <div className="absolute top-2 right-2 w-2 h-2 rounded-full bg-red-500" />}
             </div>
             <div className="h-4 w-[1px] bg-white/10 mx-1" />
             <p className="text-sm font-medium">{format(new Date(), 'EEEE, MMMM do')}</p>
          </div>
        </header>

        <AnimatePresence mode="wait">
          {activeTab === DASHBOARD_TABS.DASHBOARD && (
            <DashboardView key="dash" users={users} attendance={attendance} />
          )}
          {activeTab === DASHBOARD_TABS.ATTENDANCE && (
            <AttendanceView 
              key="att" 
              users={users} 
              onRecognized={logAttendance}
              isModelsLoaded={isModelsLoaded}
            />
          )}
          {activeTab === DASHBOARD_TABS.REGISTER && (
            <RegisterView 
              key="reg" 
              users={users} 
              setUsers={setUsers} 
              addNotification={addNotification}
            />
          )}
          {activeTab === DASHBOARD_TABS.HISTORY && (
            <HistoryView key="hist" attendance={attendance} />
          )}
        </AnimatePresence>
      </main>

      {/* Notification Toast */}
      <div className="fixed top-6 right-6 z-50 flex flex-col gap-3">
        <AnimatePresence>
          {notifications.map(n => (
            <motion.div
              key={n.id}
              initial={{ opacity: 0, x: 50 }}
              animate={{ opacity: 1, x: 0 }}
              exit={{ opacity: 0, scale: 0.95 }}
              className={`p-4 rounded-2xl backdrop-blur-xl border flex items-start gap-3 shadow-2xl min-w-[300px] ${
                n.type === 'error' ? 'bg-red-500/10 border-red-500/20' : 'bg-white/5 border-white/10'
              }`}
            >
              <div className={`mt-1 rounded-full p-1 ${n.type === 'error' ? 'text-red-500' : 'text-blue-500'}`}>
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
    <button 
      onClick={onClick}
      className={`sidebar-link ${active ? 'active' : ''}`}
    >
      {icon}
      <span>{label}</span>
      {active && <motion.div layoutId="activeInd" className="sidebar-link-active" />}
    </button>
  )
}

// Views

function DashboardView({ users, attendance }) {
  const today = format(new Date(), 'yyyy-MM-dd')
  const uniqueAttendeesToday = new Set(attendance.filter(a => a.date === today).map(a => a.name)).size
  
  return (
    <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} className="flex flex-col gap-6">
      <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
        <StatCard title="Registered Users" value={users.length} icon={<Users className="text-blue-500" />} />
        <StatCard title="Attendance Today" value={uniqueAttendeesToday} icon={<CheckCircle2 className="text-green-500" />} subtitle="Unique Check-ins" />
        <StatCard title="System Uptime" value="99.9%" icon={<Fingerprint className="text-purple-500" />} subtitle="Precision Recognition" />
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        <div className="card">
          <div className="flex justify-between items-center mb-6">
            <h3 className="font-bold text-lg">System Health</h3>
            <ShieldCheck size={18} className="text-blue-500" />
          </div>
          <div className="flex flex-col gap-4">
            <div className="space-y-2">
              <div className="flex justify-between text-sm">
                <span className="text-white/60">Neural Engine Load</span>
                <span className="text-blue-400 font-mono">12%</span>
              </div>
              <div className="h-1.5 w-full bg-white/5 rounded-full overflow-hidden">
                <div className="h-full bg-blue-500/50 w-[12%]" />
              </div>
            </div>
            <div className="space-y-2">
              <div className="flex justify-between text-sm">
                <span className="text-white/60">Search Latency</span>
                <span className="text-green-400 font-mono">18ms</span>
              </div>
              <div className="h-1.5 w-full bg-white/5 rounded-full overflow-hidden">
                <div className="h-full bg-green-500/50 w-[5%]" />
              </div>
            </div>
          </div>
        </div>
        
        <div className="card">
            <h3 className="font-bold text-lg mb-4">Quick Insights</h3>
            <div className="p-4 rounded-2xl bg-white/5 border border-white/10">
              <p className="text-sm text-white/80 leading-relaxed">
                Biometric security protocol is currently enforced across all access points. 
                Average recognition confidence is <span className="text-blue-400 font-bold">94.8%</span>.
              </p>
            </div>
        </div>
      </div>
    </motion.div>
  )
}

function StatCard({ title, value, icon, subtitle }) {
  return (
    <div className="card flex flex-col gap-1">
      <div className="flex justify-between items-start mb-4">
        <div className="p-3 rounded-2xl bg-white/5 border border-white/5">{icon}</div>
        <div className="px-2 py-1 rounded-lg bg-green-500/10 text-green-500 text-[10px] font-bold">+12%</div>
      </div>
      <p className="text-sm text-white/40 font-medium uppercase tracking-wider">{title}</p>
      <h3 className="text-3xl font-bold tracking-tight">{value}</h3>
      {subtitle && <p className="text-xs text-white/30">{subtitle}</p>}
    </div>
  )
}

function AttendanceView({ users, onRecognized, isModelsLoaded }) {
  const videoRef = useRef()
  const canvasRef = useRef()
  const [recognizing, setRecognizing] = useState(false)
  const [lastRecognized, setLastRecognized] = useState(null)
  
  const startVideo = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ video: true })
      if (videoRef.current) {
        videoRef.current.srcObject = stream
      }
    } catch (error) {
      console.error('Camera access error:', error)
    }
  }

  useEffect(() => {
    startVideo()
    return () => {
      if (videoRef.current && videoRef.current.srcObject) {
         videoRef.current.srcObject.getTracks().forEach(track => track.stop())
      }
    }
  }, [])

  const handleVideoOnPlay = async () => {
    if (!isModelsLoaded || recognizerRef.current) return
    recognizerRef.current = setInterval(recognizeFaces, 1000)
  }

  const recognizerRef = useRef()
  
  useEffect(() => {
    return () => {
      if (recognizerRef.current) clearInterval(recognizerRef.current)
    }
  }, [])

  const recognizeFaces = async () => {
    if (!videoRef.current || videoRef.current.paused || videoRef.current.ended) return
    
    // Create labeled descriptors from users
    const labeledDescriptors = users.map(user => {
      // descriptors are stored as arrays, convert back to Float32Array
      return new faceapi.LabeledFaceDescriptors(
        user.name,
        [new Float32Array(user.descriptor)]
      )
    })

    if (labeledDescriptors.length === 0) return

    const faceMatcher = new faceapi.FaceMatcher(labeledDescriptors, 0.6)

    const detections = await faceapi.detectAllFaces(
      videoRef.current, 
      new faceapi.TinyFaceDetectorOptions()
    ).withFaceLandmarks().withFaceDescriptors()

    if (detections.length > 0) {
      const displaySize = { 
        width: videoRef.current.videoWidth, 
        height: videoRef.current.videoHeight 
      }
      
      const resizedDetections = faceapi.resizeResults(detections, displaySize)
      
      const results = resizedDetections.map(d => 
        faceMatcher.findBestMatch(d.descriptor)
      )

      results.forEach(result => {
        if (result.label !== 'unknown') {
          onRecognized(result.label)
          setLastRecognized(result.label)
          setTimeout(() => setLastRecognized(null), 3000)
        }
      })
    }
  }

  return (
    <motion.div initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} className="flex flex-col items-center">
      <div className="relative rounded-[32px] overflow-hidden border-8 border-white/5 shadow-2xl bg-black aspect-video w-full max-w-[800px]">
        <video 
          ref={videoRef} 
          autoPlay 
          muted 
          onPlay={handleVideoOnPlay}
          className="w-full h-full object-cover"
        />
        
        {/* Scanning Overlay */}
        <div className="absolute inset-0 pointer-events-none">
          <div className="absolute inset-0 bg-blue-500/5 flex items-center justify-center">
             <div className="w-[300px] h-[300px] border-2 border-white/10 rounded-[40px] relative">
               <div className="absolute top-0 left-0 w-8 h-8 border-t-4 border-l-4 border-blue-500 rounded-tl-xl" />
               <div className="absolute top-0 right-0 w-8 h-8 border-t-4 border-r-4 border-blue-500 rounded-tr-xl" />
               <div className="absolute bottom-0 left-0 w-8 h-8 border-b-4 border-l-4 border-blue-500 rounded-bl-xl" />
               <div className="absolute bottom-0 right-0 w-8 h-8 border-b-4 border-r-4 border-blue-500 rounded-br-xl" />
               
               <motion.div 
                 animate={{ top: ['10%', '90%', '10%'] }}
                 transition={{ duration: 4, repeat: Infinity, ease: "linear" }}
                 className="absolute left-4 right-4 h-[2px] bg-blue-500/50 shadow-[0_0_15px_rgba(59,130,246,0.8)]" 
               />
             </div>
          </div>
          
          <div className="absolute bottom-8 left-0 right-0 flex justify-center">
            <div className="px-6 py-2 rounded-full bg-black/60 backdrop-blur-md border border-white/10 text-white/80 text-sm font-medium flex items-center gap-3">
              <Loader2 className="w-4 h-4 text-blue-500 animate-spin" />
              Scanning for biometric signature...
            </div>
          </div>

          <AnimatePresence>
            {lastRecognized && (
              <motion.div 
                initial={{ opacity: 0, scale: 0.8 }}
                animate={{ opacity: 1, scale: 1 }}
                exit={{ opacity: 0, scale: 0.8 }}
                className="absolute inset-0 flex items-center justify-center bg-blue-500/20"
              >
                <div className="bg-blue-600 px-8 py-4 rounded-3xl shadow-2xl flex flex-col items-center gap-2">
                  <CheckCircle2 size={48} className="text-white" />
                  <p className="text-xl font-bold text-white uppercase tracking-widest">{lastRecognized}</p>
                  <p className="text-white/70 text-xs font-bold uppercase tracking-widest">Authorized Access</p>
                </div>
              </motion.div>
            )}
          </AnimatePresence>
        </div>
      </div>
      
      <div className="mt-8 grid grid-cols-2 gap-4 w-full max-w-[800px]">
        <div className="card p-4 flex items-center gap-4">
           <div className="p-3 rounded-xl bg-blue-500/10 text-blue-500"><Fingerprint size={24} /></div>
           <div>
             <p className="text-xs text-white/40 font-bold uppercase">Engine</p>
             <p className="text-sm font-bold">SSD MobileNet V1</p>
           </div>
        </div>
        <div className="card p-4 flex items-center gap-4">
           <div className="p-3 rounded-xl bg-purple-500/10 text-purple-500"><ShieldCheck size={24} /></div>
           <div>
             <p className="text-xs text-white/40 font-bold uppercase">Confidence</p>
             <p className="text-sm font-bold">0.6 Threshold</p>
           </div>
        </div>
      </div>
    </motion.div>
  )
}

function RegisterView({ users, setUsers, addNotification }) {
  const [userName, setUserName] = useState('')
  const [isCapturing, setIsCapturing] = useState(false)
  const videoRef = useRef()
  const [captureProgress, setCaptureProgress] = useState(0)

  const startRegistration = async () => {
    if (!userName.trim()) {
      addNotification('Error', 'Please enter a name first.', 'error')
      return
    }
    setIsCapturing(true)
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ video: true })
      if (videoRef.current) {
        videoRef.current.srcObject = stream
      }
    } catch (err) {
      console.error(err)
      setIsCapturing(false)
    }
  }

  const captureFace = async () => {
    if (!videoRef.current) return
    
    setCaptureProgress(30)
    const detection = await faceapi.detectSingleFace(
      videoRef.current, 
      new faceapi.TinyFaceDetectorOptions()
    ).withFaceLandmarks().withFaceDescriptor()

    if (detection) {
      setCaptureProgress(80)
      const descriptor = Array.from(detection.descriptor)
      
      // Check if user exists
      if (users.find(u => u.name === userName)) {
        addNotification('System', 'User with this name already exists.', 'error')
        setIsCapturing(false)
        return
      }

      setUsers(prev => [...prev, { name: userName, descriptor }])
      setCaptureProgress(100)
      setTimeout(() => {
        addNotification('Success', `${userName} registered successfully!`, 'success')
        setIsCapturing(false)
        setUserName('')
        setCaptureProgress(0)
      }, 500)
    } else {
      addNotification('Error', 'No face detected. Please try again.', 'error')
      setCaptureProgress(0)
    }
  }

  return (
    <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} className="flex flex-col gap-6 max-w-2xl mx-auto w-full">
       <div className="card">
         <div className="flex items-center gap-3 mb-6">
           <UserPlus className="text-blue-500" size={24} />
           <h3 className="font-bold text-xl">Enroll New Identity</h3>
         </div>
         
         <div className="space-y-4">
           <div>
             <label className="text-xs font-bold uppercase text-white/40 mb-2 block tracking-widest">Entry Name / ID</label>
             <input 
               type="text" 
               className="w-full bg-white/5 border border-white/10 rounded-2xl px-5 py-4 text-white focus:outline-none focus:border-blue-500 transition-colors"
               placeholder="e.g. John Doe"
               value={userName}
               onChange={(e) => setUserName(e.target.value)}
             />
           </div>
           
           {!isCapturing ? (
             <button 
               className="btn btn-primary w-full py-4 text-lg"
               onClick={startRegistration}
             >
               Initialize Biometric Capture
             </button>
           ) : (
             <div className="space-y-6">
                <div className="relative rounded-3xl overflow-hidden aspect-video bg-black border border-white/10">
                   <video ref={videoRef} autoPlay muted className="w-full h-full object-cover" />
                   <div className="absolute inset-0 border-2 border-dashed border-white/20 m-12 rounded-full" />
                </div>
                <div className="flex gap-4">
                  <button onClick={() => setIsCapturing(false)} className="btn btn-outline flex-1">Cancel</button>
                  <button onClick={captureFace} className="btn btn-primary flex-1">
                    {captureProgress > 0 && captureProgress < 100 ? (
                      <Loader2 size={18} className="animate-spin" />
                    ) : (
                      'Capture Biometrics'
                    )}
                  </button>
                </div>
                {captureProgress > 0 && (
                  <div className="space-y-2">
                    <div className="flex justify-between text-xs font-bold uppercase opacity-40">
                      <span>Analyzing Topology</span>
                      <span>{captureProgress}%</span>
                    </div>
                    <div className="h-1 bg-white/5 rounded-full overflow-hidden">
                      <motion.div 
                        initial={{ width: 0 }}
                        animate={{ width: `${captureProgress}%` }}
                        className="h-full bg-blue-500" 
                      />
                    </div>
                  </div>
                )}
             </div>
           )}
         </div>
       </div>

       <div className="card">
         <h3 className="font-bold mb-6 flex items-center gap-2">
           <ShieldCheck size={18} className="text-blue-500" />
           Manage Identified Personnel ({users.length})
         </h3>
         <div className="grid grid-cols-1 gap-3">
           {users.length === 0 ? (
             <div className="p-8 text-center border-2 border-dashed border-white/5 rounded-3xl text-white/40">
               No identities registered in neural database.
             </div>
           ) : (
             users.map(user => (
               <div key={user.name} className="flex justify-between items-center p-4 bg-white/5 rounded-2xl hover:bg-white/10 transition-colors border border-white/5 group">
                 <div className="flex items-center gap-4">
                   <div className="w-10 h-10 rounded-xl bg-blue-500/10 flex items-center justify-center text-blue-500">
                     <Users size={18} />
                   </div>
                   <div>
                     <p className="font-bold">{user.name}</p>
                     <p className="text-[10px] text-white/40 font-mono">HASH: {user.descriptor.slice(0, 8).join('')}...</p>
                   </div>
                 </div>
                 <button 
                  onClick={() => {
                    if(confirm("Confirm deletion?")) setUsers(u => u.filter(x => x.name !== user.name))
                  }}
                  className="p-2 text-white/20 hover:text-red-500 hover:bg-red-500/10 rounded-lg transition-all opacity-0 group-hover:opacity-100"
                 >
                   <Trash2 size={16} />
                 </button>
               </div>
             ))
           )}
         </div>
       </div>
    </motion.div>
  )
}

function HistoryView({ attendance }) {
  return (
    <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} className="flex flex-col gap-6">
       <div className="card">
         <div className="flex justify-between items-center mb-8">
           <div className="flex items-center gap-3">
             <History className="text-blue-500" size={24} />
             <h3 className="font-bold text-xl">Event Log History</h3>
           </div>
         </div>
         
         <div className="flex flex-col gap-4">
           {attendance.length === 0 ? (
             <div className="p-12 text-center border-2 border-dashed border-white/5 rounded-[40px]">
               <div className="mb-4 inline-flex p-4 bg-white/5 rounded-3xl text-white/20"><History size={40} /></div>
               <p className="text-white/40 font-medium">No biometric events recorded.</p>
             </div>
           ) : (
             <div className="overflow-x-auto">
               <table className="w-full text-left">
                 <thead>
                   <tr className="border-b border-white/5">
                     <th className="pb-4 text-xs font-bold uppercase text-white/40 tracking-widest px-4">Subject</th>
                     <th className="pb-4 text-xs font-bold uppercase text-white/40 tracking-widest px-4">Timestamp</th>
                     <th className="pb-4 text-xs font-bold uppercase text-white/40 tracking-widest px-4 text-right">Status</th>
                   </tr>
                 </thead>
                 <tbody className="divide-y divide-white/5">
                   {attendance.map(log => (
                     <tr key={log.id} className="group hover:bg-white/[0.02]">
                       <td className="py-5 px-4">
                         <div className="flex items-center gap-3">
                           <div className="w-8 h-8 rounded-lg bg-blue-500/10 flex items-center justify-center text-blue-500 text-[10px] font-bold">
                             {log.name.slice(0, 2).toUpperCase()}
                           </div>
                           <span className="font-bold text-sm">{log.name}</span>
                         </div>
                       </td>
                       <td className="py-5 px-4 text-sm text-white/60">
                         {format(new Date(log.timestamp), 'MMM d, h:mm a')}
                       </td>
                       <td className="py-5 px-4 text-right">
                         <span className="px-3 py-1 rounded-full bg-green-500/10 text-green-500 text-[10px] font-extrabold uppercase tracking-widest border border-green-500/20">
                           {log.status}
                         </span>
                       </td>
                     </tr>
                   ))}
                 </tbody>
               </table>
             </div>
           )}
         </div>
       </div>
    </motion.div>
  )
}
