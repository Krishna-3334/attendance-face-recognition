// Client for the BioAccess FastAPI backend.
//
// The browser no longer runs any face model: it captures frames and posts them.
// Detection, ArcFace embedding, liveness and matching all happen server-side,
// which is why the gallery can live in a database and the matching threshold can
// be an evaluated operating point rather than a hardcoded default.

const BASE = import.meta.env.VITE_API_BASE ?? 'http://127.0.0.1:8000'

async function request(path, options = {}) {
  const res = await fetch(`${BASE}${path}`, options)
  if (!res.ok) {
    let detail = res.statusText
    try {
      const body = await res.json()
      detail = body.detail ?? detail
    } catch {
      /* non-JSON error body */
    }
    throw new Error(`${res.status}: ${detail}`)
  }
  return res.json()
}

/** Grab the current video frame as a JPEG blob. */
export function captureFrame(video, quality = 0.92) {
  const canvas = document.createElement('canvas')
  canvas.width = video.videoWidth
  canvas.height = video.videoHeight
  canvas.getContext('2d').drawImage(video, 0, 0, canvas.width, canvas.height)
  return new Promise((resolve, reject) => {
    canvas.toBlob(
      (blob) => (blob ? resolve(blob) : reject(new Error('frame capture failed'))),
      'image/jpeg',
      quality
    )
  })
}

export const api = {
  health: () => request('/api/health'),
  config: () => request('/api/config'),
  stats: () => request('/api/stats'),

  users: () => request('/api/users'),
  deleteUser: (id) => request(`/api/users/${id}`, { method: 'DELETE' }),

  /** Enrol one identity from several captured frames. */
  enrol: (name, blobs) => {
    const form = new FormData()
    form.append('name', name)
    blobs.forEach((blob, i) => form.append('files', blob, `capture_${i}.jpg`))
    return request('/api/enrol', { method: 'POST', body: form })
  },

  /** Liveness-gated recognition of a single frame. */
  recognise: (blob, { logAttendance = true } = {}) => {
    const form = new FormData()
    form.append('file', blob, 'probe.jpg')
    form.append('log_attendance', String(logAttendance))
    return request('/api/recognise', { method: 'POST', body: form })
  },

  attendance: (limit = 200) => request(`/api/attendance?limit=${limit}`),
  spoofEvents: (limit = 100) => request(`/api/spoof-events?limit=${limit}`),
}

export default api
