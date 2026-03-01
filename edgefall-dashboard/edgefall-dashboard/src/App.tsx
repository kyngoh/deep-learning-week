import { useEffect, useState } from 'react'

type Severity = 'low' | 'medium' | 'high'
type Status = 'pending' | 'sent' | 'acknowledged'

type Incident = {
  id: string
  timestamp: string
  location: string
  severity: Severity
  confidence: number
  status: Status
  camera: string
}

const BACKEND_BASE = 'http://localhost:8000'

const SEVERITY_COLOR: Record<Severity, string> = {
  low: 'text-gray-600 border-gray-300',
  medium: 'text-yellow-700 border-yellow-400',
  high: 'text-red-600 border-red-500',
}

const STATUS_LABEL: Record<Status, string> = {
  pending: 'pending',
  sent: 'sent',
  acknowledged: 'acknowledged',
}

const ZONE_OPTIONS = [
  'Yishun Ring Rd',
  'Blk 456 Yishun St 42 #08-567',
  'Blk 234 Yishun North Ave 4 #03-890',
  'Blk 789  Yishun Ave 3 #12-234',
  'Yishun Community Clinic',
  'Yishun Campus Walkway A',
  'Yishun Senior Activity Room',
]

function App() {
  const [zone, setZone] = useState('Yishun Ring Rd')
  const [incidents, setIncidents] = useState<Incident[]>([])
  const [activeIncident, setActiveIncident] = useState<Incident | null>(null)
  const [fps] = useState(17.8)
  const [videoError, setVideoError] = useState(false)

  useEffect(() => {
    const sync = async () => {
      try {
        await fetch(`${BACKEND_BASE}/zone`, {
          method: 'PUT',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ zone }),
        })
      } catch {
        // backend may be offline
      }
    }
    sync()
  }, [zone])

  useEffect(() => {
    const load = async () => {
      try {
        const res = await fetch(`${BACKEND_BASE}/incidents`)
        if (!res.ok) return
        const data = (await res.json()) as Incident[]
        setIncidents(data)
        setActiveIncident(data[0] ?? null)
      } catch (err) {
        console.error('Failed to load incidents', err)
      }
    }

    load()
    const id = setInterval(load, 3000)
    return () => clearInterval(id)
  }, [])

  return (
    <div className="min-h-screen flex flex-col">
      {/* Top app bar */}
      <header className="w-full border-b bg-white">
        <div className="max-w-6xl mx-auto flex items-center justify-between py-3 px-4">
          <div className="flex items-center gap-2">
            <div className="w-8 h-8 rounded-full bg-blue-600 flex items-center justify-center text-white font-semibold text-sm">
              E
            </div>
            <div>
              <div className="text-base font-semibold">EdgeFall SG</div>
              <div className="text-xs text-gray-500">
                Community Safety Command Center
              </div>
            </div>
          </div>

          <div className="flex items-center gap-12">
            <div className="text-xs text-gray-500">
              Dashboard for EdgeFall SG
            </div>
            <div className="flex items-center gap-2">
              <label htmlFor="zone-select" className="text-xs text-gray-600">
                Zone:
              </label>
              <select
                id="zone-select"
                value={zone}
                onChange={(e) => setZone(e.target.value)}
                className="text-xs font-semibold text-blue-700 bg-white border border-gray-300 rounded px-2 py-1.5 focus:outline-none focus:ring-2 focus:ring-blue-400"
              >
                {ZONE_OPTIONS.map((z) => (
                  <option key={z} value={z}>
                    {z}
                  </option>
                ))}
              </select>
            </div>
          </div>
        </div>
      </header>

      <main className="flex-1">
        <div className="max-w-6xl mx-auto px-4 py-4 space-y-5">
          {/* Live CCTV + Active Incident card */}
          <section className="bg-white rounded-xl border border-edge-border shadow-sm px-5 pt-4 pb-5">
            <div className="flex justify-between items-center mb-3">
              <h2 className="text-lg font-semibold">Live CCTV Feed</h2>
              <div className="flex items-center gap-4 text-xs text-gray-500">
                <div className="flex items-center gap-2">
                  <span>Low-Res Mode</span>
                  <span className="w-9 h-5 flex items-center bg-gray-200 rounded-full px-0.5">
                    <span className="w-4 h-4 bg-white rounded-full shadow" />
                  </span>
                </div>
                <span>{fps.toFixed(1)} FPS</span>
              </div>
            </div>

            <div className="grid grid-cols-1 lg:grid-cols-[2fr,1fr] gap-4">
              {/* Live feed */}
              <div>
                <div className="bg-black rounded-xl overflow-hidden relative min-h-[360px]">
                  <img
                    src={`${BACKEND_BASE}/video`}
                    alt="Live CCTV"
                    className="w-full h-[360px] object-cover"
                    onLoad={() => setVideoError(false)}
                    onError={() => setVideoError(true)}
                  />
                  {videoError && (
                    <div className="absolute inset-0 flex flex-col items-center justify-center text-white text-sm bg-black/90">
                      <span className="font-semibold">Camera feed unavailable</span>
                      <span className="text-xs text-gray-400 mt-1">
                        Start edgefall_api.py to enable live webcam + fall detection
                      </span>
                    </div>
                  )}

                  {activeIncident && !videoError && (
                    <div className="absolute top-4 right-4 bg-red-500 text-white text-xs font-semibold px-4 py-1.5 rounded-full shadow">
                      FALL DETECTED
                    </div>
                  )}
                </div>

                {/* Legend */}
                <div className="flex flex-wrap gap-4 text-xs text-gray-600 mt-3">
                  <span className="flex items-center gap-1">
                    <span className="w-2.5 h-2.5 rounded-full bg-blue-600" />
                    YOLOv8 Pose Detection
                  </span>
                  <span className="flex items-center gap-1">
                    <span className="w-2.5 h-2.5 rounded-full bg-green-500" />
                    Privacy Mode: Redacted
                  </span>
                  <span className="flex items-center gap-1">
                    <span className="w-2.5 h-2.5 rounded-full bg-orange-500" />
                    Edge Processing
                  </span>
                </div>
              </div>

              {/* Active Incident */}
              <aside className="border border-red-200 rounded-xl bg-edge-redSoft px-4 py-4 flex flex-col gap-3">
                <div className="flex items-center justify-between">
                  <div className="flex items-center gap-2">
                    <span className="text-red-500 text-lg">⚠</span>
                    <h3 className="text-sm font-semibold">Active Incident</h3>
                  </div>
                </div>

                {activeIncident ? (
                  <>
                    <div className="text-xs space-y-1 text-gray-700">
                      <div>
                        <span className="font-semibold">Timestamp:</span>{' '}
                        {activeIncident.timestamp}
                      </div>
                      <div>
                        <span className="font-semibold">Location:</span>{' '}
                        {activeIncident.location}
                      </div>
                      <div className="flex justify-between pr-4">
                        <div>
                          <span className="font-semibold">Confidence:</span>{' '}
                          {activeIncident.confidence}%
                        </div>
                        <div>
                          <span className="font-semibold">Severity:</span>{' '}
                          <span
                            className={`font-medium ${
                              activeIncident.severity === 'high'
                                ? 'text-red-600'
                                : activeIncident.severity === 'medium'
                                  ? 'text-yellow-700'
                                  : 'text-gray-600'
                            }`}
                          >
                            {activeIncident.severity}
                          </span>
                        </div>
                      </div>
                    </div>

                    <div className="rounded-lg bg-gray-900 text-gray-100 h-32 flex flex-col items-center justify-center text-xs">
                      [Redacted Snapshot]
                      <span className="text-[11px] text-gray-400 mt-1">
                        Privacy-preserving mode enabled
                      </span>
                    </div>

                    <div className="space-y-2 mt-1">
                      <div className="text-xs">
                        <span className="font-semibold">Status:</span>{' '}
                        <span className="px-2 py-0.5 rounded-full bg-blue-50 text-blue-700 text-[11px]">
                          {STATUS_LABEL[activeIncident.status]}
                        </span>
                      </div>
                      <button
                        type="button"
                        className="w-full rounded-full py-2 text-sm font-medium bg-gray-900 text-white"
                      >
                        Acknowledge
                      </button>
                      <button
                        type="button"
                        className="w-full rounded-full py-2 text-sm font-medium bg-red-500 text-white"
                      >
                        Escalate to SCDF
                      </button>
                      <button
                        type="button"
                        className="w-full rounded-full py-2 text-sm font-medium border border-gray-300 text-gray-800 bg-white"
                      >
                        False Positive
                      </button>
                    </div>
                  </>
                ) : (
                  <div className="text-xs text-gray-500">
                    No active incidents.
                  </div>
                )}
              </aside>
            </div>
          </section>

          {/* Incident Log */}
          <section className="bg-white rounded-xl border border-edge-border shadow-sm px-5 pt-4 pb-5">
            <h2 className="text-lg font-semibold mb-3">Incident Log</h2>
            <div className="overflow-x-auto">
              <table className="min-w-full text-xs">
                <thead className="border-b bg-gray-50 text-gray-500">
                  <tr className="text-left">
                    <th className="py-2 pr-4 font-medium">Timestamp</th>
                    <th className="py-2 pr-4 font-medium">Location</th>
                    <th className="py-2 pr-4 font-medium">Severity</th>
                    <th className="py-2 pr-4 font-medium">Confidence</th>
                    <th className="py-2 pr-4 font-medium">Status</th>
                    <th className="py-2 pr-4 font-medium">Transmitted</th>
                    <th className="py-2 pr-2 font-medium">Camera</th>
                  </tr>
                </thead>
                <tbody>
                  {incidents.map((inc) => (
                    <tr key={inc.id} className="border-b last:border-0">
                      <td className="py-2 pr-4 text-gray-800">
                        {inc.timestamp}
                      </td>
                      <td className="py-2 pr-4 text-gray-800">
                        {inc.location}
                      </td>
                      <td className="py-2 pr-4">
                        <span
                          className={`px-2 py-0.5 rounded-full border text-[11px] ${SEVERITY_COLOR[inc.severity]}`}
                        >
                          {inc.severity}
                        </span>
                      </td>
                      <td className="py-2 pr-4 text-gray-800">
                        {inc.confidence}%
                      </td>
                      <td className="py-2 pr-4">
                        <span className="px-2 py-0.5 rounded-full bg-gray-100 text-gray-700 text-[11px]">
                          {STATUS_LABEL[inc.status]}
                        </span>
                      </td>
                      <td className="py-2 pr-4">
                        <span className="text-green-500 text-base">●</span>
                      </td>
                      <td className="py-2 pr-2 text-gray-800">
                        {inc.camera}
                      </td>
                    </tr>
                  ))}
                  {incidents.length === 0 && (
                    <tr>
                      <td
                        colSpan={7}
                        className="py-4 text-center text-gray-400"
                      >
                        No incidents logged yet.
                      </td>
                    </tr>
                  )}
                </tbody>
              </table>
            </div>
          </section>

          <p className="text-[11px] text-gray-500 text-center pb-4">
            Privacy: All video is processed locally on the edge device. No
            frames are sent to the cloud.
          </p>
        </div>
      </main>
    </div>
  )
}

export default App
