const API_BASE = import.meta.env.VITE_API_BASE || ''
const WS_BASE = import.meta.env.VITE_WS_BASE || (typeof location !== 'undefined' ? `${location.protocol === 'https:' ? 'wss:' : 'ws:'}//${location.host}` : 'ws://127.0.0.1:9000')

export { API_BASE, WS_BASE }
