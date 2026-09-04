const BASE = import.meta.env.VITE_API_BASE || '/api/v1'

async function req(path, options = {}) {
  const res = await fetch(`${BASE}${path}`, {
    headers: { 'Content-Type': 'application/json' },
    ...options,
  })
  if (!res.ok) {
    let detail = res.statusText
    try {
      detail = (await res.json()).detail ?? detail
    } catch { /* body was not JSON */ }
    throw new Error(detail)
  }
  return res.json()
}

export const api = {
  health: () => req('/health'),
  config: () => req('/config'),

  tickers: () => req('/stocks'),
  overview: (limit = 25) => req(`/stocks/overview?limit=${limit}`),
  ticker: (s) => req(`/stocks/${s}`),
  history: (s, days = 365) => req(`/stocks/${s}/history?days=${days}`),
  stats: (s) => req(`/stocks/${s}/stats`),

  prediction: (s) => req(`/predictions/${s}`),
  modelInfo: () => req('/predictions/model'),
  trackRecord: (s) => req(`/predictions/track-record${s ? `?symbol=${s}` : ''}`),

  risk: (s) => req(`/risk/${s}`),
  riskRanking: (limit = 50) => req(`/risk/ranking?limit=${limit}`),

  anomalies: (s, lookback = 180) => req(`/anomalies/${s}?lookback_days=${lookback}`),
  recentAnomalies: (days = 30, min = 0.3) =>
    req(`/anomalies/recent?days=${days}&min_severity=${min}`),

  sentiment: (s, limit = 50) => req(`/sentiment/${s}?limit=${limit}`),
  compare: (symbols) => req(`/compare?symbols=${symbols.join(',')}`),
  alerts: (days = 7, minConfidence = 0.2) =>
    req(`/alerts?days=${days}&min_confidence=${minConfidence}`),

  ingest: (body) => req('/admin/ingest', { method: 'POST', body: JSON.stringify(body) }),
  train: (body) => req('/admin/train', { method: 'POST', body: JSON.stringify(body) }),
  job: (name) => req(`/admin/jobs/${name}`),
}

export const fmtPct = (v, digits = 2) =>
  v === null || v === undefined || Number.isNaN(v) ? '—' : `${(v * 100).toFixed(digits)}%`

export const fmtNum = (v, digits = 2) =>
  v === null || v === undefined || Number.isNaN(v) ? '—' : Number(v).toFixed(digits)

export const fmtCompact = (v) => {
  if (v === null || v === undefined || Number.isNaN(v)) return '—'
  const abs = Math.abs(v)
  if (abs >= 1e12) return `${(v / 1e12).toFixed(2)}T`
  if (abs >= 1e9) return `${(v / 1e9).toFixed(2)}B`
  if (abs >= 1e6) return `${(v / 1e6).toFixed(2)}M`
  if (abs >= 1e3) return `${(v / 1e3).toFixed(1)}K`
  return v.toFixed(0)
}
