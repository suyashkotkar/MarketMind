import { useEffect, useState } from 'react'

/** Read a design token off :root so charts never hardcode hex. */
export function token(name) {
  if (typeof window === 'undefined') return '#000'
  return getComputedStyle(document.documentElement).getPropertyValue(name).trim()
}

export const SERIES_SLOTS = [
  '--series-1', '--series-2', '--series-3', '--series-4',
  '--series-5', '--series-6', '--series-7', '--series-8',
]

/**
 * Color follows the *entity*, not its rank: once a symbol has been given a
 * slot it keeps it, so removing a series never repaints the survivors.
 * Past 8 entities we stop assigning rather than inventing a 9th hue.
 */
const assigned = new Map()
export function colorFor(key) {
  if (!assigned.has(key)) {
    if (assigned.size >= SERIES_SLOTS.length) return token('--text-muted')
    assigned.set(key, SERIES_SLOTS[assigned.size])
  }
  return token(assigned.get(key))
}
export function slotIndexFor(key) {
  return assigned.has(key) ? SERIES_SLOTS.indexOf(assigned.get(key)) : -1
}

export function useTheme() {
  const [theme, setTheme] = useState(
    () => localStorage.getItem('stockseer-theme') || 'system',
  )
  useEffect(() => {
    const root = document.documentElement
    if (theme === 'system') root.removeAttribute('data-theme')
    else root.setAttribute('data-theme', theme)
    try { localStorage.setItem('stockseer-theme', theme) } catch { /* private mode */ }
  }, [theme])
  return [theme, setTheme]
}

/** Re-render charts when the resolved theme changes. */
export function useThemeEpoch() {
  const [epoch, setEpoch] = useState(0)
  useEffect(() => {
    const bump = () => setEpoch((e) => e + 1)
    const mq = window.matchMedia('(prefers-color-scheme: dark)')
    mq.addEventListener('change', bump)
    const obs = new MutationObserver(bump)
    obs.observe(document.documentElement, { attributes: true, attributeFilter: ['data-theme'] })
    return () => { mq.removeEventListener('change', bump); obs.disconnect() }
  }, [])
  return epoch
}

/** Shared Plotly layout: recessive hairline grid, no chart junk. */
export function baseLayout(overrides = {}) {
  const ink = token('--text-primary')
  const muted = token('--text-muted')
  const grid = token('--grid')
  const axis = token('--axis')
  const surface = token('--surface-1')

  const ax = {
    gridcolor: grid, griddash: 'solid', gridwidth: 1,
    zeroline: false, linecolor: axis, linewidth: 1,
    tickfont: { color: muted, size: 11 },
    titlefont: { color: muted, size: 11 },
    automargin: true,
  }

  return {
    paper_bgcolor: surface,
    plot_bgcolor: surface,
    font: { family: "system-ui, -apple-system, 'Segoe UI', sans-serif", color: ink, size: 12 },
    margin: { l: 8, r: 12, t: 8, b: 8 },
    hovermode: 'x unified',
    hoverlabel: {
      bgcolor: surface, bordercolor: axis,
      font: { color: ink, size: 12, family: "system-ui, -apple-system, sans-serif" },
    },
    showlegend: false,
    legend: {
      orientation: 'h', y: -0.18, x: 0, font: { color: muted, size: 11 },
      bgcolor: 'rgba(0,0,0,0)',
    },
    xaxis: { ...ax, ...(overrides.xaxis || {}) },
    yaxis: { ...ax, ...(overrides.yaxis || {}) },
    ...Object.fromEntries(Object.entries(overrides).filter(([k]) => k !== 'xaxis' && k !== 'yaxis')),
  }
}

export const PLOT_CONFIG = {
  displaylogo: false,
  responsive: true,
  modeBarButtonsToRemove: ['select2d', 'lasso2d', 'autoScale2d', 'toggleSpikelines'],
}

/** Blue→dark sequential ramp (magnitude) as a Plotly colorscale. */
export function sequentialScale() {
  return [
    [0, token('--seq-100')], [0.25, token('--seq-250')], [0.5, token('--seq-400')],
    [0.75, token('--seq-550')], [1, token('--seq-700')],
  ]
}

/** Diverging blue↔red with a neutral gray midpoint (polarity). */
export function divergingScale() {
  return [
    [0, token('--div-neg')], [0.5, token('--div-mid')], [1, token('--div-pos')],
  ]
}

export const RISK_GRADE_STATUS = {
  A: '--status-good', B: '--status-good', C: '--status-warning',
  D: '--status-serious', E: '--status-critical', F: '--status-critical',
}
