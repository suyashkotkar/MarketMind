import { useState } from 'react'
import { token } from '../lib/theme'

export function Card({ title, subtitle, toolbar, children, ...rest }) {
  return (
    <section className="card" {...rest}>
      {(title || toolbar) && (
        <header>
          {title && <h2>{title}</h2>}
          {toolbar && <div className="toolbar">{toolbar}</div>}
        </header>
      )}
      {subtitle && <p className="sub">{subtitle}</p>}
      {children}
    </section>
  )
}

export function Tile({ label, value, meta, tone }) {
  return (
    <div className="tile">
      <div className="label">{label}</div>
      <div className={`value ${tone === 'sm' ? 'sm' : ''}`}>{value}</div>
      {meta && <div className="meta">{meta}</div>}
    </div>
  )
}

export function Delta({ value, digits = 2 }) {
  if (value === null || value === undefined || Number.isNaN(value)) return <span className="muted">—</span>
  const cls = value > 0 ? 'delta-up' : value < 0 ? 'delta-down' : 'muted'
  const arrow = value > 0 ? '▲' : value < 0 ? '▼' : '■'
  return (
    <span className={cls}>
      {arrow} {(value * 100).toFixed(digits)}%
    </span>
  )
}

/** Status is never carried by color alone — always icon + label. */
export function StatusBadge({ level, label }) {
  const map = {
    good: ['--status-good', '✔'], warning: ['--status-warning', '▲'],
    serious: ['--status-serious', '◆'], critical: ['--status-critical', '■'],
    info: ['--text-muted', '•'],
  }
  const [varName, glyph] = map[level] || map.info
  return (
    <span className="badge">
      <span aria-hidden="true" style={{ color: token(varName) }}>{glyph}</span>
      {label}
    </span>
  )
}

export function Spinner({ label = 'Loading…' }) {
  return <p className="loading">{label}</p>
}

export function ErrorBox({ error, onRetry }) {
  return (
    <div className="err">
      <strong>Couldn’t load that.</strong>{' '}
      <span className="muted">{String(error?.message || error)}</span>
      {onRetry && (
        <>
          {' '}
          <button className="linkish" onClick={onRetry}>Try again</button>
        </>
      )}
    </div>
  )
}

export function DataTable({ columns, rows, caption }) {
  return (
    <div className="scroll-x">
      <table className="data">
        {caption && <caption className="sub" style={{ textAlign: 'left' }}>{caption}</caption>}
        <thead>
          <tr>
            {columns.map((c) => (
              <th key={c.key} scope="col" className={c.wrap ? 'wrap' : undefined}>{c.header}</th>
            ))}
          </tr>
        </thead>
        <tbody>
          {rows.map((r, i) => (
            <tr key={r.__key ?? i}>
              {columns.map((c) => (
                <td key={c.key} className={c.wrap ? 'wrap' : undefined}>
                  {c.render ? c.render(r) : r[c.key] ?? '—'}
                </td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  )
}

/**
 * Every chart ships with a table twin: the WCAG-clean way to read the same
 * values, and the required relief for the light-mode contrast warning.
 */
export function ChartWithTable({ chart, table, tableLabel = 'table' }) {
  const [showTable, setShowTable] = useState(false)
  return (
    <>
      <div style={{ display: 'flex', justifyContent: 'flex-end', marginBottom: 4 }}>
        <button className="linkish" onClick={() => setShowTable((s) => !s)}
                aria-pressed={showTable}>
          {showTable ? 'Show chart' : `Show ${tableLabel}`}
        </button>
      </div>
      {showTable ? table : chart}
    </>
  )
}
