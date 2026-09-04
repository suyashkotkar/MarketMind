import { useEffect, useMemo, useState } from 'react'
import { api } from './lib/api'
import { useAsync } from './lib/useAsync'
import { colorFor, useTheme } from './lib/theme'
import { Card, ErrorBox, Spinner, StatusBadge } from './components/ui'
import Dashboard from './pages/Dashboard'
import StockDetail from './pages/StockDetail'
import Compare from './pages/Compare'
import ModelPage from './pages/ModelPage'

const RANGES = [
  { label: '3M', days: 90 },
  { label: '6M', days: 180 },
  { label: '1Y', days: 365 },
  { label: '5Y', days: 1825 },
]

export default function App() {
  const [page, setPage] = useState('dashboard')
  const [symbol, setSymbol] = useState(null)
  const [days, setDays] = useState(365)
  const [theme, setTheme] = useTheme()

  const health = useAsync(() => api.health(), [])
  const config = useAsync(() => api.config(), [])
  const tickers = useAsync(() => api.tickers(), [])

  const universe = useMemo(() => {
    const bench = config.data?.benchmark
    return (tickers.data ?? []).map((t) => t.symbol).filter((s) => s !== bench)
  }, [tickers.data, config.data])

  // Claim a colour slot per symbol once, in a stable order, so identity is
  // fixed for the whole session.
  useEffect(() => { universe.slice(0, 8).forEach(colorFor) }, [universe])
  useEffect(() => { if (!symbol && universe.length) setSymbol(universe[0]) }, [symbol, universe])

  const open = (s) => { setSymbol(s); setPage('stock') }

  const empty = health.data && health.data.tickers_loaded === 0

  return (
    <div className="app">
      <header className="topbar">
        <div className="brand">
          <h1>StockSeer</h1>
          <span>prediction · risk · anomalies</span>
        </div>

        <nav className="nav" aria-label="Sections">
          {[['dashboard', 'Dashboard'], ['stock', 'Stock'], ['compare', 'Compare'], ['model', 'Model']]
            .map(([id, label]) => (
              <button key={id} onClick={() => setPage(id)}
                      aria-current={page === id ? 'page' : undefined}>{label}</button>
            ))}
        </nav>

        <div className="spacer" />

        {health.data && (
          <StatusBadge
            level={health.data.status === 'ok' ? 'good' : 'warning'}
            label={health.data.model_version ? 'model ready' : 'no model'}
          />
        )}

        <div className="seg" role="group" aria-label="Colour theme">
          {['light', 'system', 'dark'].map((t) => (
            <button key={t} aria-pressed={theme === t} onClick={() => setTheme(t)}>{t}</button>
          ))}
        </div>
      </header>

      {/* One filter row above everything it scopes. */}
      {(page === 'stock' || page === 'dashboard') && (
        <div className="filterbar">
          {page === 'stock' && (
            <div className="field">
              <label htmlFor="sym">Symbol</label>
              <select id="sym" value={symbol ?? ''} onChange={(e) => setSymbol(e.target.value)}>
                {universe.map((s) => <option key={s} value={s}>{s}</option>)}
              </select>
            </div>
          )}
          {page === 'stock' && (
            <div className="field">
              <label>Range</label>
              <div className="seg" role="group" aria-label="Date range">
                {RANGES.map((r) => (
                  <button key={r.label} aria-pressed={days === r.days}
                          onClick={() => setDays(r.days)}>{r.label}</button>
                ))}
              </div>
            </div>
          )}
          <div className="spacer" />
          <span className="muted" style={{ fontSize: 12 }}>
            {health.data?.latest_price_date ? `data through ${health.data.latest_price_date}` : ''}
          </span>
        </div>
      )}

      {health.error && <ErrorBox error={health.error} onRetry={health.refresh} />}

      {empty ? (
        <Card title="Nothing ingested yet"
              subtitle="The warehouse is empty — run the ETL before the dashboard has anything to show.">
          <pre style={{
            background: 'var(--page)', padding: 12, borderRadius: 8, overflowX: 'auto',
          }}>{`python -m stockseer.cli pipeline`}</pre>
          <p className="footnote">
            That ingests the default universe, trains the direction model and writes
            the first predictions. Reload this page when it finishes.
          </p>
        </Card>
      ) : (
        <main>
          {page === 'dashboard' && <Dashboard onSelect={open} health={health.data} />}
          {page === 'stock' && (symbol
            ? <StockDetail symbol={symbol} days={days} config={config.data} />
            : <Spinner />)}
          {page === 'compare' && (universe.length
            ? <Compare universe={universe} initial={symbol ? [symbol] : []} />
            : <Spinner />)}
          {page === 'model' && <ModelPage />}
        </main>
      )}

      <p className="footnote" style={{ marginTop: 28 }}>
        StockSeer produces statistical estimates from historical data. It is not
        investment advice, and past patterns do not bind the future.
      </p>
    </div>
  )
}
