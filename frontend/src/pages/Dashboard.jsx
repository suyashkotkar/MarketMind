import { api } from '../lib/api'
import { useAsync } from '../lib/useAsync'
import { AlertsPanel, OverviewTable } from '../components/panels'
import { Card, ErrorBox, Spinner } from '../components/ui'

export default function Dashboard({ onSelect, health }) {
  const overview = useAsync(() => api.overview(50), [])
  const alerts = useAsync(() => api.alerts(30, 0.1), [])
  const ranking = useAsync(() => api.riskRanking(50), [])

  if (overview.loading && !overview.data) return <Spinner label="Loading the universe…" />
  if (overview.error && !overview.data) return <ErrorBox error={overview.error} onRetry={overview.refresh} />

  const cards = overview.data?.cards ?? []
  const safest = ranking.data?.slice(0, 5) ?? []
  const riskiest = [...(ranking.data ?? [])].reverse().slice(0, 5)

  return (
    <div className="grid" style={{ gap: 16 }}>
      <Card
        title="Universe"
        subtitle={
          health
            ? `${health.tickers_loaded} tickers · prices through ${health.latest_price_date ?? '—'} · source “${health.data_source}”`
            : undefined
        }
      >
        <div className={overview.loading ? 'stale' : ''}>
          <OverviewTable cards={cards} onSelect={onSelect} />
        </div>
      </Card>

      <div className="grid cols-2">
        <Card title="Lowest risk" subtitle="Ranked by the composite score">
          {ranking.loading && !ranking.data ? <Spinner /> : (
            <ol className="list">
              {safest.map((r) => (
                <li key={r.symbol}>
                  <button className="linkish" onClick={() => onSelect(r.symbol)}>{r.symbol}</button>
                  <span className="muted" style={{ flex: 1 }}>
                    grade {r.grade} · vol {(r.annualized_vol * 100).toFixed(0)}%
                  </span>
                  <strong>{r.risk_score.toFixed(0)}</strong>
                </li>
              ))}
            </ol>
          )}
        </Card>

        <Card title="Highest risk" subtitle="Same score, other end">
          {ranking.loading && !ranking.data ? <Spinner /> : (
            <ol className="list">
              {riskiest.map((r) => (
                <li key={r.symbol}>
                  <button className="linkish" onClick={() => onSelect(r.symbol)}>{r.symbol}</button>
                  <span className="muted" style={{ flex: 1 }}>
                    grade {r.grade} · drawdown {(r.max_drawdown * 100).toFixed(0)}%
                  </span>
                  <strong>{r.risk_score.toFixed(0)}</strong>
                </li>
              ))}
            </ol>
          )}
        </Card>
      </div>

      {alerts.error
        ? <ErrorBox error={alerts.error} onRetry={alerts.refresh} />
        : <AlertsPanel alerts={alerts.data ?? []} onSelect={onSelect} />}
    </div>
  )
}
