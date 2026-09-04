import { fmtCompact, fmtNum, fmtPct } from '../lib/api'
import { RISK_GRADE_STATUS, token } from '../lib/theme'
import {
  RiskComponentChart, SentimentTimeline,
} from './charts'
import { Card, DataTable, Delta, StatusBadge, Tile } from './ui'

const DIRECTION_TONE = { UP: 'good', DOWN: 'critical', NEUTRAL: 'info' }
const GRADE_TONE = { A: 'good', B: 'good', C: 'warning', D: 'serious', E: 'critical', F: 'critical' }

/* ------------------------------------------------------------------ */
/* Prediction: one number, given room. A gauge would be chart junk.     */
/* ------------------------------------------------------------------ */
export function PredictionPanel({ prediction, config }) {
  const p = prediction
  const pct = p.prob_up * 100
  const auc = p.model_metrics?.roc_auc
  const longThr = (config?.long_threshold ?? 0.55) * 100
  const shortThr = (config?.short_threshold ?? 0.45) * 100

  return (
    <Card
      title={`Direction over the next ${p.horizon_days} sessions`}
      subtitle={`Model ${p.model_version} · as of ${p.as_of}`}
      toolbar={<StatusBadge level={DIRECTION_TONE[p.direction]} label={p.direction} />}
    >
      <Tile
        label="Probability the price ends higher"
        value={`${pct.toFixed(1)}%`}
        meta={`Confidence ${(p.confidence * 100).toFixed(0)}% · expected move ${
          p.expected_move_pct === null || p.expected_move_pct === undefined
            ? '—' : `${p.expected_move_pct >= 0 ? '+' : ''}${p.expected_move_pct.toFixed(2)}%`}`}
      />

      <div className="probbar">
        <div className="wrap">
          <div className="track" />
          <div className="thresh" style={{ left: `${shortThr}%` }} aria-hidden="true" />
          <div className="thresh" style={{ left: '50%' }} aria-hidden="true" />
          <div className="thresh" style={{ left: `${longThr}%` }} aria-hidden="true" />
          <div className="marker" style={{ left: `calc(${pct}% - 2px)` }}
               role="img" aria-label={`Probability ${pct.toFixed(1)} percent`} />
        </div>
        <div className="scale">
          <span>0% (down)</span>
          <span>coin flip 50%</span>
          <span>100% (up)</span>
        </div>
      </div>

      <p className="footnote">
        {auc !== undefined && Number.isFinite(auc)
          ? `Out-of-fold ROC AUC ${auc.toFixed(3)} on ${p.model_metrics.n?.toLocaleString()} walk-forward test rows — `
          : ''}
        {p.disclaimer}
      </p>
    </Card>
  )
}

/* ------------------------------------------------------------------ */
export function RiskPanel({ risk }) {
  return (
    <Card
      title="Risk assessment"
      subtitle={`as of ${risk.as_of} · 0 = calm, 100 = treacherous`}
      toolbar={<StatusBadge level={GRADE_TONE[risk.grade]} label={`Grade ${risk.grade}`} />}
    >
      <div className="grid cols-4" style={{ marginBottom: 14 }}>
        <Tile label="Risk score" value={risk.risk_score.toFixed(0)} meta="out of 100" />
        <Tile label="Annualised volatility" value={fmtPct(risk.annualized_vol, 0)} tone="sm" />
        <Tile label="Max drawdown" value={fmtPct(-risk.max_drawdown, 0)} tone="sm" />
        <Tile label="95% daily VaR" value={fmtPct(-risk.var_95, 1)} tone="sm"
              meta={`CVaR ${fmtPct(-risk.cvar_95, 1)}`} />
      </div>

      <RiskComponentChart components={risk.components} />

      <div className="grid cols-4" style={{ marginTop: 14 }}>
        <Tile label="Sharpe" value={fmtNum(risk.sharpe)} tone="sm" />
        <Tile label="Sortino" value={fmtNum(risk.sortino)} tone="sm" />
        <Tile label="Beta vs benchmark" value={fmtNum(risk.beta)} tone="sm" />
        <Tile label="Median $ volume" value={fmtCompact(risk.median_dollar_volume)} tone="sm" />
      </div>

      <p className="footnote">{risk.narrative}</p>
    </Card>
  )
}

/* ------------------------------------------------------------------ */
export function SentimentPanel({ sentiment }) {
  const s = sentiment
  const tone = s.label === 'positive' ? 'good' : s.label === 'negative' ? 'critical' : 'info'
  return (
    <Card
      title="News sentiment"
      subtitle={`${s.articles.length} recent headlines scored with a finance lexicon`}
      toolbar={<StatusBadge level={tone} label={s.label ?? 'no signal'} />}
    >
      <Tile label="Average score" value={s.mean_sentiment === null ? '—' : s.mean_sentiment.toFixed(2)}
            meta={Object.entries(s.counts).map(([k, v]) => `${v} ${k}`).join(' · ')} tone="sm" />
      {s.daily?.length > 1 && (
        <div style={{ marginTop: 10 }}><SentimentTimeline daily={s.daily} /></div>
      )}
      <ul className="list" style={{ marginTop: 10 }}>
        {s.articles.slice(0, 8).map((a) => (
          <li key={a.url || a.headline}>
            <span className="icon" aria-hidden="true" style={{
              color: token(a.sentiment > 0.05 ? '--status-good'
                : a.sentiment < -0.05 ? '--status-critical' : '--text-muted'),
            }}>{a.sentiment > 0.05 ? '▲' : a.sentiment < -0.05 ? '▼' : '■'}</span>
            <span style={{ flex: 1, minWidth: 0 }}>
              {a.url ? <a href={a.url} target="_blank" rel="noreferrer">{a.headline}</a> : a.headline}
              <div className="muted" style={{ fontSize: 12 }}>
                {a.source} · score {fmtNum(a.sentiment, 2)}
              </div>
            </span>
            <span className="when">{new Date(a.published_at).toLocaleDateString()}</span>
          </li>
        ))}
      </ul>
    </Card>
  )
}

/* ------------------------------------------------------------------ */
export function AnomalyPanel({ anomalies }) {
  const { events, summary } = anomalies
  return (
    <Card
      title="Unusual activity"
      subtitle={`${summary.count} event(s) detected — z-score rules plus an isolation forest`}
    >
      {events.length === 0 ? (
        <p className="muted">Nothing out of the ordinary in this window.</p>
      ) : (
        <DataTable
          columns={[
            { key: 'date', header: 'Date' },
            {
              key: 'kind',
              header: 'Type',
              render: (r) => (
                <StatusBadge
                  level={r.severity > 0.75 ? 'critical' : r.severity > 0.5 ? 'serious' : 'warning'}
                  label={r.kind.replace(/_/g, ' ').toLowerCase()}
                />
              ),
            },
            { key: 'return_pct', header: 'Move', render: (r) => <Delta value={r.return_pct / 100} /> },
            { key: 'volume', header: 'Volume', render: (r) => fmtCompact(r.volume) },
            { key: 'severity', header: 'Severity', render: (r) => r.severity.toFixed(2) },
            { key: 'detail', header: 'What happened', wrap: true,
              render: (r) => <span className="muted">{r.detail}</span> },
          ]}
          rows={events.slice(0, 25).map((e, i) => ({ ...e, __key: `${e.date}-${e.kind}-${i}` }))}
        />
      )}
    </Card>
  )
}

/* ------------------------------------------------------------------ */
export function AlertsPanel({ alerts, onSelect }) {
  return (
    <Card title="Alerts" subtitle="Signals, risk grades and anomalies worth a look">
      {alerts.length === 0 ? (
        <p className="muted">No alerts at the current thresholds.</p>
      ) : (
        <ul className="list">
          {alerts.slice(0, 14).map((a, i) => (
            <li key={`${a.symbol}-${a.kind}-${a.date}-${i}`}>
              <StatusBadge level={a.level} label={a.kind.toLowerCase()} />
              <span style={{ flex: 1, minWidth: 0 }}>
                <button className="linkish" onClick={() => onSelect?.(a.symbol)}>{a.symbol}</button>
                <div style={{ color: 'var(--text-secondary)' }}>{a.message}</div>
              </span>
              <span className="when">{a.date}</span>
            </li>
          ))}
        </ul>
      )}
    </Card>
  )
}

/* ------------------------------------------------------------------ */
export function OverviewTable({ cards, onSelect }) {
  return (
    <DataTable
      columns={[
        {
          key: 'symbol', header: 'Symbol',
          render: (r) => <button className="linkish" onClick={() => onSelect(r.symbol)}>{r.symbol}</button>,
        },
        { key: 'latest_close', header: 'Close', render: (r) => fmtNum(r.latest_close) },
        { key: 'change_1d', header: '1 day', render: (r) => <Delta value={r.change_1d} /> },
        { key: 'change_1m', header: '1 month', render: (r) => <Delta value={r.change_1m} /> },
        {
          key: 'prob_up', header: 'P(up)',
          render: (r) => (r.prob_up === undefined ? '—' : `${(r.prob_up * 100).toFixed(1)}%`),
        },
        {
          key: 'direction', header: 'Signal',
          render: (r) => (r.direction
            ? <StatusBadge level={DIRECTION_TONE[r.direction]} label={r.direction} />
            : <span className="muted">—</span>),
        },
        { key: 'risk_score', header: 'Risk', render: (r) => (r.risk_score?.toFixed(0) ?? '—') },
        {
          key: 'grade', header: 'Grade',
          render: (r) => (r.grade
            ? <StatusBadge level={GRADE_TONE[r.grade]} label={r.grade} />
            : <span className="muted">—</span>),
        },
      ]}
      rows={cards.map((c) => ({ ...c, __key: c.symbol }))}
    />
  )
}

export { GRADE_TONE, DIRECTION_TONE, RISK_GRADE_STATUS }
