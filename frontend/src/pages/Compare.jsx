import { useState } from 'react'
import { api, fmtNum } from '../lib/api'
import { useAsync } from '../lib/useAsync'
import { ComparisonChart, CorrelationHeatmap } from '../components/charts'
import { colorFor } from '../lib/theme'
import { Card, DataTable, Delta, ErrorBox, Spinner, StatusBadge } from '../components/ui'
import { DIRECTION_TONE, GRADE_TONE } from '../components/panels'

export default function Compare({ universe, initial }) {
  const [selected, setSelected] = useState(
    () => (initial?.length >= 2 ? initial : universe.slice(0, 4)),
  )
  const cmp = useAsync(
    () => api.compare(selected),
    [selected.join(',')],
    { enabled: selected.length >= 2 },
  )

  const toggle = (s) => setSelected((cur) => (
    cur.includes(s) ? cur.filter((x) => x !== s) : cur.length >= 8 ? cur : [...cur, s]
  ))

  return (
    <div className="grid" style={{ gap: 16 }}>
      <Card title="Pick 2–8 symbols" subtitle="Colour follows the symbol, so removing one never repaints the rest">
        <div className="chips">
          {universe.map((s) => (
            <button key={s} className="chip" aria-pressed={selected.includes(s)}
                    onClick={() => toggle(s)}>
              <span className="swatch" aria-hidden="true"
                    style={{ background: selected.includes(s) ? colorFor(s) : 'var(--axis)' }} />
              {s}
            </button>
          ))}
        </div>
        {selected.length < 2 && <p className="footnote">Select at least two.</p>}
      </Card>

      {cmp.error && <ErrorBox error={cmp.error} onRetry={cmp.refresh} />}

      {cmp.data && (
        <>
          <Card title="Relative performance"
                subtitle="Every series indexed to 100 at the start of its history — one axis, so the comparison is honest">
            <div className={cmp.loading ? 'stale' : ''}>
              <ComparisonChart series={cmp.data.series} />
            </div>
          </Card>

          <Card title="Side by side" subtitle="Returns, risk and the model's current lean">
            <DataTable
              columns={[
                { key: 'symbol', header: 'Symbol' },
                { key: 'latest_close', header: 'Close', render: (r) => fmtNum(r.latest_close) },
                { key: 'return_1m', header: '1M', render: (r) => <Delta value={r.return_1m} /> },
                { key: 'return_3m', header: '3M', render: (r) => <Delta value={r.return_3m} /> },
                { key: 'return_1y', header: '1Y', render: (r) => <Delta value={r.return_1y} /> },
                { key: 'annualized_vol', header: 'Volatility', render: (r) => fmtNum(r.annualized_vol, 2) },
                { key: 'max_drawdown', header: 'Max DD', render: (r) => <Delta value={-r.max_drawdown} /> },
                { key: 'sharpe', header: 'Sharpe', render: (r) => fmtNum(r.sharpe) },
                { key: 'beta', header: 'Beta', render: (r) => fmtNum(r.beta) },
                { key: 'risk_score', header: 'Risk', render: (r) => (r.risk_score?.toFixed(0) ?? '—') },
                {
                  key: 'grade', header: 'Grade',
                  render: (r) => (r.grade ? <StatusBadge level={GRADE_TONE[r.grade]} label={r.grade} /> : '—'),
                },
                {
                  key: 'direction', header: 'Signal',
                  render: (r) => (r.direction
                    ? <StatusBadge level={DIRECTION_TONE[r.direction]} label={r.direction} />
                    : <span className="muted">—</span>),
                },
              ]}
              rows={cmp.data.rows.map((r) => ({ ...r, __key: r.symbol }))}
            />
          </Card>

          {Object.keys(cmp.data.correlation || {}).length > 1 && (
            <Card title="Return correlation"
                  subtitle="Daily log returns. Blue = move apart, red = move together, grey = unrelated">
              <CorrelationHeatmap correlation={cmp.data.correlation} />
            </Card>
          )}
        </>
      )}

      {cmp.loading && !cmp.data && <Spinner label="Comparing…" />}
    </div>
  )
}
