import { api, fmtNum, fmtPct } from '../lib/api'
import { useAsync } from '../lib/useAsync'
import { FeatureImportanceChart } from '../components/charts'
import { Card, DataTable, ErrorBox, Spinner, Tile } from '../components/ui'

export default function ModelPage() {
  const model = useAsync(() => api.modelInfo(), [])
  const record = useAsync(() => api.trackRecord(), [])

  if (model.loading && !model.data) return <Spinner label="Loading the model card…" />
  if (model.error) return <ErrorBox error={model.error} onRetry={model.refresh} />

  const m = model.data
  const oof = m.metrics?.out_of_fold ?? {}
  const bt = m.metrics?.backtest ?? {}
  const cal = m.metrics?.calibration ?? {}

  return (
    <div className="grid" style={{ gap: 16 }}>
      <Card
        title="Model card"
        subtitle={`${m.model_type} · horizon ${m.horizon_days} sessions · trained ${new Date(m.trained_at).toLocaleString()}`}
      >
        <div className="grid cols-4">
          <Tile label="Out-of-fold ROC AUC" value={fmtNum(oof.roc_auc, 3)}
                meta={`${oof.n?.toLocaleString() ?? '—'} walk-forward test rows`} />
          <Tile label="Accuracy" value={fmtPct(oof.accuracy, 1)} tone="sm"
                meta={`base rate ${fmtPct(oof.base_rate, 1)}`} />
          <Tile label="Brier score" value={fmtNum(oof.brier, 3)} tone="sm"
                meta={`calibrated with ${cal.method ?? '—'}`} />
          <Tile label="Training rows" value={m.n_rows?.toLocaleString()} tone="sm"
                meta={`${m.n_features} features · ${m.tickers.length} tickers`} />
        </div>
        <p className="footnote">
          Five-day equity direction is close to a coin flip; an honest AUC here sits
          just above 0.50. Treat anything far higher as a leakage bug rather than a
          discovery. Folds are chronological with a {m.horizon_days}-session embargo,
          so no future row ever trains a model that scores a past one.
        </p>
      </Card>

      <div className="grid cols-2">
        <Card title="What the model leans on" subtitle="Share of total split gain, top 12 features">
          <FeatureImportanceChart importance={m.feature_importance} />
        </Card>

        <Card title="Signal vs buy-and-hold"
              subtitle="Non-overlapping periods, equal weight, no trading costs — a sanity check, not a strategy">
          <div className="grid cols-2">
            <Tile label="Signal mean return" value={fmtPct(bt.signal_mean_return, 2)} tone="sm"
                  meta={`per ${m.horizon_days}-session period`} />
            <Tile label="Hold mean return" value={fmtPct(bt.hold_mean_return, 2)} tone="sm" />
            <Tile label="Signal Sharpe" value={fmtNum(bt.signal_sharpe)} tone="sm" />
            <Tile label="Hold Sharpe" value={fmtNum(bt.hold_sharpe)} tone="sm" />
          </div>
          <p className="footnote">
            {bt.n_periods} periods · on average {fmtNum(bt.avg_positions, 1)} positions open ·
            hit rate {fmtPct(bt.signal_hit_rate, 1)}.
          </p>
        </Card>
      </div>

      <Card title="Per-fold results" subtitle="Each fold trains only on data that precedes its test window">
        <DataTable
          columns={[
            { key: 'fold', header: 'Fold' },
            { key: 'n', header: 'Test rows', render: (r) => r.n.toLocaleString() },
            { key: 'roc_auc', header: 'ROC AUC', render: (r) => fmtNum(r.roc_auc, 3) },
            { key: 'accuracy', header: 'Accuracy', render: (r) => fmtPct(r.accuracy, 1) },
            { key: 'precision', header: 'Precision', render: (r) => fmtPct(r.precision, 1) },
            { key: 'recall', header: 'Recall', render: (r) => fmtPct(r.recall, 1) },
            { key: 'brier', header: 'Brier', render: (r) => fmtNum(r.brier, 3) },
          ]}
          rows={(m.metrics?.folds ?? []).map((f) => ({ ...f, __key: f.fold }))}
        />
      </Card>

      <Card title="Live track record" subtitle="Predictions this deployment already made, scored against what happened">
        {record.data?.n
          ? (
            <div className="grid cols-3">
              <Tile label="Scored predictions" value={record.data.n.toLocaleString()} tone="sm" />
              <Tile label="Hit rate" value={fmtPct(record.data.hit_rate, 1)} tone="sm" />
              <Tile label="Mean realised return" value={fmtPct(record.data.mean_realized_return, 2)} tone="sm" />
            </div>
          )
          : <p className="muted">No prediction has aged past its horizon yet — check back in a few sessions.</p>}
      </Card>
    </div>
  )
}
