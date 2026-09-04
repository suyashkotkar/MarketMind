import { api, fmtCompact, fmtNum } from '../lib/api'
import { useAsync } from '../lib/useAsync'
import {
  MacdChart, PriceChart, RsiChart, VolumeChart,
} from '../components/charts'
import {
  AnomalyPanel, PredictionPanel, RiskPanel, SentimentPanel,
} from '../components/panels'
import { Card, Delta, ErrorBox, Spinner, Tile } from '../components/ui'

export default function StockDetail({ symbol, days, config }) {
  const history = useAsync(() => api.history(symbol, days), [symbol, days])
  const stats = useAsync(() => api.stats(symbol), [symbol])
  const prediction = useAsync(() => api.prediction(symbol), [symbol])
  const risk = useAsync(() => api.risk(symbol), [symbol])
  const sentiment = useAsync(() => api.sentiment(symbol, 40), [symbol])
  const anomalies = useAsync(() => api.anomalies(symbol, days), [symbol, days])

  if (history.error && !history.data) {
    return <ErrorBox error={history.error} onRetry={history.refresh} />
  }

  const bars = history.data?.bars ?? []
  const indicators = history.data?.indicators ?? {}
  const s = stats.data

  return (
    <div className="grid" style={{ gap: 16 }}>
      <Card title={`${symbol} snapshot`} subtitle={s ? `${s.n_bars} sessions · ${s.first_date} → ${s.last_date}` : undefined}>
        <div className="grid cols-4">
          <Tile label="Last close" value={s ? fmtNum(s.latest_close) : '—'}
                meta={s ? <>1 day <Delta value={s.change_1d} /></> : null} />
          <Tile label="1 month" value={s ? <Delta value={s.change_1m} /> : '—'} tone="sm" />
          <Tile label="1 year" value={s ? <Delta value={s.change_1y} /> : '—'} tone="sm" />
          <Tile label="52-week range" tone="sm"
                value={s ? `${fmtNum(s.low_52w, 0)} – ${fmtNum(s.high_52w, 0)}` : '—'}
                meta={s ? `avg volume ${fmtCompact(s.avg_volume_30d)}` : null} />
        </div>
      </Card>

      <Card title="Price and moving averages" subtitle="Candles with the 20- and 50-session averages">
        <div className={history.loading ? 'stale' : ''}>
          {bars.length ? <PriceChart bars={bars} indicators={indicators} symbol={symbol} />
            : <Spinner />}
        </div>
      </Card>

      <div className="grid cols-2">
        <Card title="Volume" subtitle="Shares traded per session">
          {bars.length ? <VolumeChart bars={bars} /> : <Spinner />}
        </Card>
        <Card title="RSI (14)" subtitle="Momentum oscillator, Wilder smoothing">
          {indicators.rsi_14 ? <RsiChart bars={bars} indicators={indicators} /> : <Spinner />}
        </Card>
      </div>

      <Card title="MACD (12, 26, 9)" subtitle="Trend strength; the histogram is MACD minus its signal line">
        {indicators.macd ? <MacdChart bars={bars} indicators={indicators} /> : <Spinner />}
      </Card>

      <div className="grid cols-2">
        {prediction.error
          ? <Card title="Direction"><ErrorBox error={prediction.error} onRetry={prediction.refresh} /></Card>
          : prediction.data
            ? <PredictionPanel prediction={prediction.data} config={config} />
            : <Card title="Direction"><Spinner /></Card>}

        {risk.error
          ? <Card title="Risk"><ErrorBox error={risk.error} onRetry={risk.refresh} /></Card>
          : risk.data ? <RiskPanel risk={risk.data} /> : <Card title="Risk"><Spinner /></Card>}
      </div>

      <div className="grid cols-2">
        {sentiment.error
          ? <Card title="News sentiment"><p className="muted">{String(sentiment.error.message)}</p></Card>
          : sentiment.data ? <SentimentPanel sentiment={sentiment.data} />
            : <Card title="News sentiment"><Spinner /></Card>}

        {anomalies.data ? <AnomalyPanel anomalies={anomalies.data} />
          : <Card title="Unusual activity"><Spinner /></Card>}
      </div>
    </div>
  )
}
