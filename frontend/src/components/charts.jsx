import { useMemo } from 'react'
import createPlotlyComponent from 'react-plotly.js/factory'
import Plotly from 'plotly.js-dist-min'
import {
  PLOT_CONFIG, baseLayout, colorFor, divergingScale, sequentialScale, token,
  useThemeEpoch,
} from '../lib/theme'
import { ChartWithTable, DataTable } from './ui'
import { fmtCompact, fmtNum, fmtPct } from '../lib/api'

const Plot = createPlotlyComponent(Plotly)

function Figure({ data, layout, height = 300 }) {
  return (
    <div className="chart-wrap">
      <Plot
        data={data}
        layout={{ ...layout, height }}
        config={PLOT_CONFIG}
        style={{ width: '100%' }}
        useResizeHandler
      />
    </div>
  )
}

/* ------------------------------------------------------------------ */
/* Price + trend. One y-axis; volume gets its own chart rather than a  */
/* second scale glued onto this one.                                   */
/* ------------------------------------------------------------------ */
export function PriceChart({ bars, indicators, symbol }) {
  const epoch = useThemeEpoch()
  const figure = useMemo(() => {
    const x = bars.map((b) => b.date)
    const close = bars.map((b) => b.adj_close)
    const sma20 = indicators?.sma_20_ratio
      ? close.map((c, i) => {
        const r = indicators.sma_20_ratio[i]
        return r === null || r === undefined ? null : c / (1 + r)
      })
      : null
    const sma50 = indicators?.sma_50_ratio
      ? close.map((c, i) => {
        const r = indicators.sma_50_ratio[i]
        return r === null || r === undefined ? null : c / (1 + r)
      })
      : null

    const traces = [
      {
        type: 'candlestick', x,
        open: bars.map((b) => b.open), high: bars.map((b) => b.high),
        low: bars.map((b) => b.low), close: bars.map((b) => b.close),
        name: symbol,
        // Up/down is a state, so it wears the delta tokens the tables use —
        // and the candle's own geometry already encodes the sign, so colour
        // is never the only channel carrying it.
        increasing: { line: { color: token('--delta-up'), width: 1 } },
        decreasing: { line: { color: token('--delta-down'), width: 1 } },
        hoverinfo: 'x+y',
      },
    ]
    if (sma20) traces.push({
      type: 'scatter', mode: 'lines', x, y: sma20, name: '20-day average',
      line: { color: token('--series-1'), width: 2 },
    })
    if (sma50) traces.push({
      type: 'scatter', mode: 'lines', x, y: sma50, name: '50-day average',
      line: { color: token('--series-4'), width: 2 },
    })

    return {
      data: traces,
      layout: baseLayout({
        showlegend: true,
        xaxis: { rangeslider: { visible: false }, type: 'date' },
        yaxis: { title: { text: 'Price', font: { size: 11, color: token('--text-muted') } }, tickprefix: '' },
        margin: { l: 8, r: 12, t: 8, b: 44 },
      }),
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [bars, indicators, symbol, epoch])

  const rows = bars.slice(-30).reverse()
  return (
    <ChartWithTable
      tableLabel="last 30 sessions"
      chart={<Figure {...figure} height={340} />}
      table={
        <DataTable
          caption="Most recent 30 sessions"
          columns={[
            { key: 'date', header: 'Date' },
            { key: 'open', header: 'Open', render: (r) => fmtNum(r.open) },
            { key: 'high', header: 'High', render: (r) => fmtNum(r.high) },
            { key: 'low', header: 'Low', render: (r) => fmtNum(r.low) },
            { key: 'close', header: 'Close', render: (r) => fmtNum(r.close) },
            { key: 'volume', header: 'Volume', render: (r) => fmtCompact(r.volume) },
          ]}
          rows={rows.map((r) => ({ ...r, __key: r.date }))}
        />
      }
    />
  )
}

export function VolumeChart({ bars }) {
  const epoch = useThemeEpoch()
  const figure = useMemo(() => ({
    data: [{
      type: 'bar', x: bars.map((b) => b.date), y: bars.map((b) => b.volume),
      name: 'Volume', marker: { color: token('--seq-250'), line: { width: 0 } },
      hovertemplate: '%{x|%d %b %Y}<br>%{y:.3s} shares<extra></extra>',
    }],
    layout: baseLayout({
      xaxis: { type: 'date' },
      yaxis: { title: { text: 'Shares', font: { size: 11, color: token('--text-muted') } } },
      bargap: 0.15,
    }),
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }), [bars, epoch])
  return <Figure {...figure} height={150} />
}

/* ------------------------------------------------------------------ */
/* Oscillators — separate panels, one scale each                       */
/* ------------------------------------------------------------------ */
export function RsiChart({ bars, indicators }) {
  const epoch = useThemeEpoch()
  const figure = useMemo(() => {
    const x = bars.map((b) => b.date)
    const shapes = [30, 70].map((v) => ({
      type: 'line', xref: 'paper', x0: 0, x1: 1, y0: v, y1: v,
      line: { color: token('--axis'), width: 1 },
    }))
    return {
      data: [{
        type: 'scatter', mode: 'lines', x, y: indicators.rsi_14, name: 'RSI (14)',
        line: { color: token('--series-1'), width: 2 },
        hovertemplate: '%{x|%d %b %Y}<br>RSI %{y:.1f}<extra></extra>',
      }],
      layout: baseLayout({
        xaxis: { type: 'date' },
        yaxis: { range: [0, 100], dtick: 25 },
        shapes,
        annotations: [
          { x: 0, xref: 'paper', y: 70, yanchor: 'bottom', text: 'overbought 70',
            showarrow: false, font: { size: 10, color: token('--text-muted') } },
          { x: 0, xref: 'paper', y: 30, yanchor: 'top', text: 'oversold 30',
            showarrow: false, font: { size: 10, color: token('--text-muted') } },
        ],
      }),
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [bars, indicators, epoch])
  return <Figure {...figure} height={190} />
}

export function MacdChart({ bars, indicators }) {
  const epoch = useThemeEpoch()
  const figure = useMemo(() => {
    const x = bars.map((b) => b.date)
    const hist = indicators.macd_hist || []
    return {
      data: [
        {
          type: 'bar', x, y: hist, name: 'Histogram',
          marker: {
            // Same up/down language as the candles and the table deltas; the
            // bar's side of the zero line carries the sign independently.
            color: hist.map((v) => (v >= 0 ? token('--delta-up') : token('--delta-down'))),
            line: { width: 0 },
          },
          hovertemplate: '%{x|%d %b %Y}<br>histogram %{y:.3f}<extra></extra>',
        },
        {
          type: 'scatter', mode: 'lines', x, y: indicators.macd, name: 'MACD',
          line: { color: token('--series-1'), width: 2 },
        },
        {
          type: 'scatter', mode: 'lines', x, y: indicators.macd_signal, name: 'Signal',
          line: { color: token('--series-2'), width: 2 },
        },
      ],
      layout: baseLayout({
        showlegend: true, xaxis: { type: 'date' }, bargap: 0.1,
        margin: { l: 8, r: 12, t: 8, b: 44 },
      }),
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [bars, indicators, epoch])
  return <Figure {...figure} height={210} />
}

/* ------------------------------------------------------------------ */
/* Risk components — nominal categories, so ONE series color           */
/* ------------------------------------------------------------------ */
export function RiskComponentChart({ components }) {
  const epoch = useThemeEpoch()
  const sorted = [...components].sort((a, b) => a.contribution - b.contribution)
  const labels = sorted.map((c) => c.name.replace(/_/g, ' '))
  const figure = useMemo(() => ({
    data: [{
      type: 'bar', orientation: 'h', x: sorted.map((c) => c.contribution), y: labels,
      marker: { color: token('--series-1'), line: { width: 0 } },
      text: sorted.map((c) => c.contribution.toFixed(1)),
      textposition: 'outside', cliponaxis: false,
      textfont: { color: token('--text-secondary'), size: 11 },
      hovertemplate: '%{y}<br>contributes %{x:.1f} of the total<extra></extra>',
    }],
    layout: baseLayout({
      xaxis: { title: { text: 'Points contributed to the 0–100 score', font: { size: 11, color: token('--text-muted') } } },
      yaxis: { gridcolor: 'rgba(0,0,0,0)' },
      bargap: 0.35, margin: { l: 8, r: 40, t: 8, b: 40 },
    }),
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }), [components, epoch])

  return (
    <ChartWithTable
      tableLabel="values"
      chart={<Figure {...figure} height={230} />}
      table={
        <DataTable
          columns={[
            { key: 'name', header: 'Component', render: (r) => r.name.replace(/_/g, ' ') },
            { key: 'raw', header: 'Raw', render: (r) => fmtNum(r.raw, 4) },
            { key: 'scaled', header: 'Scaled 0–100', render: (r) => fmtNum(r.scaled, 1) },
            { key: 'weight', header: 'Weight', render: (r) => fmtNum(r.weight, 2) },
            { key: 'contribution', header: 'Contribution', render: (r) => fmtNum(r.contribution, 1) },
          ]}
          rows={components.map((c) => ({ ...c, __key: c.name }))}
        />
      }
    />
  )
}

/* ------------------------------------------------------------------ */
/* Comparison — indexed to 100 so one axis serves every series         */
/* ------------------------------------------------------------------ */
export function ComparisonChart({ series }) {
  const epoch = useThemeEpoch()
  const symbols = Object.keys(series)
  const figure = useMemo(() => ({
    data: symbols.map((s) => ({
      type: 'scatter', mode: 'lines', name: s,
      x: series[s].dates, y: series[s].normalized,
      line: { color: colorFor(s), width: 2 },
      hovertemplate: `${s}: %{y:.1f}<extra></extra>`,
    })),
    layout: baseLayout({
      showlegend: true,
      xaxis: { type: 'date' },
      yaxis: { title: { text: 'Indexed to 100 at start', font: { size: 11, color: token('--text-muted') } } },
      margin: { l: 8, r: 12, t: 8, b: 44 },
    }),
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }), [series, epoch])
  return <Figure {...figure} height={340} />
}

/* ------------------------------------------------------------------ */
/* Correlation — polarity data, so a diverging scale with a gray zero  */
/* ------------------------------------------------------------------ */
export function CorrelationHeatmap({ correlation }) {
  const epoch = useThemeEpoch()
  const symbols = Object.keys(correlation)
  const z = symbols.map((r) => symbols.map((c) => correlation[r][c]))
  const figure = useMemo(() => ({
    data: [{
      type: 'heatmap', z, x: symbols, y: symbols,
      colorscale: divergingScale(), zmin: -1, zmax: 1, zmid: 0,
      xgap: 2, ygap: 2,
      colorbar: {
        title: { text: 'ρ', font: { size: 11, color: token('--text-muted') } },
        tickfont: { color: token('--text-muted'), size: 10 },
        outlinewidth: 0, thickness: 10, len: 0.8,
      },
      hovertemplate: '%{y} vs %{x}<br>correlation %{z:.2f}<extra></extra>',
    }],
    layout: baseLayout({
      xaxis: { gridcolor: 'rgba(0,0,0,0)', constrain: 'domain' },
      // Square cells: a correlation matrix stretched into letterboxes reads
      // as if the pairs had different weights.
      yaxis: { gridcolor: 'rgba(0,0,0,0)', autorange: 'reversed',
               scaleanchor: 'x', constrain: 'domain' },
      margin: { l: 8, r: 8, t: 8, b: 8 },
    }),
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }), [correlation, epoch])

  return (
    <ChartWithTable
      tableLabel="matrix"
      chart={<Figure {...figure} height={300} />}
      table={
        <DataTable
          columns={[{ key: 'sym', header: '' },
            ...symbols.map((s) => ({ key: s, header: s, render: (r) => fmtNum(r[s], 2) }))]}
          rows={symbols.map((r) => ({ __key: r, sym: r, ...correlation[r] }))}
        />
      }
    />
  )
}

/* ------------------------------------------------------------------ */
/* Sentiment over time — polarity around zero                          */
/* ------------------------------------------------------------------ */
export function SentimentTimeline({ daily }) {
  const epoch = useThemeEpoch()
  const figure = useMemo(() => ({
    data: [{
      type: 'bar', x: daily.map((d) => d.date), y: daily.map((d) => d.sentiment),
      marker: {
        color: daily.map((d) => (d.sentiment >= 0 ? token('--delta-up') : token('--delta-down'))),
        line: { width: 0 },
      },
      customdata: daily.map((d) => d.articles),
      hovertemplate: '%{x}<br>score %{y:.2f} · %{customdata} article(s)<extra></extra>',
    }],
    layout: baseLayout({
      xaxis: { type: 'date' },
      yaxis: { range: [-1, 1], zeroline: true, zerolinecolor: token('--axis'), zerolinewidth: 1 },
      bargap: 0.35,
    }),
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }), [daily, epoch])

  return (
    <ChartWithTable
      tableLabel="daily scores"
      chart={<Figure {...figure} height={200} />}
      table={
        <DataTable
          columns={[
            { key: 'date', header: 'Date' },
            { key: 'sentiment', header: 'Score', render: (r) => fmtNum(r.sentiment, 2) },
            { key: 'articles', header: 'Articles' },
          ]}
          rows={daily.map((d) => ({ ...d, __key: d.date }))}
        />
      }
    />
  )
}

/* ------------------------------------------------------------------ */
/* Feature importance — one series, ordered magnitude                  */
/* ------------------------------------------------------------------ */
export function FeatureImportanceChart({ importance }) {
  const epoch = useThemeEpoch()
  const entries = Object.entries(importance).slice(0, 12).reverse()
  const figure = useMemo(() => ({
    data: [{
      type: 'bar', orientation: 'h',
      x: entries.map(([, v]) => v), y: entries.map(([k]) => k.replace(/_/g, ' ')),
      marker: { color: token('--series-1'), line: { width: 0 } },
      hovertemplate: '%{y}<br>%{x:.1%} of total split gain<extra></extra>',
    }],
    layout: baseLayout({
      xaxis: { tickformat: '.0%' },
      yaxis: { gridcolor: 'rgba(0,0,0,0)' },
      bargap: 0.35, margin: { l: 8, r: 24, t: 8, b: 36 },
    }),
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }), [importance, epoch])
  return <Figure {...figure} height={300} />
}

export { fmtPct }
