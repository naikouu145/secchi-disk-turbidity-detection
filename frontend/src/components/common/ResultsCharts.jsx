import {
  Bar,
  BarChart,
  CartesianGrid,
  Cell,
  Legend,
  Pie,
  PieChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts'

function toPercent(value, digits = 1) {
  const safe = Number(value)
  if (!Number.isFinite(safe)) {
    return '0.0%'
  }
  return `${(safe * 100).toFixed(digits)}%`
}

export function SourcePieChart({ data = [] }) {
  return (
    <ResponsiveContainer width="100%" height="100%">
      <PieChart>
        <Pie
          data={data}
          dataKey="value"
          nameKey="name"
          innerRadius={38}
          outerRadius={68}
          label={(entry) => `${entry.name}: ${(entry.value * 100).toFixed(0)}%`}
        >
          {data.map((entry) => (
            <Cell
              key={entry.name}
              fill={entry.name === 'Algal' ? '#0f766e' : entry.name === 'Sediment' ? '#b45309' : '#64748b'}
            />
          ))}
        </Pie>
        <Tooltip formatter={(value) => toPercent(Number(value), 1)} />
      </PieChart>
    </ResponsiveContainer>
  )
}

export function FeatureBarChart({ data = [] }) {
  return (
    <ResponsiveContainer width="100%" height="100%">
      <BarChart data={data} layout="vertical" margin={{ left: 24, right: 8, top: 8, bottom: 8 }}>
        <CartesianGrid strokeDasharray="3 3" />
        <XAxis type="number" />
        <YAxis type="category" dataKey="name" width={140} />
        <Tooltip formatter={(value) => Number(value).toFixed(4)} />
        <Bar dataKey="value" fill="#0f766e" radius={[0, 6, 6, 0]} />
      </BarChart>
    </ResponsiveContainer>
  )
}

export function ProbabilityBarChart({ data = [] }) {
  return (
    <ResponsiveContainer width="100%" height="100%">
      <BarChart data={data}>
        <CartesianGrid strokeDasharray="3 3" />
        <XAxis dataKey="name" />
        <YAxis tickFormatter={(value) => `${Math.round(value * 100)}%`} />
        <Tooltip formatter={(value) => toPercent(Number(value), 2)} />
        <Legend />
        <Bar dataKey="value" name="Probability" fill="#0f766e" />
      </BarChart>
    </ResponsiveContainer>
  )
}
