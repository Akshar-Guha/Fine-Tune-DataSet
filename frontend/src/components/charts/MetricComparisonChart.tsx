import React from 'react';
import {
  ResponsiveContainer,
  BarChart,
  CartesianGrid,
  XAxis,
  YAxis,
  Tooltip,
  Legend,
  Bar,
} from 'recharts';

import { ColabChartEntry } from '../../types/colab';

interface MetricComparisonChartProps {
  data: ColabChartEntry[];
  title?: string;
}

const MetricComparisonChart: React.FC<MetricComparisonChartProps> = ({ data, title }) => {
  if (!data || data.length === 0) {
    return null;
  }

  return (
    <div>
      {title && <h4 className="text-lg font-semibold mb-4">{title}</h4>}
      <div className="h-64">
        <ResponsiveContainer width="100%" height="100%">
          <BarChart data={data}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis dataKey="metric" />
            <YAxis allowDecimals={true} />
            <Tooltip />
            <Legend />
            <Bar dataKey="baseline" fill="#60a5fa" name="Baseline" />
            <Bar dataKey="projected" fill="#34d399" name="Projected" />
          </BarChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
};

export default MetricComparisonChart;
