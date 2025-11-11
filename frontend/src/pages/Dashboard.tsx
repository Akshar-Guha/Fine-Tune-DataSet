import React, { useEffect, useState } from 'react';
import { useQuery } from '@tanstack/react-query';
import { Activity, TrendingUp, Zap, DollarSign } from 'lucide-react';

import Card from '../components/ui/Card';
import { modelsAPI } from '../api/models';
import { jobsAPI } from '../api/jobs';

interface SummaryCardProps {
  title: string;
  value: string | number;
  icon: React.ReactNode;
  trend?: { value: number; isPositive: boolean };
}

const SummaryCard: React.FC<SummaryCardProps> = ({ title, value, icon, trend }) => {
  return (
    <Card className="p-6">
      <div className="flex items-center justify-between">
        <div>
          <p className="text-sm font-medium text-gray-600">{title}</p>
          <p className="text-2xl font-bold mt-2">{value}</p>
          {trend && (
            <p className={`text-sm mt-2 ${trend.isPositive ? 'text-green-600' : 'text-red-600'}`}>
              {trend.isPositive ? '↑' : '↓'} {trend.value}% from last week
            </p>
          )}
        </div>
        <div className="p-3 bg-blue-100 rounded-lg text-blue-600">
          {icon}
        </div>
      </div>
    </Card>
  );
};

const Dashboard: React.FC = () => {
  // Fetch models data
  const { data: modelsData } = useQuery({
    queryKey: ['models'],
    queryFn: () => modelsAPI.list({ limit: 10 }),
  });

  // Fetch jobs data
  const { data: jobsData } = useQuery({
    queryKey: ['jobs'],
    queryFn: () => jobsAPI.list({ limit: 10 }),
  });

  // Count models by status
  const [modelStatusCounts, setModelStatusCounts] = useState({
    deployed: 0,
    training: 0,
    completed: 0,
  });

  // Count jobs by status
  const [jobStatusCounts, setJobStatusCounts] = useState({
    pending: 0,
    running: 0,
    completed: 0,
    failed: 0,
  });

  useEffect(() => {
    if (modelsData?.models) {
      const counts = {
        deployed: 0,
        training: 0,
        completed: 0,
      };
      
      modelsData.models.forEach(model => {
        if (model.versions && model.versions.length > 0) {
          const latestVersion = model.versions[0];
          if (latestVersion.status === 'deployed') counts.deployed++;
          if (latestVersion.status === 'training') counts.training++;
          if (latestVersion.status === 'completed') counts.completed++;
        }
      });
      
      setModelStatusCounts(counts);
    }
  }, [modelsData]);

  useEffect(() => {
    if (jobsData?.jobs) {
      const counts = {
        pending: 0,
        running: 0,
        completed: 0,
        failed: 0,
      };
      
      jobsData.jobs.forEach(job => {
        if (job.status === 'pending') counts.pending++;
        if (job.status === 'running') counts.running++;
        if (job.status === 'completed') counts.completed++;
        if (job.status === 'failed') counts.failed++;
      });
      
      setJobStatusCounts(counts);
    }
  }, [jobsData]);

  return (
    <div className="space-y-6">
      <h1 className="text-2xl font-bold text-gray-900">Dashboard</h1>
      
      {/* Summary Cards */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
        <SummaryCard
          title="Active Models"
          value={modelsData?.total || 0}
          icon={<Activity className="w-6 h-6" />}
          trend={{ value: 12, isPositive: true }}
        />
        
        <SummaryCard
          title="Running Jobs"
          value={jobStatusCounts.running}
          icon={<TrendingUp className="w-6 h-6" />}
          trend={{ value: 5, isPositive: true }}
        />
        
        <SummaryCard
          title="Deployed Models"
          value={modelStatusCounts.deployed}
          icon={<Zap className="w-6 h-6" />}
          trend={{ value: 8, isPositive: true }}
        />
        
        <SummaryCard
          title="Failed Jobs"
          value={jobStatusCounts.failed}
          icon={<DollarSign className="w-6 h-6" />}
          trend={{ value: 3, isPositive: false }}
        />
      </div>
      
      {/* Recent Jobs */}
      <Card title="Recent Jobs">
        <div className="overflow-x-auto">
          <table className="min-w-full divide-y divide-gray-200">
            <thead className="bg-gray-50">
              <tr>
                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Job Name</th>
                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Type</th>
                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Status</th>
                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Created</th>
              </tr>
            </thead>
            <tbody className="bg-white divide-y divide-gray-200">
              {jobsData?.jobs.slice(0, 5).map((job) => (
                <tr key={job.job_id}>
                  <td className="px-6 py-4 whitespace-nowrap text-sm font-medium text-gray-900">{job.name}</td>
                  <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">{job.job_type}</td>
                  <td className="px-6 py-4 whitespace-nowrap">
                    <span className={`px-2 inline-flex text-xs leading-5 font-semibold rounded-full 
                      ${job.status === 'completed' ? 'bg-green-100 text-green-800' : 
                        job.status === 'failed' ? 'bg-red-100 text-red-800' :
                        job.status === 'running' ? 'bg-blue-100 text-blue-800' :
                        'bg-yellow-100 text-yellow-800'}`}>
                      {job.status}
                    </span>
                  </td>
                  <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                    {new Date(job.created_at).toLocaleString()}
                  </td>
                </tr>
              ))}
              {(!jobsData?.jobs || jobsData.jobs.length === 0) && (
                <tr>
                  <td colSpan={4} className="px-6 py-4 text-center text-sm text-gray-500">
                    No jobs found
                  </td>
                </tr>
              )}
            </tbody>
          </table>
        </div>
      </Card>
      
      {/* Recent Models */}
      <Card title="Recent Models">
        <div className="overflow-x-auto">
          <table className="min-w-full divide-y divide-gray-200">
            <thead className="bg-gray-50">
              <tr>
                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Model Name</th>
                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Base Model</th>
                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Latest Version</th>
                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Status</th>
              </tr>
            </thead>
            <tbody className="bg-white divide-y divide-gray-200">
              {modelsData?.models.slice(0, 5).map((model) => (
                <tr key={model.id}>
                  <td className="px-6 py-4 whitespace-nowrap text-sm font-medium text-gray-900">{model.name}</td>
                  <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">{model.base_model}</td>
                  <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                    {model.versions && model.versions.length > 0 ? model.versions[0].version : 'N/A'}
                  </td>
                  <td className="px-6 py-4 whitespace-nowrap">
                    <span className={`px-2 inline-flex text-xs leading-5 font-semibold rounded-full 
                      ${model.versions && model.versions.length > 0 && model.versions[0].status === 'deployed' ? 'bg-green-100 text-green-800' : 
                        model.versions && model.versions.length > 0 && model.versions[0].status === 'training' ? 'bg-blue-100 text-blue-800' :
                        'bg-gray-100 text-gray-800'}`}>
                      {model.versions && model.versions.length > 0 ? model.versions[0].status : 'N/A'}
                    </span>
                  </td>
                </tr>
              ))}
              {(!modelsData?.models || modelsData.models.length === 0) && (
                <tr>
                  <td colSpan={4} className="px-6 py-4 text-center text-sm text-gray-500">
                    No models found
                  </td>
                </tr>
              )}
            </tbody>
          </table>
        </div>
      </Card>
    </div>
  );
};

export default Dashboard;
