import React, { useState } from 'react';
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import { Eye, X, RefreshCw, Clock, CheckCircle, AlertCircle, Loader, FileJson, BrainCircuit } from 'lucide-react';
import Card from '../components/ui/Card';
import Button from '../components/ui/Button';
import { jobsAPI } from '../api/jobs';
import { Job, JobStatus } from '../types/jobs';

const Jobs: React.FC = () => {
  const [selectedStatus, setSelectedStatus] = useState<JobStatus | 'all'>('all');
  const [selectedJob, setSelectedJob] = useState<Job | null>(null);
  const queryClient = useQueryClient();

  const { data: jobsData, isLoading, refetch } = useQuery({
    queryKey: ['jobs', selectedStatus],
    queryFn: () =>
      jobsAPI.list({
        status: selectedStatus === 'all' ? undefined : selectedStatus,
        limit: 100,
      }),
    refetchInterval: 5000, // Auto-refresh every 5 seconds
  });

  const cancelJobMutation = useMutation({
    mutationFn: (jobId: string) => jobsAPI.cancel(jobId),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['jobs'] });
    },
  });

  const getStatusIcon = (status: JobStatus) => {
    switch (status) {
      case 'completed':
        return <CheckCircle className="w-5 h-5 text-green-600" />;
      case 'failed':
        return <AlertCircle className="w-5 h-5 text-red-600" />;
      case 'running':
        return <Loader className="w-5 h-5 text-blue-600 animate-spin" />;
      case 'pending':
        return <Clock className="w-5 h-5 text-yellow-600" />;
      case 'cancelled':
        return <X className="w-5 h-5 text-gray-600" />;
      default:
        return <Clock className="w-5 h-5 text-gray-600" />;
    }
  };

  const getStatusColor = (status: JobStatus) => {
    switch (status) {
      case 'completed':
        return 'bg-green-100 text-green-800';
      case 'failed':
        return 'bg-red-100 text-red-800';
      case 'running':
        return 'bg-blue-100 text-blue-800';
      case 'pending':
        return 'bg-yellow-100 text-yellow-800';
      case 'cancelled':
        return 'bg-gray-100 text-gray-800';
      default:
        return 'bg-gray-100 text-gray-800';
    }
  };

  const formatDuration = (job: Job) => {
    if (job.completed_at && job.started_at) {
      const duration =
        (new Date(job.completed_at).getTime() - new Date(job.started_at).getTime()) / 1000;
      const hours = Math.floor(duration / 3600);
      const minutes = Math.floor((duration % 3600) / 60);
      const seconds = Math.floor(duration % 60);
      return `${hours}h ${minutes}m ${seconds}s`;
    }
    if (job.started_at) {
      const elapsed = (Date.now() - new Date(job.started_at).getTime()) / 1000;
      const hours = Math.floor(elapsed / 3600);
      const minutes = Math.floor((elapsed % 3600) / 60);
      return `${hours}h ${minutes}m (running)`;
    }
    return 'N/A';
  };

  const statusFilters = [
    { value: 'all', label: 'All Jobs' },
    { value: 'pending', label: 'Pending' },
    { value: 'running', label: 'Running' },
    { value: 'completed', label: 'Completed' },
    { value: 'failed', label: 'Failed' },
    { value: 'cancelled', label: 'Cancelled' },
  ];

  return (
    <div className="space-y-6">
      <div className="flex justify-between items-center">
        <div>
          <h1 className="text-2xl font-bold text-gray-900">Training Jobs</h1>
          <p className="text-gray-600 mt-1">Monitor your fine-tuning jobs</p>
        </div>
        <Button onClick={() => refetch()} variant="secondary">
          <RefreshCw className="w-4 h-4 mr-2" />
          Refresh
        </Button>
      </div>

      {/* Status Filters */}
      <div className="flex gap-2 overflow-x-auto pb-2">
        {statusFilters.map((filter) => (
          <button
            key={filter.value}
            onClick={() => setSelectedStatus(filter.value as JobStatus | 'all')}
            className={`px-4 py-2 rounded-lg whitespace-nowrap transition-colors ${
              selectedStatus === filter.value
                ? 'bg-blue-600 text-white'
                : 'bg-white text-gray-700 border border-gray-300 hover:bg-gray-50'
            }`}
          >
            {filter.label}
          </button>
        ))}
      </div>

      {isLoading ? (
        <div className="text-center py-12">
          <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-600 mx-auto"></div>
          <p className="mt-4 text-gray-600">Loading jobs...</p>
        </div>
      ) : jobsData?.jobs && jobsData.jobs.length > 0 ? (
        <div className="grid grid-cols-1 gap-4">
          {jobsData.jobs.map((job) => (
            <Card key={job.job_id} className="p-6">
              <div className="flex items-start justify-between mb-4">
                <div className="flex items-start gap-3 flex-1">
                  {getStatusIcon(job.status)}
                  <div className="flex-1">
                    <div className="flex items-center gap-3">
                      <h3 className="text-lg font-semibold text-gray-900">{job.name}</h3>
                      <span
                        className={`px-2 py-1 rounded-full text-xs font-medium ${getStatusColor(
                          job.status
                        )}`}
                      >
                        {job.status}
                      </span>
                    </div>
                    <div className="flex gap-6 mt-2 text-sm text-gray-600">
                      <span>
                        <strong>Type:</strong> {job.job_type}
                      </span>
                      <span>
                        <strong>Created:</strong>{' '}
                        {new Date(job.created_at).toLocaleString()}
                      </span>
                      {job.started_at && (
                        <span>
                          <strong>Duration:</strong> {formatDuration(job)}
                        </span>
                      )}
                    </div>
                    {job.base_model && (
                      <p className="mt-2 text-sm text-gray-600">
                        <strong>Model:</strong> {job.base_model}
                      </p>
                    )}
                  </div>
                </div>
              </div>

              {job.metrics && (
                <div className="grid grid-cols-4 gap-4 p-4 bg-gray-50 rounded-lg mb-4">
                  {Object.entries(job.metrics).slice(0, 4).map(([key, value]) => (
                    <div key={key}>
                      <p className="text-xs text-gray-600 capitalize">
                        {key.replace(/_/g, ' ')}
                      </p>
                      <p className="text-lg font-semibold">
                        {typeof value === 'number' ? value.toFixed(4) : value}
                      </p>
                    </div>
                  ))}
                </div>
              )}

              {job.error && (
                <div className="mb-4 p-3 bg-red-50 border border-red-200 rounded-lg">
                  <p className="text-sm text-red-800">
                    <strong>Error:</strong> {job.error}
                  </p>
                </div>
              )}

              <div className="flex gap-2">
                <Button variant="secondary" size="sm" onClick={() => setSelectedJob(job)}>
                  <Eye className="w-4 h-4 mr-1" />
                  View Details
                </Button>
                {(job.status === 'running' || job.status === 'pending') && (
                  <Button
                    variant="danger"
                    size="sm"
                    onClick={() => {
                      if (confirm('Are you sure you want to cancel this job?')) {
                        cancelJobMutation.mutate(job.job_id);
                      }
                    }}
                  >
                    <X className="w-4 h-4 mr-1" />
                    Cancel
                  </Button>
                )}
              </div>
            </Card>
          ))}
        </div>
      ) : (
        <Card className="p-12 text-center">
          <p className="text-gray-600">
            No jobs found. Start a fine-tuning job to see it here!
          </p>
          <Button className="mt-4" onClick={() => (window.location.href = '/finetune')}>
            Start Fine-Tuning
          </Button>
        </Card>
      )}
      <JobDetailModal job={selectedJob} onClose={() => setSelectedJob(null)} />
    </div>
  );
};

const JobDetailModal: React.FC<{ job: Job | null; onClose: () => void }> = ({ job, onClose }) => {
  if (!job) return null;

  return (
    <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center p-4 z-50">
      <Card className="w-full max-w-2xl max-h-[90vh] overflow-y-auto">
        <div className="flex justify-between items-center mb-4">
          <h2 className="text-xl font-bold">Job Details</h2>
          <Button onClick={onClose} variant="ghost" size="sm">
            <X className="w-5 h-5" />
          </Button>
        </div>
        <div className="space-y-4">
          <div>
            <h3 className="font-semibold text-lg mb-2">Configuration</h3>
            <pre className="bg-gray-100 p-4 rounded-lg text-sm overflow-x-auto">
              {JSON.stringify(job.config, null, 2)}
            </pre>
          </div>
          {job.metrics && (
            <div>
              <h3 className="font-semibold text-lg mb-2">Metrics</h3>
              <pre className="bg-gray-100 p-4 rounded-lg text-sm overflow-x-auto">
                {JSON.stringify(job.metrics, null, 2)}
              </pre>
            </div>
          )}
          {job.artifacts && (
            <div>
              <h3 className="font-semibold text-lg mb-2">Artifacts</h3>
              <pre className="bg-gray-100 p-4 rounded-lg text-sm overflow-x-auto">
                {JSON.stringify(job.artifacts, null, 2)}
              </pre>
            </div>
          )}
        </div>
      </Card>
    </div>
  );
};

export default Jobs;
