import { apiClient } from './client';
import {
  Job,
  JobCreateRequest,
  JobsResponse,
  JobStatus,
  JobType,
} from '../types/jobs';

export const jobsAPI = {
  // List all jobs
  list: async (params?: {
    job_type?: JobType;
    status?: JobStatus;
    skip?: number;
    limit?: number;
  }): Promise<JobsResponse> => {
    return apiClient.get('/jobs', { params });
  },

  // Get job details
  get: async (jobId: string): Promise<Job> => {
    return apiClient.get(`/jobs/${jobId}`);
  },

  // Submit a new fine-tuning job
  submitFineTuning: async (job: JobCreateRequest): Promise<Job> => {
    return apiClient.post('/jobs/finetuning', job);
  },

  // Submit a new quantization job
  submitQuantization: async (job: { model_id: string; method: string }): Promise<Job> => {
    return apiClient.post('/jobs/quantization', job);
  },

  // Submit a new RLHF job
  submitRLHF: async (job: JobCreateRequest): Promise<Job> => {
    return apiClient.post('/jobs/rlhf', job);
  },

  // Cancel a job
  cancel: async (jobId: string): Promise<{ message: string }> => {
    return apiClient.post(`/jobs/${jobId}/cancel`);
  },

  // Get job logs
  getLogs: async (jobId: string, tail: number = 100): Promise<{ logs: string[] }> => {
    return apiClient.get(`/jobs/${jobId}/logs`, { params: { tail } });
  },
};
