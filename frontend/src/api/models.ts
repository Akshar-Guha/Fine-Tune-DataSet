import { apiClient } from './client';
import {
  Model,
  ModelMetrics,
  ModelVersion,
  ModelSearchResult,
  Deployment,
} from '../types/models';

export const modelsAPI = {
  // Search for models on HuggingFace Hub
  search: async (params: {
    query?: string;
    task?: string;
    limit?: number;
  }): Promise<ModelSearchResult[]> => {
    return apiClient.get('/models/search', { params });
  },

  // Download a model
  download: async (modelId: string): Promise<{ path: string }> => {
    return apiClient.post(`/models/download`, { model_id: modelId });
  },

  // List all local models
  list: async (): Promise<Model[]> => {
    return apiClient.get('/models/local');
  },

  // Get model details
  get: async (modelId: string): Promise<Model> => {
    return apiClient.get(`/models/${modelId}`);
  },

  // Delete a local model
  delete: async (modelId: string): Promise<{ success: boolean }> => {
    return apiClient.delete(`/models/${modelId}`);
  },

  // Get model metrics
  getMetrics: async (
    modelId: string,
    timeRange?: string
  ): Promise<ModelMetrics> => {
    return apiClient.get(`/models/${modelId}/metrics`, {
      params: { time_range: timeRange },
    });
  },

  // Get model versions
  getVersions: async (modelId: string): Promise<ModelVersion[]> => {
    return apiClient.get(`/models/${modelId}/versions`);
  },

  // Quantize a model
  quantize: async (
    modelId: string,
    method: 'awq' | 'gptq' | 'gguf',
    bits: number = 4
  ): Promise<{ artifact_id: string }> => {
    return apiClient.post(`/models/${modelId}/quantize`, { method, bits });
  },

  // Deploy a model
  deploy: async (
    modelId: string,
    backend: 'ollama' | 'tgi' | 'vllm'
  ): Promise<Deployment> => {
    return apiClient.post(`/models/${modelId}/deploy`, { backend });
  },

  // List deployments
  listDeployments: async (): Promise<Deployment[]> => {
    return apiClient.get('/deployments');
  },

  // Stop a deployment
  stopDeployment: async (deploymentId: string): Promise<{ success: boolean }> => {
    return apiClient.delete(`/deployments/${deploymentId}`);
  },
};
