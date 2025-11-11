import { apiClient } from './client';
import {
  Dataset,
  DatasetSearchResult,
  DatasetVersion,
} from '../types/datasets';

export const datasetsAPI = {
  // Search for datasets on HuggingFace Hub
  search: async (params: {
    query?: string;
    task?: string;
    limit?: number;
  }): Promise<DatasetSearchResult[]> => {
    return apiClient.get('/datasets/search', { params });
  },

  // Download a dataset
  download: async (datasetId: string): Promise<{ path: string }> => {
    return apiClient.post(`/datasets/download`, { dataset_id: datasetId });
  },

  // List all local datasets
  list: async (): Promise<Dataset[]> => {
    return apiClient.get('/datasets/local');
  },

  // Get dataset details
  get: async (datasetId: string): Promise<Dataset> => {
    return apiClient.get(`/datasets/${datasetId}`);
  },

  // Upload a local file as a new dataset
  upload: async (file: File): Promise<Dataset> => {
    const formData = new FormData();
    formData.append('file', file);

    return apiClient.post('/datasets/upload', formData, {
      headers: {
        'Content-Type': 'multipart/form-data',
      },
    });
  },

  // Get dataset versions
  getVersions: async (datasetId: string): Promise<DatasetVersion[]> => {
    return apiClient.get(`/datasets/${datasetId}/versions`);
  },

  // Delete a local dataset
  delete: async (datasetId: string): Promise<{ success: boolean }> => {
    return apiClient.delete(`/datasets/${datasetId}`);
  },
};
