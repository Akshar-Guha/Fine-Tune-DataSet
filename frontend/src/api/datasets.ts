import { apiClient } from './client';
import type {
  Dataset,
  DatasetSearchResult,
  DatasetUploadResponse,
  DatasetUploadOptions,
  DatasetVersion,
  LocalDatasetSummary,
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

  // List local datasets
  list: async (): Promise<LocalDatasetSummary[]> => {
    return apiClient.get('/datasets/local');
  },

  // Download a dataset
  download: async (datasetId: string): Promise<{ path: string }> => {
    return apiClient.post(`/datasets/download`, { dataset_id: datasetId });
  },

  // Process an existing local dataset through the pipeline
  process: async (
    datasetId: string,
    options?: DatasetUploadOptions
  ): Promise<DatasetUploadResponse> => {
    return apiClient.post(`/datasets/process/${datasetId}`, {
      dataset_id: datasetId,
      text_column: options?.text_column,
      label_column: options?.label_column,
      enable_auto_labeling: options?.enable_auto_labeling,
      auto_labeling_rules: options?.auto_labeling_rules,
      quality_score_threshold: options?.quality_score_threshold,
      strip_text: options?.strip_text,
      drop_missing_text: options?.drop_missing_text,
      drop_missing_label: options?.drop_missing_label,
      lowercase_labels: options?.lowercase_labels,
    });
  },

  // Get dataset details
  get: async (datasetId: string): Promise<Dataset> => {
    return apiClient.get(`/datasets/${datasetId}`);
  },

  // Upload a local file as a new dataset with pipeline configuration
  upload: async (
    file: File,
    options?: DatasetUploadOptions
  ): Promise<DatasetUploadResponse> => {
    const formData = new FormData();
    formData.append('file', file);

    // Add pipeline configuration options
    if (options?.dataset_name) {
      formData.append('dataset_name', options.dataset_name);
    }
    if (options?.text_column) {
      formData.append('text_column', options.text_column);
    }
    if (options?.label_column) {
      formData.append('label_column', options.label_column);
    }

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

  // Get validation results for a dataset
  getValidationResults: async (datasetId: string) => {
    return apiClient.get(`/datasets/${datasetId}/validation/latest`);
  },

  // Get validation history for a dataset
  getValidationHistory: async (datasetId: string, limit?: number) => {
    return apiClient.get(`/datasets/${datasetId}/validation/history`, {
      params: { limit }
    });
  },

  // List recent validation results
  listValidationResults: async (limit?: number) => {
    return apiClient.get('/datasets/validation/recent', {
      params: { limit }
    });
  },

  // Fetch a stored quality report by path
  getQualityReport: async (reportPath: string) => {
    return apiClient.get('/datasets/quality-report', {
      params: { report_path: reportPath },
    });
  },
};
