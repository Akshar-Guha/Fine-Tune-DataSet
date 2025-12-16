import { apiClient } from './client';
import {
  ColabConfigRequest,
  ColabValidationResponse,
  ColabNotebookResponse,
} from '../types/colab';

export const colabAPI = {
  validateConfig: async (
    payload: ColabConfigRequest
  ): Promise<ColabValidationResponse> => {
    return apiClient.post('/colab/validate-config', payload);
  },

  generateNotebook: async (
    payload: ColabConfigRequest
  ): Promise<ColabNotebookResponse> => {
    return apiClient.post('/colab/generate-notebook', payload);
  },
};

export const resolveDownloadUrl = (downloadPath: string): string => {
  if (downloadPath.startsWith('http')) {
    return downloadPath;
  }

  const apiBase = import.meta.env.VITE_API_URL as string | undefined;
  if (!apiBase) {
    return downloadPath;
  }

  const trimmedBase = apiBase.replace(/\/?api\/v1\/?$/i, '');
  return `${trimmedBase}${downloadPath}`;
};
