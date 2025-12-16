import { useMutation } from '@tanstack/react-query';
import { AxiosError } from 'axios';
import {
  ColabConfigRequest,
  ColabValidationResponse,
  ColabNotebookResponse,
} from '../types/colab';
import { colabAPI } from '../api/colab';

const extractErrorMessage = (error: AxiosError | Error): string => {
  if ('isAxiosError' in error && error.isAxiosError) {
    const axiosError = error as AxiosError<{ detail?: string } | string>;
    if (axiosError.response?.data) {
      if (typeof axiosError.response.data === 'string') {
        return axiosError.response.data;
      }
      if (typeof axiosError.response.data.detail === 'string') {
        return axiosError.response.data.detail;
      }
    }
  }
  return error.message;
};

export const useColabValidation = () =>
  useMutation<ColabValidationResponse, Error, ColabConfigRequest>({
    mutationFn: async (payload) => colabAPI.validateConfig(payload),
    onError: (error) => {
      console.error('Colab validation failed:', extractErrorMessage(error as AxiosError | Error));
    },
  });

export const useColabNotebookGeneration = () =>
  useMutation<ColabNotebookResponse, Error, ColabConfigRequest>({
    mutationFn: async (payload) => colabAPI.generateNotebook(payload),
    onError: (error) => {
      console.error('Colab notebook generation failed:', extractErrorMessage(error as AxiosError | Error));
    },
  });
