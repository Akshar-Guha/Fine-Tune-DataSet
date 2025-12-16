import React, { useEffect, useRef, useState } from 'react';
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import { datasetsAPI } from '../api/datasets';
import { Dataset, DatasetSearchResult, DatasetUploadResponse, DatasetUploadOptions, LocalDatasetSummary } from '../types/datasets';
import DatasetPipelineConfig from '../components/datasets/DatasetPipelineConfig';
import DatasetPipelineResults from '../components/datasets/DatasetPipelineResults';
import PipelineStatus from '../components/datasets/PipelineStatus';

type PipelineStage = 'idle' | 'uploading' | 'processing' | 'validating' | 'completed' | 'failed';

type PipelineState = {
  status: PipelineStage;
  progress: number;
  message: string;
  error?: string;
};

type ProgressStep = {
  delay: number;
  status?: PipelineStage;
  progress?: number;
  message?: string;
};

const DatasetRegistry: React.FC = () => {
  const queryClient = useQueryClient();
  const [searchQuery, setSearchQuery] = useState('');
  const [searchTask, setSearchTask] = useState('text-generation');
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [pipelineConfig, setPipelineConfig] = useState<DatasetUploadOptions>({});
  const [uploadResult, setUploadResult] = useState<DatasetUploadResponse | null>(null);
  const [showConfig, setShowConfig] = useState(false);
  const [selectedLocalDataset, setSelectedLocalDataset] = useState<LocalDatasetSummary | null>(null);
  const [showProcessConfig, setShowProcessConfig] = useState(false);
  const [pipelineStatus, setPipelineStatus] = useState<PipelineState>({ status: 'idle', progress: 0, message: '' });
  const progressTimeoutsRef = useRef<ReturnType<typeof setTimeout>[]>([]);
  const activeRunRef = useRef<number | null>(null);
  const runCounterRef = useRef(0);

  const clearProgressTimeouts = () => {
    progressTimeoutsRef.current.forEach((timeoutId) => clearTimeout(timeoutId));
    progressTimeoutsRef.current = [];
  };

  useEffect(() => {
    return () => {
      clearProgressTimeouts();
      activeRunRef.current = null;
    };
  }, []);

  const beginPipelineRun = (initial: { status: PipelineStage; progress: number; message: string }) => {
    clearProgressTimeouts();
    const runId = runCounterRef.current + 1;
    runCounterRef.current = runId;
    activeRunRef.current = runId;
    setPipelineStatus({ ...initial, error: undefined });
    return runId;
  };

  const simulateProgressUpdates = (runId: number, steps: ProgressStep[]) => {
    steps.forEach((step) => {
      const timeoutId = window.setTimeout(() => {
        if (activeRunRef.current !== runId) {
          return;
        }
        setPipelineStatus((prev) => ({
          ...prev,
          status: step.status ?? prev.status,
          progress: step.progress ?? prev.progress,
          message: step.message ?? prev.message,
          error: undefined,
        }));
      }, step.delay);
      progressTimeoutsRef.current.push(timeoutId);
    });
  };

  const completePipelineRun = (runId: number, message: string) => {
    if (activeRunRef.current !== runId) {
      return;
    }
    clearProgressTimeouts();
    activeRunRef.current = null;
    setPipelineStatus({ status: 'completed', progress: 100, message, error: undefined });
  };

  const failPipelineRun = (runId: number, message: string, errorMessage: string) => {
    if (activeRunRef.current !== runId) {
      return;
    }
    clearProgressTimeouts();
    activeRunRef.current = null;
    setPipelineStatus({ status: 'failed', progress: 0, message, error: errorMessage });
  };

  // Query for searching datasets on HuggingFace Hub
  const {
    data: searchResults,
    isLoading: isSearching,
    refetch: searchDatasets,
  } = useQuery<DatasetSearchResult[], Error>({
    queryKey: ['searchDatasets', searchQuery, searchTask],
    queryFn: () => datasetsAPI.search({ query: searchQuery, task: searchTask, limit: 20 }),
    enabled: false,
  });

  // Query for listing local datasets
  const { data: localDatasets, isLoading: isLoadingLocal } = useQuery<LocalDatasetSummary[], Error>({
    queryKey: ['localDatasets'],
    queryFn: datasetsAPI.list,
  });

  // Mutation for downloading a dataset
  const downloadMutation = useMutation<any, Error, string>({
    mutationFn: datasetsAPI.download,
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['localDatasets'] });
    },
  });

  // Mutation for processing a local dataset
  const processMutation = useMutation<DatasetUploadResponse, Error, { datasetId: string; config: DatasetUploadOptions }>({
    mutationFn: async ({ datasetId, config }) => {
      const runId = beginPipelineRun({ status: 'processing', progress: 10, message: 'Starting dataset processing...' });
      simulateProgressUpdates(runId, [
        { delay: 500, progress: 30, message: 'Loading dataset...' },
        { delay: 1000, status: 'processing', progress: 50, message: 'Applying transformations...' },
        { delay: 2000, status: 'validating', progress: 80, message: 'Running quality validation...' },
      ]);

      try {
        const result = await datasetsAPI.process(datasetId, config);
        completePipelineRun(runId, 'Dataset processing completed!');
        return result;
      } catch (error) {
        const errorMessage = error instanceof Error ? error.message : 'An error occurred during processing';
        failPipelineRun(runId, 'Processing failed', errorMessage);
        throw error;
      }
    },
    onSuccess: (result) => {
      setUploadResult(result);
      queryClient.invalidateQueries({ queryKey: ['localDatasets'] });
      setSelectedLocalDataset(null);
      setShowProcessConfig(false);
    },
    onError: () => {
      setUploadResult(null);
    },
  });

  // Mutation for uploading a dataset with pipeline configuration
  const uploadMutation = useMutation<DatasetUploadResponse, Error, { file: File; config: DatasetUploadOptions }>({
    mutationFn: async ({ file, config }) => {
      const runId = beginPipelineRun({ status: 'uploading', progress: 10, message: 'Uploading file to server...' });
      simulateProgressUpdates(runId, [
        { delay: 500, progress: 30, message: 'File uploaded, starting processing...' },
        { delay: 1000, status: 'processing', progress: 50, message: 'Processing dataset columns and cleaning data...' },
        { delay: 2000, status: 'validating', progress: 80, message: 'Running quality validation checks...' },
      ]);

      try {
        const result = await datasetsAPI.upload(file, config);
        completePipelineRun(runId, 'Pipeline completed successfully!');
        return result;
      } catch (error) {
        const errorMessage = error instanceof Error ? error.message : 'An error occurred during processing';
        failPipelineRun(runId, 'Pipeline failed', errorMessage);
        throw error;
      }
    },
    onSuccess: (result) => {
      setUploadResult(result);
      queryClient.invalidateQueries({ queryKey: ['localDatasets'] });
      setSelectedFile(null);
      setShowConfig(false);
    },
    onError: () => {
      setUploadResult(null);
    },
  });

  const handleSearch = (e: React.FormEvent) => {
    e.preventDefault();
    searchDatasets();
  };

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files) {
      setSelectedFile(e.target.files[0]);
      setShowConfig(true);
      setUploadResult(null);
      setPipelineStatus({ status: 'idle', progress: 0, message: '' });
    }
  };

  const handleUpload = () => {
    if (selectedFile) {
      uploadMutation.mutate({ file: selectedFile, config: pipelineConfig });
    }
  };

  const handleConfigChange = (config: DatasetUploadOptions) => {
    setPipelineConfig(config);
  };

  const handleProcess = () => {
    if (selectedLocalDataset) {
      processMutation.mutate({ datasetId: selectedLocalDataset.dataset_id, config: pipelineConfig });
    }
  };

  const resetUpload = () => {
    clearProgressTimeouts();
    activeRunRef.current = null;
    setSelectedFile(null);
    setUploadResult(null);
    setShowConfig(false);
    setPipelineConfig({});
    setPipelineStatus({ status: 'idle', progress: 0, message: '', error: undefined });
  };

  const resetProcess = () => {
    clearProgressTimeouts();
    activeRunRef.current = null;
    setSelectedLocalDataset(null);
    setUploadResult(null);
    setShowProcessConfig(false);
    setPipelineConfig({});
    setPipelineStatus({ status: 'idle', progress: 0, message: '', error: undefined });
  };

  const handleDatasetSelection = (dataset: LocalDatasetSummary) => {
    setSelectedLocalDataset(dataset);
    setShowProcessConfig(true);
    setUploadResult(null);
    setPipelineStatus({ status: 'idle', progress: 0, message: '' });
  };

  return (
    <div className="p-6">
      <h1 className="text-2xl font-bold mb-4">Dataset Registry</h1>

      {/* Search Section */}
      <div className="mb-8 p-4 border rounded-lg">
        <h2 className="text-xl font-semibold mb-2">Search HuggingFace Hub</h2>
        <form onSubmit={handleSearch} className="flex gap-2">
          <input
            type="text"
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
            placeholder="Search for datasets..."
            className="border p-2 rounded-md flex-grow"
          />
          <select
            value={searchTask}
            onChange={(e) => setSearchTask(e.target.value)}
            className="border p-2 rounded-md"
          >
            <option value="text-generation">Text Generation</option>
            <option value="text-classification">Text Classification</option>
          </select>
          <button type="submit" className="bg-blue-500 text-white p-2 rounded-md">
            {isSearching ? 'Searching...' : 'Search'}
          </button>
        </form>

        {searchResults && (
          <div className="mt-4">
            <h3 className="font-semibold">Search Results:</h3>
            <ul className="list-disc pl-5 mt-2">
              {searchResults.map((dataset) => (
                <li key={dataset.dataset_id} className="mb-2">
                  {dataset.dataset_id} ({dataset.downloads} downloads)
                  <button
                    onClick={() => downloadMutation.mutate(dataset.dataset_id)}
                    className="ml-4 bg-green-500 text-white px-2 py-1 text-xs rounded"
                    disabled={downloadMutation.isPending}
                  >
                    {downloadMutation.isPending ? 'Downloading...' : 'Download'}
                  </button>
                </li>
              ))}
            </ul>
          </div>
        )}
      </div>

      {/* Upload Section */}
      <div className="mb-8 p-4 border rounded-lg">
        <h2 className="text-xl font-semibold mb-2">Upload Local Dataset</h2>
        <div className="space-y-4">
          <div className="flex gap-2">
            <input
              type="file"
              onChange={handleFileChange}
              className="border p-2 rounded-md flex-grow"
              accept=".csv,.json,.jsonl,.parquet,.txt"
            />
            {selectedFile && (
              <button
                onClick={resetUpload}
                className="bg-gray-500 text-white px-4 py-2 rounded-md"
              >
                Clear
              </button>
            )}
          </div>

          {selectedFile && (
            <div className="text-sm text-gray-600">
              Selected: {selectedFile.name} ({(selectedFile.size / 1024 / 1024).toFixed(2)} MB)
            </div>
          )}

          {/* Pipeline Status */}
          <PipelineStatus
            status={pipelineStatus.status}
            progress={pipelineStatus.progress}
            message={pipelineStatus.message}
            error={pipelineStatus.error}
          />

          {showConfig && selectedFile && pipelineStatus.status === 'idle' && (
            <DatasetPipelineConfig
              file={selectedFile}
              onConfigChange={handleConfigChange}
              initialConfig={pipelineConfig}
            />
          )}

          {showConfig && selectedFile && pipelineStatus.status === 'idle' && (
            <div className="flex gap-2">
              <button
                onClick={handleUpload}
                className="bg-purple-500 text-white p-2 rounded-md"
                disabled={uploadMutation.isPending}
              >
                {uploadMutation.isPending ? 'Processing...' : 'Upload & Process'}
              </button>
              <button
                onClick={() => setShowConfig(false)}
                className="bg-gray-500 text-white p-2 rounded-md"
              >
                Cancel
              </button>
            </div>
          )}

          {/* Upload Results */}
          {uploadResult && pipelineStatus.status === 'completed' && (
            <DatasetPipelineResults result={uploadResult} />
          )}

          {/* Upload Error */}
          {pipelineStatus.status === 'failed' && (
            <div className="p-4 border border-red-300 rounded-lg bg-red-50">
              <h4 className="text-red-800 font-semibold">Upload Failed</h4>
              <p className="text-red-600 text-sm mt-1">
                {pipelineStatus.error || 'An error occurred during upload'}
              </p>
              <button
                onClick={resetUpload}
                className="mt-2 bg-red-600 text-white px-3 py-1 text-sm rounded hover:bg-red-700"
              >
                Try Again
              </button>
            </div>
          )}
        </div>
      </div>

      {/* Local Datasets Section */}
      <div>
        <h2 className="text-xl font-semibold mb-2">Local Datasets</h2>
        {isLoadingLocal ? (
          <p>Loading local datasets...</p>
        ) : (
          <div className="space-y-2">
            {localDatasets?.map((dataset) => (
              <div key={dataset.dataset_id} className="flex items-center gap-2 p-2 border rounded">
                <input
                  type="radio"
                  id={dataset.dataset_id}
                  name="selectedDataset"
                  checked={selectedLocalDataset?.dataset_id === dataset.dataset_id}
                  onChange={() => handleDatasetSelection(dataset)}
                  className="mr-2"
                />
                <label htmlFor={dataset.dataset_id} className="flex-1 cursor-pointer">
                  <div className="font-medium">{dataset.dataset_id}</div>
                  <div className="text-xs text-gray-500 mt-1">
                    {dataset.num_rows} rows × {dataset.num_columns} columns • {(dataset.size_mb).toFixed(2)} MB
                  </div>
                </label>
              </div>
            ))}

            {selectedLocalDataset && (
              <div className="mt-4 p-4 border rounded-lg bg-blue-50">
                <h3 className="font-semibold">Selected Dataset: {selectedLocalDataset.dataset_id}</h3>
                <p className="text-sm text-gray-600 mt-1">
                  {selectedLocalDataset.num_rows} rows, {selectedLocalDataset.num_columns} columns
                </p>
              </div>
            )}
          </div>
        )}

        {/* Process Configuration */}
        {showProcessConfig && selectedLocalDataset && pipelineStatus.status === 'idle' && (
          <div className="mt-4">
            <DatasetPipelineConfig
              onConfigChange={handleConfigChange}
              initialConfig={pipelineConfig}
            />
            <div className="flex gap-2 mt-4">
              <button
                onClick={handleProcess}
                className="bg-green-500 text-white p-2 rounded-md"
                disabled={processMutation.isPending}
              >
                {processMutation.isPending ? 'Processing...' : 'Process Dataset'}
              </button>
              <button
                onClick={resetProcess}
                className="bg-gray-500 text-white p-2 rounded-md"
              >
                Cancel
              </button>
            </div>
          </div>
        )}

        {/* Pipeline Status for Processing */}
        {pipelineStatus.status !== 'idle' && showProcessConfig && (
          <div className="mt-4">
            <PipelineStatus
              status={pipelineStatus.status}
              progress={pipelineStatus.progress}
              message={pipelineStatus.message}
              error={pipelineStatus.error}
            />
          </div>
        )}

        {/* Process Results */}
        {uploadResult && showProcessConfig && pipelineStatus.status === 'completed' && (
          <div className="mt-4">
            <DatasetPipelineResults result={uploadResult} />
          </div>
        )}

        {/* Process Error */}
        {pipelineStatus.status === 'failed' && showProcessConfig && (
          <div className="mt-4 p-4 border border-red-300 rounded-lg bg-red-50">
            <h4 className="text-red-800 font-semibold">Processing Failed</h4>
            <p className="text-red-600 text-sm mt-1">
              {pipelineStatus.error || 'An error occurred during processing'}
            </p>
            <button
              onClick={resetProcess}
              className="mt-2 bg-red-600 text-white px-3 py-1 text-sm rounded hover:bg-red-700"
            >
              Try Again
            </button>
          </div>
        )}
      </div>
    </div>
  );
};

export default DatasetRegistry;
