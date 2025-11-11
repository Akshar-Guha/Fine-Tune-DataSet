import React, { useState } from 'react';
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import { datasetsAPI } from '../api/datasets';
import { Dataset, DatasetSearchResult } from '../types/datasets';

const DatasetRegistry: React.FC = () => {
  const queryClient = useQueryClient();
  const [searchQuery, setSearchQuery] = useState('');
  const [searchTask, setSearchTask] = useState('text-generation');
  const [selectedFile, setSelectedFile] = useState<File | null>(null);

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
  const { data: localDatasets, isLoading: isLoadingLocal } = useQuery<Dataset[], Error>({
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

  // Mutation for uploading a dataset
  const uploadMutation = useMutation<any, Error, File>({
    mutationFn: datasetsAPI.upload,
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['localDatasets'] });
      setSelectedFile(null);
    },
  });

  const handleSearch = (e: React.FormEvent) => {
    e.preventDefault();
    searchDatasets();
  };

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files) {
      setSelectedFile(e.target.files[0]);
    }
  };

  const handleUpload = () => {
    if (selectedFile) {
      uploadMutation.mutate(selectedFile);
    }
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
        <div className="flex gap-2">
          <input type="file" onChange={handleFileChange} className="border p-2 rounded-md" />
          <button
            onClick={handleUpload}
            className="bg-purple-500 text-white p-2 rounded-md"
            disabled={!selectedFile || uploadMutation.isPending}
          >
            {uploadMutation.isPending ? 'Uploading...' : 'Upload'}
          </button>
        </div>
      </div>

      {/* Local Datasets Section */}
      <div>
        <h2 className="text-xl font-semibold mb-2">Local Datasets</h2>
        {isLoadingLocal ? (
          <p>Loading local datasets...</p>
        ) : (
          <ul className="list-disc pl-5">
            {localDatasets?.map((dataset) => (
              <li key={dataset.dataset_id} className="mb-2">
                {dataset.dataset_id} ({dataset.size_mb.toFixed(2)} MB)
              </li>
            ))}
          </ul>
        )}
      </div>
    </div>
  );
};

export default DatasetRegistry;
