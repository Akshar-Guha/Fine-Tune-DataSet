import React, { useState } from 'react';
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import { modelsAPI } from '../api/models';
import { Model, ModelSearchResult } from '../types/models';

const ModelRegistry: React.FC = () => {
  const queryClient = useQueryClient();
  const [searchQuery, setSearchQuery] = useState('');
  const [searchTask, setSearchTask] = useState('text-generation');

  // Query for searching models on HuggingFace Hub
  const {
    data: searchResults,
    isLoading: isSearching,
    refetch: searchModels,
  } = useQuery<ModelSearchResult[], Error>({
    queryKey: ['searchModels', searchQuery, searchTask],
    queryFn: () => modelsAPI.search({ query: searchQuery, task: searchTask, limit: 20 }),
    enabled: false, // Only run when searchModels is called
  });

  // Query for listing local models
  const { data: localModels, isLoading: isLoadingLocal } = useQuery<Model[], Error>({
    queryKey: ['localModels'],
    queryFn: modelsAPI.list,
  });

  // Mutation for downloading a model
  const downloadMutation = useMutation<any, Error, string>({
    mutationFn: modelsAPI.download,
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['localModels'] });
    },
  });

  const handleSearch = (e: React.FormEvent) => {
    e.preventDefault();
    searchModels();
  };

  return (
    <div className="p-6">
      <h1 className="text-2xl font-bold mb-4">Model Registry</h1>

      {/* Search Section */}
      <div className="mb-8 p-4 border rounded-lg">
        <h2 className="text-xl font-semibold mb-2">Search HuggingFace Hub</h2>
        <form onSubmit={handleSearch} className="flex gap-2">
          <input
            type="text"
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
            placeholder="Search for models..."
            className="border p-2 rounded-md flex-grow"
          />
          <select
            value={searchTask}
            onChange={(e) => setSearchTask(e.target.value)}
            className="border p-2 rounded-md"
          >
            <option value="text-generation">Text Generation</option>
            <option value="text-classification">Text Classification</option>
            <option value="token-classification">Token Classification</option>
          </select>
          <button type="submit" className="bg-blue-500 text-white p-2 rounded-md">
            {isSearching ? 'Searching...' : 'Search'}
          </button>
        </form>

        {searchResults && (
          <div className="mt-4">
            <h3 className="font-semibold">Search Results:</h3>
            <ul className="list-disc pl-5 mt-2">
              {searchResults.map((model) => (
                <li key={model.model_id} className="mb-2">
                  {model.model_id} ({model.downloads} downloads)
                  <button
                    onClick={() => downloadMutation.mutate(model.model_id)}
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

      {/* Local Models Section */}
      <div>
        <h2 className="text-xl font-semibold mb-2">Local Models</h2>
        {isLoadingLocal ? (
          <p>Loading local models...</p>
        ) : (
          <ul className="list-disc pl-5">
            {localModels?.map((model) => (
              <li key={model.id} className="mb-2">
                {model.name} ({model.size_gb?.toFixed(2)} GB)
              </li>
            ))}
          </ul>
        )}
      </div>
    </div>
  );
};

export default ModelRegistry;
