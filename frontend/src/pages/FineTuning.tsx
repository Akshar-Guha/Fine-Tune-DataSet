import React, { useState } from 'react';
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import { modelsAPI } from '../api/models';
import { datasetsAPI } from '../api/datasets';
import { jobsAPI } from '../api/jobs';
import { Model } from '../types/models';
import { Dataset } from '../types/datasets';
import { FineTuningJobCreateRequest } from '../types/jobs';

const FineTuning: React.FC = () => {
  const queryClient = useQueryClient();
  const [selectedModel, setSelectedModel] = useState<string>('');
  const [selectedDataset, setSelectedDataset] = useState<string>('');
  const [jobName, setJobName] = useState('');
  const [loraRank, setLoraRank] = useState(8);
  const [loraAlpha, setLoraAlpha] = useState(16);
  const [loraDropout, setLoraDropout] = useState(0.1);
  const [numEpochs, setNumEpochs] = useState(3);
  const [learningRate, setLearningRate] = useState(2e-4);
  const [seqLength, setSeqLength] = useState(512);

  // Fetch local models and datasets
  const { data: models } = useQuery<Model[], Error>({
    queryKey: ['localModels'],
    queryFn: modelsAPI.list,
  });
  const { data: datasets } = useQuery<Dataset[], Error>({
    queryKey: ['localDatasets'],
    queryFn: datasetsAPI.list,
  });

  // Mutation for submitting a fine-tuning job
  const submitJobMutation = useMutation<any, Error, FineTuningJobCreateRequest>({
    mutationFn: jobsAPI.submitFineTuning,
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['jobs'] });
      // Optionally, navigate to jobs page or show success message
    },
  });

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (!selectedModel || !selectedDataset || !jobName) {
      alert('Please fill in all fields');
      return;
    }

    const jobRequest: FineTuningJobCreateRequest = {
      name: jobName,
      base_model: selectedModel,
      dataset_id: selectedDataset,
      config: {
        lora_rank: loraRank,
        lora_alpha: loraAlpha,
        lora_dropout: loraDropout,
        num_epochs: numEpochs,
        learning_rate: learningRate,
        max_seq_length: seqLength,
      },
    };

    submitJobMutation.mutate(jobRequest);
  };

  return (
    <div className="p-6">
      <h1 className="text-2xl font-bold mb-4">Fine-Tune a Model</h1>

      <form onSubmit={handleSubmit} className="space-y-4">
        <div>
          <label className="block font-semibold">Job Name</label>
          <input
            type="text"
            value={jobName}
            onChange={(e) => setJobName(e.target.value)}
            className="border p-2 rounded-md w-full"
            required
          />
        </div>

        <div>
          <label className="block font-semibold">Select a Base Model</label>
          <select
            value={selectedModel}
            onChange={(e) => setSelectedModel(e.target.value)}
            className="border p-2 rounded-md w-full"
            required
          >
            <option value="">-- Select Model --</option>
            {models?.map((model) => (
              <option key={model.id} value={model.name}>
                {model.name}
              </option>
            ))}
          </select>
        </div>

        <div>
          <label className="block font-semibold">Select a Dataset</label>
          <select
            value={selectedDataset}
            onChange={(e) => setSelectedDataset(e.target.value)}
            className="border p-2 rounded-md w-full"
            required
          >
            <option value="">-- Select Dataset --</option>
            {datasets?.map((dataset) => (
              <option key={dataset.dataset_id} value={dataset.dataset_id}>
                {dataset.dataset_id}
              </option>
            ))}
          </select>
        </div>

        <div>
          <label className="block font-semibold">LoRA Rank</label>
          <input
            type="number"
            value={loraRank}
            onChange={(e) => setLoraRank(parseInt(e.target.value))}
            className="border p-2 rounded-md w-full"
          />
        </div>

        <div>
          <label className="block font-semibold">Number of Epochs</label>
          <input
            type="number"
            value={numEpochs}
            onChange={(e) => setNumEpochs(parseInt(e.target.value))}
            className="border p-2 rounded-md w-full"
          />
        </div>

        <div>
          <label className="block font-semibold">LoRA Alpha</label>
          <input
            type="number"
            value={loraAlpha}
            onChange={(e) => setLoraAlpha(parseInt(e.target.value))}
            className="border p-2 rounded-md w-full"
          />
        </div>

        <div>
          <label className="block font-semibold">LoRA Dropout</label>
          <input
            type="number"
            step="0.01"
            value={loraDropout}
            onChange={(e) => setLoraDropout(parseFloat(e.target.value))}
            className="border p-2 rounded-md w-full"
          />
        </div>

        <div>
          <label className="block font-semibold">Learning Rate</label>
          <input
            type="number"
            step="0.00001"
            value={learningRate}
            onChange={(e) => setLearningRate(parseFloat(e.target.value))}
            className="border p-2 rounded-md w-full"
          />
        </div>

        <div>
          <label className="block font-semibold">Sequence Length</label>
          <input
            type="number"
            value={seqLength}
            onChange={(e) => setSeqLength(parseInt(e.target.value))}
            className="border p-2 rounded-md w-full"
          />
        </div>

        <button
          type="submit"
          className="bg-blue-500 text-white p-2 rounded-md"
          disabled={submitJobMutation.isPending}
        >
          {submitJobMutation.isPending ? 'Submitting...' : 'Submit Fine-Tuning Job'}
        </button>
      </form>
    </div>
  );
};

export default FineTuning;
