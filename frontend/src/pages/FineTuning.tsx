import React, { useMemo, useState } from 'react';
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import Card from '../components/ui/Card';
import Button from '../components/ui/Button';
import { modelsAPI } from '../api/models';
import { datasetsAPI } from '../api/datasets';
import { jobsAPI } from '../api/jobs';
import { Model } from '../types/models';
import { Dataset } from '../types/datasets';
import { FineTuningJobCreateRequest } from '../types/jobs';
import MetricComparisonChart from '../components/charts/MetricComparisonChart';
import { useColabValidation, useColabNotebookGeneration } from '../hooks/useColab';
import {
  ColabConfigRequest,
  ColabValidationResponse,
  ColabValidationSuccess,
} from '../types/colab';
import { resolveDownloadUrl } from '../api/colab';

const isValidationSuccess = (
  data: ColabValidationResponse | undefined
): data is ColabValidationSuccess => Boolean(data && data.valid);

const FineTuning: React.FC = () => {
  const queryClient = useQueryClient();
  const [selectedModel, setSelectedModel] = useState('');
  const [selectedDataset, setSelectedDataset] = useState('');
  const [jobName, setJobName] = useState('');
  const [experimentName, setExperimentName] = useState('');
  const [loraRank, setLoraRank] = useState(8);
  const [loraAlpha, setLoraAlpha] = useState(16);
  const [loraDropout, setLoraDropout] = useState(0.1);
  const [numEpochs, setNumEpochs] = useState(3);
  const [batchSize, setBatchSize] = useState(2);
  const [learningRate, setLearningRate] = useState(2e-4);
  const [seqLength, setSeqLength] = useState(512);

  const { data: models } = useQuery<Model[], Error>({
    queryKey: ['localModels'],
    queryFn: modelsAPI.list,
  });

  const { data: datasets } = useQuery<Dataset[], Error>({
    queryKey: ['localDatasets'],
    queryFn: datasetsAPI.list,
  });

  const submitJobMutation = useMutation<any, Error, FineTuningJobCreateRequest>({
    mutationFn: jobsAPI.submitFineTuning,
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['jobs'] });
    },
  });

  const colabValidation = useColabValidation();
  const colabNotebook = useColabNotebookGeneration();

  const colabConfig = useMemo<ColabConfigRequest>(
    () => ({
      base_model: selectedModel,
      dataset_id: selectedDataset,
      experiment_name: experimentName || jobName || 'fine-tuning-experiment',
      lora_rank: loraRank,
      lora_alpha: loraAlpha,
      num_epochs: numEpochs,
      batch_size: batchSize,
      learning_rate: learningRate,
      max_seq_length: seqLength,
    }),
    [
      batchSize,
      experimentName,
      jobName,
      loraAlpha,
      loraRank,
      learningRate,
      numEpochs,
      selectedDataset,
      selectedModel,
      seqLength,
    ]
  );

  const successData: ColabValidationSuccess | undefined =
    colabNotebook.data ?? (isValidationSuccess(colabValidation.data) ? colabValidation.data : undefined);

  const validationIssues =
    colabValidation.data && !colabValidation.data.valid ? colabValidation.data.issues : [];

  const notebookDownloadUrl = colabNotebook.data
    ? resolveDownloadUrl(colabNotebook.data.download_url)
    : undefined;

  const runtimeRecommendation = successData?.recommendation;
  const preferredRuntime = successData?.preferred_runtime;
  const localSupported = successData?.local_supported;

  const handleSubmit = (e: React.FormEvent<HTMLFormElement>) => {
    e.preventDefault();
    if (!selectedModel || !selectedDataset || !jobName) {
      alert('Please complete the required fields.');
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
        batch_size: batchSize,
        max_seq_length: seqLength,
      },
    };

    submitJobMutation.mutate(jobRequest);
  };

  const handleValidate = () => {
    colabValidation.mutate(colabConfig, {
      onSuccess: () => colabNotebook.reset(),
    });
  };

  const handleGenerateNotebook = () => {
    colabNotebook.mutate(colabConfig);
  };

  return (
    <div className="space-y-6 p-6">
      <h1 className="text-2xl font-bold text-gray-900">Fine-Tune a Model</h1>

      <Card title="Training Configuration" className="space-y-6">
        <form onSubmit={handleSubmit} className="space-y-6">
          <div className="grid grid-cols-1 gap-4 md:grid-cols-2">
            <div className="flex flex-col space-y-2">
              <label className="font-semibold text-gray-700">Job Name</label>
              <input
                type="text"
                value={jobName}
                onChange={(e) => setJobName(e.target.value)}
                className="w-full rounded-md border border-gray-300 p-2"
                required
              />
            </div>

            <div className="flex flex-col space-y-2">
              <label className="font-semibold text-gray-700">Experiment Name (Colab)</label>
              <input
                type="text"
                value={experimentName}
                onChange={(e) => setExperimentName(e.target.value)}
                placeholder="Defaults to job name"
                className="w-full rounded-md border border-gray-300 p-2"
              />
            </div>

            <div className="flex flex-col space-y-2">
              <label className="font-semibold text-gray-700">Base Model</label>
              <select
                value={selectedModel}
                onChange={(e) => setSelectedModel(e.target.value)}
                className="w-full rounded-md border border-gray-300 p-2"
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

            <div className="flex flex-col space-y-2">
              <label className="font-semibold text-gray-700">Dataset</label>
              <select
                value={selectedDataset}
                onChange={(e) => setSelectedDataset(e.target.value)}
                className="w-full rounded-md border border-gray-300 p-2"
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

            <div className="flex flex-col space-y-2">
              <label className="font-semibold text-gray-700">LoRA Rank</label>
              <input
                type="number"
                min={1}
                value={loraRank}
                onChange={(e) => {
                  const value = parseInt(e.target.value, 10);
                  setLoraRank(Number.isNaN(value) ? 1 : value);
                }}
                className="w-full rounded-md border border-gray-300 p-2"
              />
            </div>

            <div className="flex flex-col space-y-2">
              <label className="font-semibold text-gray-700">LoRA Alpha</label>
              <input
                type="number"
                min={1}
                value={loraAlpha}
                onChange={(e) => {
                  const value = parseInt(e.target.value, 10);
                  setLoraAlpha(Number.isNaN(value) ? 1 : value);
                }}
                className="w-full rounded-md border border-gray-300 p-2"
              />
            </div>

            <div className="flex flex-col space-y-2">
              <label className="font-semibold text-gray-700">LoRA Dropout</label>
              <input
                type="number"
                step="0.01"
                min={0}
                value={loraDropout}
                onChange={(e) => {
                  const value = parseFloat(e.target.value);
                  setLoraDropout(Number.isNaN(value) ? 0 : value);
                }}
                className="w-full rounded-md border border-gray-300 p-2"
              />
            </div>

            <div className="flex flex-col space-y-2">
              <label className="font-semibold text-gray-700">Epochs</label>
              <input
                type="number"
                min={1}
                value={numEpochs}
                onChange={(e) => {
                  const value = parseInt(e.target.value, 10);
                  setNumEpochs(Number.isNaN(value) ? 1 : value);
                }}
                className="w-full rounded-md border border-gray-300 p-2"
              />
            </div>

            <div className="flex flex-col space-y-2">
              <label className="font-semibold text-gray-700">Batch Size</label>
              <input
                type="number"
                min={1}
                value={batchSize}
                onChange={(e) => {
                  const value = parseInt(e.target.value, 10);
                  setBatchSize(Number.isNaN(value) ? 1 : value);
                }}
                className="w-full rounded-md border border-gray-300 p-2"
              />
            </div>

            <div className="flex flex-col space-y-2">
              <label className="font-semibold text-gray-700">Learning Rate</label>
              <input
                type="number"
                step="0.00001"
                min={0}
                value={learningRate}
                onChange={(e) => {
                  const value = parseFloat(e.target.value);
                  setLearningRate(Number.isNaN(value) ? 0 : value);
                }}
                className="w-full rounded-md border border-gray-300 p-2"
              />
            </div>

            <div className="flex flex-col space-y-2">
              <label className="font-semibold text-gray-700">Max Sequence Length</label>
              <input
                type="number"
                min={64}
                value={seqLength}
                onChange={(e) => {
                  const value = parseInt(e.target.value, 10);
                  setSeqLength(Number.isNaN(value) ? 64 : value);
                }}
                className="w-full rounded-md border border-gray-300 p-2"
              />
            </div>
          </div>

          <div className="space-y-3">
            <div className="flex flex-col gap-3 sm:flex-row">
              <Button
                type="button"
                variant="primary"
                onClick={handleValidate}
                disabled={colabValidation.isPending || colabNotebook.isPending}
              >
                {colabValidation.isPending ? 'Validating…' : 'Validate Configuration'}
              </Button>
              <Button
                type="button"
                variant="secondary"
                onClick={handleGenerateNotebook}
                disabled={!successData || colabNotebook.isPending}
              >
                {colabNotebook.isPending ? 'Generating Notebook…' : 'Generate Notebook'}
              </Button>
            </div>

            {successData && (
              <p className={`text-sm ${localSupported ? 'text-green-600' : 'text-yellow-600'}`}>
                {localSupported
                  ? 'Your laptop can attempt this training locally. Expect longer runtimes and keep other GPU apps closed.'
                  : 'Your laptop is VRAM constrained for this configuration. Prefer running on Google Colab for stable results.'}
              </p>
            )}
          </div>

        </form>

        {colabValidation.isError && (
          <p className="text-sm text-red-600">{colabValidation.error?.message}</p>
        )}

        {validationIssues.length > 0 && (
          <div className="rounded-md border border-red-200 bg-red-50 p-4">
            <h4 className="font-semibold text-red-700">Validation Issues</h4>
            <ul className="mt-2 list-disc space-y-1 pl-5 text-sm text-red-700">
              {validationIssues.map((issue) => (
                <li key={issue}>{issue}</li>
              ))}
            </ul>
          </div>
        )}

        {successData && (
          <div className="space-y-6">
            <div
              className={`rounded-lg border p-4 ${preferredRuntime === 'local' ? 'border-green-200 bg-green-50' : 'border-yellow-200 bg-yellow-50'}`}
            >
              <h4 className="font-semibold text-gray-800">Runtime Recommendation</h4>
              <p className="mt-1 text-sm text-gray-700">{runtimeRecommendation}</p>
              <dl className="mt-3 grid grid-cols-1 gap-2 text-sm text-gray-700 sm:grid-cols-2">
                <div className="flex flex-col">
                  <dt className="font-medium text-gray-800">Preferred Runtime</dt>
                  <dd className="capitalize">{preferredRuntime}</dd>
                </div>
                <div className="flex flex-col">
                  <dt className="font-medium text-gray-800">Local Support</dt>
                  <dd>{localSupported ? '✅ Supported with care' : '⚠️ Not recommended'}</dd>
                </div>
              </dl>

              <div className="mt-3 rounded-md border border-gray-200 bg-white p-3 text-xs text-gray-600">
                <p className="font-semibold text-gray-700">Laptop Hardware Profile</p>
                <div className="mt-2 grid grid-cols-1 gap-x-4 gap-y-1 sm:grid-cols-2">
                  <span><strong>CPU:</strong> {successData.hardware_profile.cpu}</span>
                  <span><strong>GPU:</strong> {successData.hardware_profile.gpu} ({successData.hardware_profile.gpu_vram_gb}GB)</span>
                  <span><strong>RAM:</strong> {successData.hardware_profile.ram_gb}GB</span>
                  <span><strong>OS:</strong> {successData.hardware_profile.os}</span>
                </div>
                <ul className="mt-2 list-disc space-y-1 pl-5">
                  {successData.hardware_profile.notes.map((note) => (
                    <li key={note}>{note}</li>
                  ))}
                </ul>
              </div>
            </div>

            <div className="grid grid-cols-1 gap-4 md:grid-cols-2">
              <div className="rounded-lg border border-gray-200 p-4">
                <h4 className="font-semibold text-gray-800">Runtime & Cost</h4>
                <dl className="mt-2 space-y-2 text-sm text-gray-700">
                  <div className="flex justify-between">
                    <dt>Estimated Time</dt>
                    <dd>{successData.estimated_time}</dd>
                  </div>
                  <div className="flex justify-between">
                    <dt>Estimated Cost</dt>
                    <dd>{successData.estimated_cost}</dd>
                  </div>
                  <div className="flex justify-between">
                    <dt>Runtime (minutes)</dt>
                    <dd>{successData.estimates.runtime_minutes}</dd>
                  </div>
                  <div className="flex justify-between">
                    <dt>GPU Memory (GB)</dt>
                    <dd>{successData.estimates.gpu_memory_gb}</dd>
                  </div>
                  <div className="flex justify-between">
                    <dt>Gradient Accumulation</dt>
                    <dd>{successData.estimates.gradient_accumulation_steps}</dd>
                  </div>
                </dl>
              </div>

              <div className="rounded-lg border border-gray-200 p-4">
                <h4 className="font-semibold text-gray-800">Notes</h4>
                <ul className="mt-2 list-disc space-y-1 pl-5 text-sm text-gray-700">
                  {successData.notes.map((note) => (
                    <li key={note}>{note}</li>
                  ))}
                </ul>
              </div>
            </div>

            <MetricComparisonChart data={successData.chart} title="Baseline vs Projected Metrics" />

            <div className="grid grid-cols-1 gap-4 md:grid-cols-2">
              <div className="rounded-lg border border-gray-200 p-4">
                <h4 className="font-semibold text-gray-800">Baseline Metrics</h4>
                <dl className="mt-2 space-y-2 text-sm text-gray-700">
                  {Object.entries(successData.baseline_metrics).map(([metric, value]) => (
                    <div key={metric} className="flex justify-between">
                      <dt className="capitalize">{metric.replace('_', ' ')}</dt>
                      <dd>{value ?? '—'}</dd>
                    </div>
                  ))}
                </dl>
              </div>

              <div className="rounded-lg border border-gray-200 p-4">
                <h4 className="font-semibold text-gray-800">Projected Metrics</h4>
                <dl className="mt-2 space-y-2 text-sm text-gray-700">
                  {Object.entries(successData.projected_metrics).map(([metric, value]) => (
                    <div key={metric} className="flex justify-between">
                      <dt className="capitalize">{metric.replace('_', ' ')}</dt>
                      <dd>{value ?? '—'}</dd>
                    </div>
                  ))}
                </dl>
              </div>
            </div>

            {notebookDownloadUrl && (
              <div className="rounded-lg border border-blue-200 bg-blue-50 p-4">
                <h4 className="font-semibold text-blue-800">Notebook Ready</h4>
                <p className="mt-1 text-sm text-blue-900">{colabNotebook.data?.message}</p>
                <div className="mt-3 flex flex-col gap-3 sm:flex-row sm:items-center">
                  <a
                    href={notebookDownloadUrl}
                    target="_blank"
                    rel="noopener noreferrer"
                    className="text-sm font-semibold text-blue-700 hover:underline"
                  >
                    Download Notebook
                  </a>
                  <ul className="list-disc space-y-1 pl-5 text-sm text-blue-900">
                    {colabNotebook.data?.instructions.map((instruction) => (
                      <li key={instruction}>{instruction}</li>
                    ))}
                  </ul>
                </div>
              </div>
            )}
          </div>
        )}
      </Card>
    </div>
  );
};

export default FineTuning;
