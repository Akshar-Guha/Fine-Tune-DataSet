export interface ColabConfigRequest {
  base_model: string;
  dataset_id: string;
  experiment_name: string;
  lora_rank: number;
  lora_alpha: number;
  num_epochs: number;
  batch_size: number;
  learning_rate: number;
  max_seq_length: number;
}

export interface ColabEstimates {
  runtime_minutes: number;
  gpu_memory_gb: number;
  effective_batch_tokens: number;
  gradient_accumulation_steps: number;
}

export interface ColabMetricSet {
  train_loss?: number;
  eval_loss?: number;
  perplexity?: number;
  accuracy?: number;
}

export interface ColabChartEntry {
  metric: string;
  baseline: number;
  projected: number;
}

export interface ColabHardwareProfile {
  cpu: string;
  gpu: string;
  gpu_vram_gb: number;
  ram_gb: number;
  os: string;
  notes: string[];
}

export interface ColabValidationSuccess {
  valid: true;
  estimated_time: string;
  estimated_cost: string;
  estimates: ColabEstimates;
  baseline_metrics: ColabMetricSet;
  projected_metrics: ColabMetricSet;
  chart: ColabChartEntry[];
  notes: string[];
  local_supported: boolean;
  preferred_runtime: 'local' | 'colab';
  recommendation: string;
  hardware_profile: ColabHardwareProfile;
}

export interface ColabValidationFailure {
  valid: false;
  issues: string[];
}

export type ColabValidationResponse = ColabValidationSuccess | ColabValidationFailure;

export interface ColabNotebookResponse extends ColabValidationSuccess {
  message: string;
  download_url: string;
  instructions: string[];
}
