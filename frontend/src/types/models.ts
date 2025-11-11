export enum ModelStatus {
  DRAFT = "draft",
  TRAINING = "training",
  COMPLETED = "completed",
  DEPLOYED = "deployed",
  ARCHIVED = "archived",
  FAILED = "failed",
}

export interface Model {
  id: string;
  name: string;
  description?: string;
  base_model: string;
  created_at: string;
  updated_at: string;
  versions?: ModelVersion[];
  size_gb?: number;
  quantization?: string;
}

export interface ModelSearchResult {
  model_id: string;
  author: string;
  downloads: number;
  likes: number;
  params_billions?: number;
  tags: string[];
  library: string;
  local: boolean;
}

export interface Deployment {
  id: string;
  model_id: string;
  backend: 'ollama' | 'tgi' | 'vllm';
  status: 'running' | 'stopped' | 'failed';
  endpoint: string;
  created_at: string;
}

export interface ModelVersion {
  id: string;
  version: string;
  llm_id: string;
  status: ModelStatus;
  created_by: string;
  created_at: string;
  updated_at: string;
  completed_at?: string;
  training_dataset_version_id?: string;
  validation_dataset_version_id?: string;
  model_size?: number;
  checkpoint_path?: string;
  config?: any;
  training_metrics?: any;
  validation_metrics?: any;
  parameters: TrainingParameter[];
  tags: Tag[];
}

export interface TrainingParameter {
  id: string;
  name: string;
  description?: string;
  type: string;
  default_value?: string;
  min_value?: string;
  max_value?: string;
  created_at: string;
}

export interface Tag {
  id: string;
  name: string;
  description?: string;
  created_at: string;
}

export interface ModelMetrics {
  accuracy?: number;
  precision?: number;
  recall?: number;
  f1_score?: number;
  perplexity?: number;
  history: MetricHistory[];
  confusion_matrix: ConfusionMatrix;
}

export interface MetricHistory {
  timestamp: string;
  accuracy: number;
  precision: number;
  recall: number;
  f1_score: number;
}

export interface ConfusionMatrix {
  labels: string[];
  values: number[][];
}
