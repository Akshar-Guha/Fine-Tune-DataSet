export enum JobType {
  SFT_TRAINING = "sft_training",
  QUANTIZATION = "quantization",
  RLHF = "rlhf",
  EVALUATION = "evaluation",
}

export enum JobStatus {
  PENDING = "pending",
  RUNNING = "running",
  COMPLETED = "completed",
  FAILED = "failed",
  CANCELLED = "cancelled",
}

export interface Job {
  job_id: string;
  name: string;
  job_type: JobType;
  status: JobStatus;
  config: any;
  dataset_id?: string;
  base_model?: string;
  created_at: string;
  duration_seconds?: number;
  error?: string;
  metrics?: any;
}

export interface FineTuningJobCreateRequest {
  name: string;
  base_model: string;
  dataset_id: string;
  config: {
    lora_rank?: number;
    lora_alpha?: number;
    num_epochs?: number;
    learning_rate?: number;
  };
}

export interface QuantizationJobCreateRequest {
  model_id: string;
  method: 'awq' | 'gptq' | 'gguf';
  bits?: number;
}

export interface RLHFJobCreateRequest {
  name: string;
  base_model: string;
  dataset_id: string;
  config: any; // Simplified for now
}

export type JobCreateRequest = 
  | FineTuningJobCreateRequest 
  | QuantizationJobCreateRequest 
  | RLHFJobCreateRequest;

export interface JobsResponse {
  jobs: Job[];
  total: number;
}
