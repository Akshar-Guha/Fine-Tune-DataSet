export interface Dataset {
  dataset_id: string;
  local_path: string;
  num_rows: number;
  num_columns: number;
  columns: string[];
  size_mb: number;
  splits: string[];
  downloaded: boolean;
}

export interface DatasetSearchResult {
  dataset_id: string;
  author: string;
  downloads: number;
  likes: number;
  tags: string[];
  size_gb?: number;
  local: boolean;
}

export interface LocalDatasetSummary {
  dataset_id: string;
  local_path: string;
  num_rows: number;
  num_columns: number;
  columns: string[];
  size_mb: number;
  splits: string[];
  downloaded: boolean;
}

// This can be used for more detailed version info if needed later
export interface DatasetVersion {
  id: string;
  dataset_id: string;
  version: string;
  status: 'draft' | 'processing' | 'ready' | 'archived' | 'failed';
  created_by: string;
  created_at: string;
  file_path?: string;
  num_records?: number;
  size_bytes?: number;
}

// Pipeline-related types
export interface DatasetValidationIssue {
  type: string;
  severity: 'low' | 'medium' | 'high' | 'critical';
  message: string;
  column?: string;
  row?: number;
}

export interface DatasetValidationSummary {
  status: string;
  quality_score?: number;
  issues: DatasetValidationIssue[];
  quality_report_path: string;
}

export interface DatasetUploadResponse {
  dataset_id: string;
  dataset_name: string;
  quality_passed: boolean;
  validation: DatasetValidationSummary;
  registry_path?: string;
  processed_file_path?: string;
}

export interface AutoLabelingRule {
  label: string;
  keywords: string[];
}

export interface DatasetPipelineConfig {
  text_column?: string;
  label_column?: string;
  enable_auto_labeling?: boolean;
  auto_labeling_rules?: AutoLabelingRule[];
  quality_score_threshold?: number;
  strip_text?: boolean;
  drop_missing_text?: boolean;
  drop_missing_label?: boolean;
  lowercase_labels?: boolean;
  allowed_label_values?: string[];
}

export interface DatasetUploadOptions extends DatasetPipelineConfig {
  dataset_name?: string;
}

export interface QualityReportCheck {
  status?: string;
  message?: string;
  issues?: DatasetValidationIssue[];
  [key: string]: unknown;
}

export interface QualityReportSummary {
  total_rows?: number;
  valid_rows?: number;
  invalid_rows?: number;
  text_column_stats?: Record<string, number>;
  label_distribution?: Record<string, number>;
  [key: string]: unknown;
}

export interface QualityReport {
  timestamp?: string;
  total_rows?: number;
  total_columns?: number;
  quality_score?: number;
  issues?: DatasetValidationIssue[];
  checks?: Record<string, QualityReportCheck>;
  summary?: QualityReportSummary;
  [key: string]: unknown;
}
