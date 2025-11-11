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
