import torch
from torch.utils.data import Dataset
import pandas as pd
import os
import h5py 
import numpy as np
from sklearn.neighbors import kneighbors_graph
class Generic_MIL_ODx_Dataset(Dataset):
    def __init__(self, mode, data_dir, omic_dir,
                 csv_path=None, slide_data_df=None, patient_to_omic_map=None, 
                 label_col=None, positive_label=None, print_info=True,k_neighbors=8):
        """
        Refactored constructor.
        - Pass 'csv_path' for the main dataset.
        - Pass 'slide_data_df' and 'patient_to_omic_map' for splits.
        """
        self.mode = mode
        self.data_dir = data_dir
        self.omic_dir = omic_dir
        self.label_col = label_col # Store the label column name
        self.positive_label = positive_label # Store the positive class string
        self.num_classes = 2
        self.k_neighbors = k_neighbors

        if csv_path is not None:
            # --- Initial Loading & Mapping ---
            print("Scanning RNA directory to create patient-to-file mapping...")
            all_omic_files = [f for f in os.listdir(omic_dir) if f.endswith('.npy')]
            self.patient_to_omic_file_map = {filename[:12]: filename for filename in all_omic_files}
            print(f"Found {len(self.patient_to_omic_file_map)} unique patient RNA files.")
            
            slide_data = pd.read_csv(csv_path)
            slide_data['label'] = slide_data[self.label_col].apply(
            lambda x: 1 if str(x).lower() == str(self.positive_label).lower() else 0
        ).astype(int)
            
            original_len = len(slide_data)
            slide_data = slide_data[slide_data['case_id'].isin(self.patient_to_omic_file_map.keys())]
            print(f"Filtered slide data to {len(slide_data)} samples with available RNA files (from {original_len} total).")

        elif slide_data_df is not None and patient_to_omic_map is not None:
            # --- Split Creation ---
            # The necessary state is passed directly.
            slide_data = slide_data_df
            self.patient_to_omic_file_map = patient_to_omic_map # Set the map immediately.
        else:
            raise ValueError("Invalid arguments. Provide either 'csv_path' or both 'slide_data_df' and 'patient_to_omic_map'.")

        self.slide_data = slide_data


        original_len = len(slide_data)
        slide_data = slide_data[slide_data['case_id'].isin(self.patient_to_omic_file_map.keys())]
        print(f"Filtered slide data to {len(slide_data)} samples with available RNA files (from {original_len} total).")
      

        
        self.slide_cls_ids = [[] for i in range(self.num_classes)]
        for i in range(self.num_classes):
            self.slide_cls_ids[i] = self.slide_data.index[self.slide_data['label'] == i].tolist()
        
        # This will now work correctly for both the main dataset and splits.
        self._get_omic_dim()

        if print_info:
            self.summarize()
            
    def getlabel(self, idx):
        return self.slide_data['label'].iloc[idx]
    
    def _get_omic_dim(self):
        if len(self.slide_data) == 0:
            self.omic_dim = 0
            return

        sample_case_id = self.slide_data['case_id'].iloc[0]
        sample_filename = self.patient_to_omic_file_map.get(sample_case_id)
        
        if sample_filename:
            omic_path = os.path.join(self.omic_dir, sample_filename)
            rna_features = np.load(omic_path)
            self.omic_dim = rna_features.shape[0]
        else:
            self.omic_dim = 0
            print(f"Warning: Could not find a corresponding RNA file for sample case ID {sample_case_id}")

    def get_omic_dim(self):
        return self.omic_dim



    def _get_omic_dim(self):
        if len(self.slide_data) == 0:
            self.omic_dim = 0
            return

        sample_case_id = self.slide_data['case_id'].iloc[0]
        sample_filename = self.patient_to_omic_file_map.get(sample_case_id)
        
        if sample_filename:
            omic_path = os.path.join(self.omic_dir, sample_filename)
            rna_features = np.load(omic_path)
            self.omic_dim = rna_features.shape[0]
        else:
            self.omic_dim = 0
            print(f"Warning: Could not find a corresponding RNA file for sample case ID {sample_case_id}")


    def summarize(self):
        print("\n--- Dataset Summary ---")
        print(f"Number of samples: {len(self.slide_data)}")
        # ... (rest of summary is fine)
        print("---------------------\n")

    def __len__(self):
        return len(self.slide_data)

    def __getitem__(self, idx):
        row = self.slide_data.iloc[idx]
        case_id = row['case_id']
        slide_id = row['slide_id']
        label = row['label']

        h5_path = os.path.join(self.data_dir, f"{slide_id.replace('.svs', '')}.h5")
        try:
            with h5py.File(h5_path, 'r') as hf:
                wsi_features = torch.from_numpy(hf['features'][:]).float()
                
                # --- START: GRAPH CONSTRUCTION LOGIC ---
                # Best practice: Check if a pre-computed graph exists in the h5 file
                edge_index_key = f'edge_index_k{self.k_neighbors}'
                if edge_index_key in hf:
                    edge_index = torch.from_numpy(hf[edge_index_key][:]).long()
                else:
                    # Fallback: Compute on-the-fly if not pre-computed
                    # This will be slow during training, a warning is useful
                    if 'coords' in hf:
                        coords = hf['coords'][:]
                        # Using mode='distance' can be memory intensive for large graphs
                        # mode='connectivity' is sparse and efficient
                        adj_matrix = kneighbors_graph(coords, self.k_neighbors, mode='connectivity', include_self=False)
                        edge_index = torch.from_numpy(np.stack(adj_matrix.nonzero())).long()
                    else:
                        # If no coords, we cannot build a graph.
                        print(f"Warning: 'coords' not found in {h5_path}. Cannot build graph. Returning empty edge_index.")
                        edge_index = torch.empty((2, 0), dtype=torch.long)
                # --- END: GRAPH CONSTRUCTION LOGIC ---

        except Exception as e:
            print(f"ERROR loading data for slide_id {slide_id}: {e}")
            # Return dummy data to allow training to continue
            wsi_features = torch.zeros(1, 1536)
            edge_index = torch.empty((2, 0), dtype=torch.long)
            # We still need rna_features of the correct dimension
            rna_features = torch.zeros(self.omic_dim) if self.omic_dim > 0 else torch.empty(0)
            return (wsi_features, edge_index), rna_features, -1, "LOAD_ERROR"

        # === START: ROBUST FILENAME LOOKUP ===
        # 1. Get the correct, full filename from our pre-computed map.
        omic_filename = self.patient_to_omic_file_map.get(case_id)
        if not omic_filename:
            # This should ideally not happen if the dataset was filtered correctly.
            raise FileNotFoundError(f"RNA file for case_id {case_id} not found in the map.")
        
        # 2. Construct the full path and load the data.
        omic_path = os.path.join(self.omic_dir, omic_filename)
        rna_features = torch.from_numpy(np.load(omic_path)).float()
        # === END: ROBUST FILENAME LOOKUP ===
        
        return (wsi_features, edge_index), rna_features, label, slide_id
        
    def get_split_from_df(self, all_splits, split_key='train'):
        """
        Modified to use slide_id for splitting instead of case_id.
        """
        print(f"\n--- [DEBUG] In get_split_from_df for split: '{split_key}' ---")

        # 1. Get the list of slide_ids for the current split
        if split_key not in all_splits.columns:
            print(f"  [DEBUG] Column '{split_key}' not found in splits file. Returning empty dataset.")
            # Return an empty dataframe to create an empty dataset
            split_df = pd.DataFrame(columns=self.slide_data.columns)
        else:
            split_slide_ids = all_splits[split_key].dropna().unique()
            print(f"  [DEBUG] Found {len(split_slide_ids)} unique slide IDs in the '{split_key}' column of the splits file.")
            
            # 2. Filter the main dataset's DataFrame to only include these slide_ids.
            print(f"  [DEBUG] Main dataset has {len(self.slide_data)} slides before filtering.")
            split_df = self.slide_data[self.slide_data['slide_id'].isin(split_slide_ids)].reset_index(drop=True)
            print(f"  [DEBUG] After filtering, the split DataFrame has {len(split_df)} slides.")

        # This check is crucial
        if len(split_df) == 0:
            print(f"  [CRITICAL WARNING] The '{split_key}' split is EMPTY. This will cause downstream errors.")
        
        split_dataset = Generic_MIL_ODx_Dataset(
            slide_data_df=split_df,
            patient_to_omic_map=self.patient_to_omic_file_map,
            positive_label=self.positive_label,
            mode=self.mode,
            data_dir=self.data_dir,
            omic_dir=self.omic_dir,
            label_col=self.label_col,
            print_info=False,
            k_neighbors=self.k_neighbors,
        )
        return split_dataset

    def return_splits(self, from_id: bool=False, csv_path: str=None):
        assert csv_path 
        all_splits = pd.read_csv(csv_path)
        train_split = self.get_split_from_df(all_splits=all_splits, split_key='train')
        val_split = self.get_split_from_df(all_splits=all_splits, split_key='val')
        return train_split, val_split
class GCAD_Inference_WSI_Dataset(Dataset):
    """A lightweight dataset for GCAD-Net inference that loads WSI features AND the graph."""
    def __init__(self, test_df, data_dir, k_neighbors=8):
        self.data_df = test_df.reset_index(drop=True)
        self.data_dir = data_dir
        self.k_neighbors = k_neighbors

    def __len__(self):
        return len(self.data_df)

    def __getitem__(self, idx):
        row = self.data_df.iloc[idx]
        slide_id = row['slide_id']
        label = row['label']
        h5_path = os.path.join(self.data_dir, f"{slide_id.replace('.svs', '')}.h5")
        
        try:
            with h5py.File(h5_path, 'r') as hf:
                wsi_features = torch.from_numpy(hf['features'][:]).float()
                
                # Re-use the same graph logic as in the training dataset
                edge_index_key = f'edge_index_k{self.k_neighbors}'
                if edge_index_key in hf:
                    edge_index = torch.from_numpy(hf[edge_index_key][:]).long()
                else:
                    coords = hf['coords'][:]
                    adj_matrix = kneighbors_graph(coords, self.k_neighbors, mode='connectivity', include_self=False)
                    edge_index = torch.from_numpy(np.stack(adj_matrix.nonzero())).long()
                    
        except Exception as e:
            print(f"ERROR loading data for slide_id {slide_id}: {e}")
            wsi_features = torch.zeros(1, 1536)
            edge_index = torch.empty((2, 0), dtype=torch.long)
            return (wsi_features, edge_index), -1, "LOAD_ERROR"

        return (wsi_features, edge_index), label, slide_id
class Inference_WSI_Dataset(Dataset):
    """A lightweight dataset for inference that ONLY loads WSI features."""
    def __init__(self, test_df, data_dir):
        self.data_df = test_df.reset_index(drop=True)
        self.data_dir = data_dir

    def __len__(self):
        return len(self.data_df)

    def __getitem__(self, idx):
        row = self.data_df.iloc[idx]
        slide_id = row['slide_id']
        label = row['label']

        h5_path = os.path.join(self.data_dir, f"{slide_id.replace('.svs', '')}.h5")
        
        try:
            with h5py.File(h5_path, 'r') as hf:
                wsi_features = torch.from_numpy(hf['features'][:]).float()
        except FileNotFoundError:
            print(f"ERROR: WSI feature file not found at {h5_path}.")
            wsi_features = torch.zeros(1, 1024) 

        return wsi_features, label, slide_id