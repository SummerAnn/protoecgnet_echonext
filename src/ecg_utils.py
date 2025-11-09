import pandas as pd
import numpy as np
import wfdb
import ast
import os
import torch
from pathlib import Path
from typing import Dict, Optional, Tuple
from sklearn.preprocessing import StandardScaler
import pickle
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt

# FILE PATHS TO CHANGE
# HPC
DATASET_PATH = "/gpfs/data/bbj-lab/users/sethis/physionet.org/files/ptb-xl/1.0.3"
STANDARDIZATION_PATH = '/gpfs/data/bbj-lab/users/sethis/experiments/preprocessing'
SCP_GROUP_PATH = "scp_statementsRegrouped2.csv"

# ---------------------------------------------------------------------
# EchoNext configuration (optional)
# ---------------------------------------------------------------------
_ECHONEXT_CACHE: Dict[Tuple[str, str], Dict[str, object]] = {}

def _get_echonext_config() -> Optional[Dict[str, object]]:
    processed_dir = os.environ.get("ECHONEXT_PROCESSED_DIR")
    if not processed_dir:
        return None

    processed_path = Path(processed_dir).expanduser().resolve()
    if not processed_path.exists():
        raise FileNotFoundError(
            f"[EchoNext] Processed directory not found: {processed_path}"
        )

    mode = os.environ.get("ECHONEXT_TASK_MODE", "multilabel").lower()
    if mode not in {"binary", "multilabel", "multitask"}:
        raise ValueError(f"[EchoNext] Unsupported ECHONEXT_TASK_MODE={mode}")

    labels_dir = os.environ.get("ECHONEXT_LABELS_DIR")
    labels_path = Path(labels_dir).expanduser().resolve() if labels_dir else processed_path

    limit = os.environ.get("ECHONEXT_LIMIT_SAMPLES")
    limit_samples = int(limit) if limit and limit.isdigit() else None

    multitask_cols_env = os.environ.get("ECHONEXT_MULTITASK_COLUMNS")
    multitask_columns = None
    if multitask_cols_env:
        multitask_columns = [int(idx.strip()) for idx in multitask_cols_env.split(",") if idx.strip().isdigit()]

    return {
        "processed_dir": processed_path,
        "labels_dir": labels_path,
        "task_mode": mode,
        "limit_samples": limit_samples,
        "multitask_columns": multitask_columns,
    }

def _ensure_echonext_cache(cfg: Dict[str, object]) -> Dict[str, object]:
    key = (str(cfg["processed_dir"]), cfg["task_mode"])
    cached = _ECHONEXT_CACHE.get(key)
    if cached is None:
        cached = _load_echonext_data(cfg)
        _ECHONEXT_CACHE[key] = cached
    return cached

def remove_baseline_wander(X, sampling_rate=100, cutoff=0.5, order=1):
    """
    Applies a high-pass Butterworth filter to remove baseline wander.
    Operates on numpy arrays shaped (N, 1000, 12) or (N, 12, 1000).
    No need to apply a low-pass filter if using 100 Hz data
    """
    b, a = butter(order, cutoff / (sampling_rate / 2), btype='high', analog=False)
    
    if X.ndim == 3 and X.shape[1] == 1000 and X.shape[2] == 12:
        # Shape: (N, 1000, 12)
        for i in range(X.shape[0]):
            for j in range(12):
                X[i, :, j] = filtfilt(b, a, X[i, :, j])
    elif X.ndim == 3 and X.shape[1] == 12 and X.shape[2] == 1000:
        # Shape: (N, 12, 1000)
        for i in range(X.shape[0]):
            for j in range(12):
                X[i, j, :] = filtfilt(b, a, X[i, j, :])
    else:
        raise ValueError(f"Unexpected shape for baseline wander correction: {X.shape}")
    
    return X

def plot_ecg(
    raw_ecg,
    sampling_rate=100,
    ecg_id=None,
    true_labels=None,
    prototype_labels=None,
    rhythm_strip1='II',
    rhythm_strip2=None,
    rhythm_strip3=None,
    prototype_idx=None,
    similarity_score=None
):
    import matplotlib.pyplot as plt
    import numpy as np

    if raw_ecg.shape[0] == 12 and raw_ecg.shape[1] != 12:
        raw_ecg = raw_ecg.T
    if raw_ecg.shape[1] != 12:
        raise ValueError(f"Unexpected ECG shape: {raw_ecg.shape}")

    lead_names = ['I', 'II', 'III', 'aVR', 'aVL', 'aVF',
                  'V1', 'V2', 'V3', 'V4', 'V5', 'V6']
    lead_name_to_idx = {name: i for i, name in enumerate(lead_names)}

    rhythm_leads = []
    for lead in [rhythm_strip1, rhythm_strip2, rhythm_strip3]:
        if lead is not None:
            lead_str = lead if isinstance(lead, str) else lead[0]  
            rhythm_leads.append(lead_name_to_idx[lead_str])

    mm_per_mv = 10
    mm_per_sec = 25
    paper_speed = mm_per_sec
    gain = mm_per_mv

    duration = raw_ecg.shape[0] / sampling_rate
    stacked_leads = [
        ['I', 'aVR', 'V1', 'V4'],
        ['II', 'aVL', 'V2', 'V5'],
        ['III', 'aVF', 'V3', 'V6']
    ]

    num_rows = len(stacked_leads) + len(rhythm_leads)
    row_spacing_mm = 50

    fig_width_in = (duration * paper_speed) / 25.4
    fig_height_in = (num_rows * row_spacing_mm) / 25.4

    fig, ax = plt.subplots(figsize=(fig_width_in, fig_height_in))
    ax.set_xlim(0, duration)
    ax.set_ylim(0, num_rows * row_spacing_mm)
    ax.axis('off')

    def draw_paper_grid():
        for x in np.arange(0, duration + 0.04, 0.04):
            ax.axvline(x, color='pink', linewidth=0.5, zorder=0)
        for y in np.arange(0, num_rows * row_spacing_mm + 0.1, 1):
            is_large = (y * 0.1) % 0.5 == 0
            ax.axhline(y, color='pink' if not is_large else 'red', linewidth=0.5 if not is_large else 1.0, zorder=0)
        for x in np.arange(0, duration + 0.2, 0.2):
            ax.axvline(x, color='red', linewidth=1.0, zorder=0)

    draw_paper_grid()

    label_fontsize = 12
    row_baseline = num_rows * row_spacing_mm - row_spacing_mm / 2

    for row_idx, lead_row in enumerate(stacked_leads):
        for col_idx, lead in enumerate(lead_row):
            lead_idx = lead_name_to_idx[lead]
            start = col_idx * 250
            end = (col_idx + 1) * 250
            signal = raw_ecg[start:end, lead_idx] * gain
            t = np.linspace(col_idx * 2.5, (col_idx + 1) * 2.5, 250)
            v_offset = row_baseline - row_idx * row_spacing_mm
            ax.plot(t, signal + v_offset, color='black', linewidth=1.0)
            ax.text(t[0] + 0.1, v_offset + 16, lead, fontsize=label_fontsize, fontweight='bold')

    # Rhythm leads: full 10 seconds
    t = np.linspace(0, 10, raw_ecg.shape[0])
    for i, lead_idx in enumerate(rhythm_leads):
        v_offset = row_baseline - (len(stacked_leads) + i) * row_spacing_mm
        signal = raw_ecg[:, lead_idx] * gain
        ax.plot(t, signal + v_offset, color='black', linewidth=1.0)
        ax.text(0.1, v_offset + 16, f"{lead_names[lead_idx]}", fontsize=label_fontsize, fontweight='bold')

    # Title
    title = f"ECG {ecg_id}"
    if true_labels:
        title += f" | True: {true_labels}"
    if prototype_idx is not None: 
        title += f" | Prototype Labels: {prototype_labels}"
        similarity_score = round(similarity_score, 3)
        title += f" | Similarity Score: {similarity_score}"
    fig.suptitle(title, fontsize=14, y=0.98)

    plt.tight_layout()
    return fig

def standardize_signals(X_train, X_val, X_test, output_folder, mode):
    scaler_path = os.path.join(output_folder, f"standard_scaler_{mode}.pkl")
    if os.path.exists(scaler_path):
        with open(scaler_path, "rb") as f:
            ss = pickle.load(f)
    else:
        ss = StandardScaler()

        if mode == '1D':
            # Reshape to (num_samples * timepoints, leads) 
            X_train_flat = X_train.reshape(-1, X_train.shape[-1])  # Shape (N * 1000, 12)
        elif mode == '2D':
            # Reshape to (num_samples * height * width, leads) 
            X_train_flat = X_train.reshape(X_train.shape[0], X_train.shape[1], -1)  # (N, 12, H*W)
            X_train_flat = X_train_flat.transpose(1, 0, 2).reshape(X_train.shape[1], -1).T  # (12, N*H*W)

        ss.fit(X_train_flat)

        with open(scaler_path, "wb") as f:
            pickle.dump(ss, f)

    return apply_standardizer(X_train, ss, mode), apply_standardizer(X_val, ss, mode), apply_standardizer(X_test, ss, mode)

def apply_standardizer(X, ss, mode):
    if mode == '1D':
        X_shape = X.shape  # (N, 1000, 12)
        X_flat = X.reshape(-1, X.shape[-1])  # (N * 1000, 12) 
        X_std = ss.transform(X_flat)  
        return X_std.reshape(X_shape)  # Reshape back to (N, 1000, 12)

    elif mode == '2D':
        X_shape = X.shape  # (N, 12, H, W)
        X_flat = X.reshape(X.shape[0], X.shape[1], -1)  # (N, 12, H*W)
        X_flat = X_flat.transpose(1, 0, 2).reshape(X.shape[1], -1).T  # (12, N*H*W)
        X_std = ss.transform(X_flat)  
        X_std = X_std.T.reshape(X.shape[1], X.shape[0], -1).transpose(1, 0, 2)  
        return X_std.reshape(X_shape)  # Restore (N, 12, H, W)

def load_label_mappings(custom_groups=False, prototype_category=None):
    echonext_cfg = _get_echonext_config()
    if echonext_cfg is not None:
        if custom_groups:
            raise NotImplementedError("[EchoNext] custom_groups are not supported with EchoNext data.")
        cache = _ensure_echonext_cache(echonext_cfg)
        return cache["label_map"]

    if custom_groups:
        label_df = pd.read_csv(os.path.join(DATASET_PATH, SCP_GROUP_PATH), index_col=0) 

        # Ensure the new column exists
        assert "prototype_category" in label_df.columns, "Missing 'prototype_category' column in regrouped SCP file."

        # Filter by category if specified
        if prototype_category in [1, 2, 3, 4]:
            custom_labels = label_df[label_df["prototype_category"] == prototype_category].index.tolist()
        else:
            custom_labels = label_df.index.tolist()

        print(f"Loaded {len(custom_labels)} custom group labels (Category: {prototype_category})")

        return {
            "custom": custom_labels,
        }
    else: 
        label_df = pd.read_csv(os.path.join(DATASET_PATH, "scp_statements.csv"), index_col=0)

        # Extract mappings from SCP codes
        scp_to_superclass = label_df["diagnostic_class"].dropna().to_dict()  # SCP → Superclass
        scp_to_subclass = label_df["diagnostic_subclass"].dropna().to_dict()  # SCP → Subclass

        superdiagnostic_labels = label_df[label_df["diagnostic_class"].notna()]["diagnostic_class"].unique().tolist()
        subdiagnostic_labels = label_df[label_df["diagnostic_subclass"].notna()]["diagnostic_subclass"].unique().tolist()
        all_labels = label_df.index.tolist() 
        
        diagnostic_labels = label_df[label_df["diagnostic"] == 1.0].index.tolist()
        form_labels = label_df[label_df["form"] == 1.0].index.tolist()
        rhythm_labels = label_df[label_df["rhythm"] == 1.0].index.tolist()

        print(f"Loaded Label Mappings - Super: {len(superdiagnostic_labels)}, Sub: {len(subdiagnostic_labels)}, "
            f"All: {len(all_labels)}, Diagnostic: {len(diagnostic_labels)}, Form: {len(form_labels)}, "
            f"Rhythm: {len(rhythm_labels)}")

        return {
            "superdiagnostic": superdiagnostic_labels,
            "subdiagnostic": subdiagnostic_labels,
            "all": all_labels,
            "diagnostic": diagnostic_labels,
            "form": form_labels,
            "rhythm": rhythm_labels,
            "scp_to_superclass": scp_to_superclass,
            "scp_to_subclass": scp_to_subclass
        }


def load_raw_data(sampling_rate, label_type, df, custom_groups=False):
    label_mappings = load_label_mappings(custom_groups=custom_groups,
                                prototype_category=None if not custom_groups else int(label_type))
    selected_labels = label_mappings["custom"] if custom_groups else label_mappings[label_type]

    if not custom_groups: 
        scp_to_superclass = label_mappings["scp_to_superclass"]
        scp_to_subclass = label_mappings["scp_to_subclass"]

    # Define which column to use for label mapping
    if label_type == "superdiagnostic":
        label_map = scp_to_superclass
    elif label_type == "subdiagnostic":
        label_map = scp_to_subclass
    else:
        label_map = None  # "all", "diagnostic", "form", and "rhythm" directly use SCP codes

    # Containers
    signals = []
    valid_indices = []
    labels = []
    filtered_out = 0  # Track filtered samples

    for idx, filename in enumerate(df.filename_lr if sampling_rate == 100 else df.filename_hr):
        full_path = os.path.join(DATASET_PATH, filename)

        if not os.path.exists(full_path + ".dat"):
            print(f"Missing file: {full_path}. Skipping...")
            continue

        try:
            # Load ECG signal
            signal, _ = wfdb.rdsamp(full_path)

            # Extract SCP codes
            label_dict = df.iloc[idx].scp_codes  
            label_vector = np.zeros(len(selected_labels))
            has_valid_label = False

            contains_norm = False 
            for scp_code in label_dict.keys():
                if custom_groups:
                    # Only use SCP codes directly if they are in the selected custom group
                    if scp_code in selected_labels:
                        label_vector[selected_labels.index(scp_code)] = 1
                        has_valid_label = True
                elif label_type in ["all", "diagnostic", "form", "rhythm"]:
                    if scp_code in selected_labels:
                        label_vector[selected_labels.index(scp_code)] = 1
                        has_valid_label = True
                else:
                    # Only if label_map is not None
                    mapped_label = label_map.get(scp_code, None)
                    if mapped_label in selected_labels:
                        label_vector[selected_labels.index(mapped_label)] = 1
                        has_valid_label = True
                        if mapped_label == "NORM":
                            contains_norm = True
            if contains_norm and "NORM" in selected_labels:
                label_vector[selected_labels.index("NORM")] = 1
                has_valid_label = True


            # If at least one valid label is found, add to the dataset (only for superdiagnostic/subdiagnostic groupings)
            if label_type in ["superdiagnostic", "subdiagnostic"]:
                if has_valid_label:
                    signals.append(signal)
                    valid_indices.append(df.iloc[idx].name)
                    labels.append(label_vector)
                else:
                    filtered_out += 1
            else: 
                signals.append(signal)
                valid_indices.append(df.iloc[idx].name)
                labels.append(label_vector)

        except Exception as e:
            print(f"Error loading {full_path}: {e}")

    X = np.array(signals)
    y = np.array(labels)

    print(f"Final retained samples: {X.shape[0]}/{len(df)}")

    return X, y, valid_indices

 #Dataset Classes 
class PTBXL_Dataset_1D(Dataset):
    def __init__(self, X, y, sample_ids=None, return_sample_ids=False):
        self.X = torch.tensor(X, dtype=torch.float32).permute(0, 2, 1)  # (N, 12, 1000)
        self.y = torch.tensor(y, dtype=torch.float32) 
        self.sample_ids = sample_ids if sample_ids is not None else np.arange(len(y)) 
        self.return_sample_ids = return_sample_ids 

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        if self.return_sample_ids: 
            return self.X[idx], self.y[idx], self.sample_ids[idx]
        else: 
            return self.X[idx], self.y[idx]

class PTBXL_Dataset_2D(Dataset):
    def __init__(self, X, y, sample_ids=None, return_sample_ids=False):
        self.X = torch.tensor(X, dtype=torch.float32).unsqueeze(1).permute(0, 1, 3, 2)  # (N, 1, 12, 1000)
        self.y = torch.tensor(y, dtype=torch.float32)  
        self.sample_ids = sample_ids if sample_ids is not None else np.arange(len(y)) 
        self.return_sample_ids = return_sample_ids 

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        if self.return_sample_ids: 
            return self.X[idx], self.y[idx], self.sample_ids[idx]
        else: 
            return self.X[idx], self.y[idx]

# DataLoader Function
def get_dataloaders(batch_size=32, mode="2D", sampling_rate=100, label_set="superdiagnostic", work_num=4, return_sample_ids=False, custom_groups=False, standardize=False, remove_baseline=False):
    echonext_cfg = _get_echonext_config()
    if echonext_cfg is not None:
        cache = _ensure_echonext_cache(echonext_cfg)
        loaders = _build_echonext_dataloaders(
            cache,
            batch_size=batch_size,
            as_2d=(mode == "2D"),
            num_workers=work_num,
            return_sample_ids=return_sample_ids,
        )
        return loaders

    df = pd.read_csv(os.path.join(DATASET_PATH, "ptbxl_database.csv"), index_col="ecg_id")
    df.scp_codes = df.scp_codes.apply(lambda x: ast.literal_eval(x)) 
    
    # Split data using PTB-XL folds (Train: folds 1-8, Val: fold 9, Test: fold 10)
    train_df = df[df.strat_fold <= 8]
    val_df = df[df.strat_fold == 9]
    test_df = df[df.strat_fold == 10]
    
    X_train, y_train, train_sample_ids = load_raw_data(sampling_rate, label_set, train_df, custom_groups=custom_groups)
    X_val, y_val, val_sample_ids = load_raw_data(sampling_rate, label_set, val_df, custom_groups=custom_groups)
    X_test, y_test, test_sample_ids = load_raw_data(sampling_rate, label_set, test_df, custom_groups=custom_groups)

    # Compute class weights
    def compute_class_weights(y):
        class_counts = np.sum(y, axis=0)  
        total_samples = sum(class_counts)
        class_weights = torch.tensor([(total_samples - c) / c for c in class_counts], dtype=torch.float32)
        return torch.tensor(class_weights, dtype=torch.float32)

    class_weights = compute_class_weights(y_train)

    # Apply preprocessing (optional)
    output_folder = STANDARDIZATION_PATH
    os.makedirs(output_folder, exist_ok=True)

    if remove_baseline:
        X_train = remove_baseline_wander(X_train, sampling_rate=sampling_rate)
        X_val = remove_baseline_wander(X_val, sampling_rate=sampling_rate)
        X_test = remove_baseline_wander(X_test, sampling_rate=sampling_rate)

    if standardize: 
        X_train, X_val, X_test = standardize_signals(X_train, X_val, X_test, output_folder, mode)

    print("\n--- Data Summary ---")
    print(f"Training set: {len(train_df)} samples, Validation set: {len(val_df)} samples, Test set: {len(test_df)} samples")
    print(f"Loaded training data: X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
    print(f"Loaded validation data: X_val shape: {X_val.shape}, y_val shape: {y_val.shape}")
    print(f"Loaded test data: X_test shape: {X_test.shape}, y_test shape: {y_test.shape}")
    print(f"Label distribution (Train): {np.sum(y_train, axis=0)}")
    print(f"Label distribution (Validation): {np.sum(y_val, axis=0)}")
    print(f"Label distribution (Test): {np.sum(y_test, axis=0)}")
    print(f"Training set class weights: {class_weights}")
    
    train_dataset = PTBXL_Dataset_1D(X_train, y_train, train_sample_ids, return_sample_ids) if mode == '1D' else \
                    PTBXL_Dataset_2D(X_train, y_train, train_sample_ids, return_sample_ids)
    val_dataset = PTBXL_Dataset_1D(X_val, y_val, val_sample_ids, return_sample_ids) if mode == '1D' else \
                  PTBXL_Dataset_2D(X_val, y_val, val_sample_ids, return_sample_ids)
    test_dataset = PTBXL_Dataset_1D(X_test, y_test, test_sample_ids, return_sample_ids) if mode == '1D' else \
                   PTBXL_Dataset_2D(X_test, y_test, test_sample_ids, return_sample_ids)

    #Get validation set class weights for weighted random sampler
    val_class_weights = compute_class_weights(y_val) 
    sample_weights = np.dot(y_val, val_class_weights.numpy())  # Assign sample weights based on label presence
    val_sampler = WeightedRandomSampler(sample_weights, num_samples=len(y_val), replacement=True)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=work_num)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, sampler=val_sampler, shuffle=False, num_workers=work_num)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=work_num)
    
    return train_loader, val_loader, test_loader, class_weights


# ---------------------------------------------------------------------
# EchoNext helpers
# ---------------------------------------------------------------------
class EchoNextDataset(Dataset):
    def __init__(
        self,
        features: np.memmap,
        labels: Optional[np.memmap],
        as_2d: bool,
        sample_ids: np.ndarray,
        indices: np.ndarray,
        return_sample_ids: bool,
    ):
        self._features = features
        self._labels = labels
        self._as_2d = as_2d
        self._sample_ids = sample_ids
        self._indices = indices
        self._return_ids = return_sample_ids

    def __len__(self) -> int:
        return int(self._indices.shape[0])

    def __getitem__(self, idx):
        real_idx = int(self._indices[idx])
        x = np.asarray(self._features[real_idx])
        if x.ndim == 2:
            if x.shape[0] != 12 and x.shape[1] == 12:
                x = x.T
        elif x.ndim == 3:
            # squeeze channel dimension if present
            if x.shape[0] == 1 and x.shape[1] == 12:
                x = x[0]
        x_t = torch.from_numpy(x.astype(np.float32, copy=False))
        if self._as_2d:
            if x_t.ndim == 2:
                x_t = x_t.unsqueeze(0)
        labels = None
        if self._labels is not None:
            y = np.asarray(self._labels[real_idx])
            if y.ndim == 0:
                y = np.array([y], dtype=np.float32)
            labels = torch.from_numpy(y.astype(np.float32, copy=False))

        if self._return_ids:
            sample_id = int(self._sample_ids[real_idx])
            if labels is None:
                return x_t, sample_id
            return x_t, labels, sample_id

        if labels is None:
            return x_t
        return x_t, labels


def _load_echonext_data(cfg: Dict[str, object]) -> Dict[str, object]:
    processed_dir: Path = cfg["processed_dir"]
    task_mode: str = cfg["task_mode"]

    splits = {}
    for split in ("train", "val", "test"):
        x_path = processed_dir / f"X_{split}.npy"
        if not x_path.exists():
            raise FileNotFoundError(f"[EchoNext] Missing feature file: {x_path}")
        features = np.load(x_path, mmap_mode="r")

        labels = _load_echonext_labels(cfg, split, expected_len=features.shape[0])
        sample_ids = _load_echonext_ids(processed_dir, split, expected_len=features.shape[0])

        splits[split] = {
            "features": features,
            "labels": labels,
            "sample_ids": sample_ids,
        }

    label_names = _resolve_echonext_label_names(cfg, splits["train"]["labels"])
    n_classes = splits["train"]["labels"].shape[1] if splits["train"]["labels"] is not None else 1

    label_map = {
        "all": label_names,
        "train_labels": splits["train"]["labels"],
        "val_labels": splits["val"]["labels"],
        "test_labels": splits["test"]["labels"],
        "n_classes": n_classes,
        "label_names": label_names,
        "custom_groups": False,
        "label_set": "all",
        "processed_dir": str(processed_dir),
        "mode": task_mode,
    }

    return {
        "config": cfg,
        "splits": splits,
        "label_map": label_map,
    }


def _build_echonext_dataloaders(
    cache: Dict[str, object],
    batch_size: int,
    as_2d: bool,
    num_workers: int,
    return_sample_ids: bool,
):
    cfg = cache["config"]
    splits = cache["splits"]
    limit = cfg.get("limit_samples")

    datasets = {}
    subset_indices = {}
    for split_name, payload in splits.items():
        num_examples = payload["features"].shape[0]
        indices = np.arange(num_examples, dtype=np.int64)
        if limit is not None:
            indices = indices[: limit]
        sample_ids = payload["sample_ids"]
        if sample_ids.shape[0] != num_examples:
            sample_ids = np.arange(num_examples, dtype=np.int64)
        dataset = EchoNextDataset(
            features=payload["features"],
            labels=payload["labels"],
            as_2d=as_2d,
            sample_ids=sample_ids,
            indices=indices,
            return_sample_ids=return_sample_ids,
        )
        datasets[split_name] = dataset
        subset_indices[split_name] = indices

    class_weights = _compute_echonext_class_weights(
        splits["train"]["labels"],
        subset_indices["train"],
    )

    train_loader = DataLoader(
        datasets["train"],
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        datasets["val"],
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    test_loader = DataLoader(
        datasets["test"],
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_loader, val_loader, test_loader, class_weights


def _compute_echonext_class_weights(labels: Optional[np.memmap], indices: np.ndarray) -> torch.Tensor:
    if labels is None:
        return torch.ones(1, dtype=torch.float32)
    subset = np.asarray(labels)[indices]
    if subset.ndim == 1:
        subset = subset[:, None]
    total = subset.shape[0]
    positives = subset.sum(axis=0).astype(np.float32)
    weights = (total - positives) / (positives + 1e-6)
    return torch.from_numpy(weights.astype(np.float32))


def _load_echonext_labels(cfg: Dict[str, object], split: str, expected_len: int) -> Optional[np.memmap]:
    mode = cfg["task_mode"]
    processed_dir: Path = cfg["processed_dir"]
    labels_dir: Path = cfg["labels_dir"]

    candidates = []
    if mode == "binary":
        candidates.extend([
            processed_dir / f"{split}_binary.npy",
            processed_dir / f"y_{split}_binary.npy",
            processed_dir / f"y_{split}.npy",
        ])
    elif mode == "multilabel":
        candidates.extend([
            processed_dir / f"{split}_multilabel.npy",
            processed_dir / f"y_{split}_multilabel.npy",
        ])
    elif mode == "multitask":
        candidates.extend([
            processed_dir / f"{split}_multitask.npy",
            processed_dir / f"y_{split}_multitask.npy",
            labels_dir / f"EchoNext_{split}_labels_multilabel.npy",
            processed_dir / f"{split}_multilabel.npy",
            processed_dir / f"y_{split}_multilabel.npy",
        ])
    else:
        raise ValueError(f"[EchoNext] Unsupported task mode: {mode}")

    selected = None
    for cand in candidates:
        if cand.exists():
            selected = np.load(cand, mmap_mode="r")
            break

    if selected is None:
        raise FileNotFoundError(f"[EchoNext] Could not locate label file for split '{split}' (mode={mode}).")

    if selected.shape[0] != expected_len:
        raise ValueError(
            f"[EchoNext] Label length mismatch for split '{split}': "
            f"{selected.shape[0]} vs expected {expected_len}"
        )

    # Normalise to 2D float32
    if selected.ndim == 1:
        selected = selected[:, None]
    selected = selected.astype(np.float32, copy=False)

    if mode == "binary":
        if selected.shape[1] != 1:
            selected = selected[:, :1]
    elif mode == "multitask":
        columns = cfg.get("multitask_columns")
        if columns:
            selected = selected[:, columns]
        elif selected.shape[1] > 11:
            selected = selected[:, :11]

    return selected


def _load_echonext_ids(processed_dir: Path, split: str, expected_len: int) -> np.ndarray:
    candidate = processed_dir / f"pids_{split}.npy"
    if candidate.exists():
        arr = np.load(candidate, mmap_mode="r")
        if arr.shape[0] == expected_len:
            return np.asarray(arr)
    return np.arange(expected_len, dtype=np.int64)


def _resolve_echonext_label_names(cfg: Dict[str, object], labels: Optional[np.memmap]) -> list:
    if labels is None:
        return ["output"]
    n_outputs = labels.shape[1] if labels.ndim > 1 else 1

    override = os.environ.get("ECHONEXT_LABEL_NAMES")
    if override:
        parts = [name.strip() for name in override.split(",") if name.strip()]
        if len(parts) == n_outputs:
            return parts

    csv_path = cfg["processed_dir"] / "labels_multilabel.csv"
    if csv_path.exists():
        header = pd.read_csv(csv_path, nrows=0)
        candidates = [str(col) for col in header.columns]
        if len(candidates) == n_outputs:
            return candidates

    return [f"class_{i}" for i in range(n_outputs)]
