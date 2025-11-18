"""EchoNext dataset and dataloader for ProtoECGNet adaptation."""

import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, Sampler
from typing import Tuple, Optional, List
from sklearn.preprocessing import MultiLabelBinarizer


class EchoNextDataset(Dataset):
    """EchoNext dataset loader for multi-label ECG classification.
    
    Loads waveforms from .npy files and labels from metadata CSV.
    Supports train/val/test splits.
    """
    
    def __init__(
        self,
        dataset_root: str,
        split: str = "train",
        label_set: str = "cat1",
        limit: Optional[int] = None,
        mlb: Optional[MultiLabelBinarizer] = None,
    ):
        """
        Args:
            dataset_root: Root directory containing EchoNext files
            split: One of 'train', 'val', 'test'
            label_set: Label set identifier (e.g., 'cat1')
            limit: Optional limit on number of samples (for smoke tests)
            mlb: Optional pre-fitted MultiLabelBinarizer (for consistent label space)
        """
        self.dataset_root = os.path.expanduser(dataset_root)
        self.split = split
        self.label_set = label_set
        self.mlb = mlb
        
        # Load metadata
        meta_path = os.path.join(self.dataset_root, "EchoNext_metadata_100k.csv")
        if not os.path.exists(meta_path):
            raise FileNotFoundError(f"Missing metadata CSV at {meta_path}")
        
        df = pd.read_csv(meta_path)
        if "split" not in df.columns:
            raise ValueError(f"Metadata CSV missing 'split' column")
        
        # Filter by split
        split_df = df[df["split"] == split].reset_index(drop=True)
        
        # Load waveforms
        waveforms_path = os.path.join(self.dataset_root, f"EchoNext_{split}_waveforms.npy")
        if not os.path.exists(waveforms_path):
            raise FileNotFoundError(f"Missing waveforms file at {waveforms_path}")
        
        print(f"[EchoNext] Loading waveforms from {waveforms_path}...")
        waveforms = np.load(waveforms_path, mmap_mode="r")  # Memory-mapped for efficiency
        
        # Handle different waveform shapes
        if waveforms.ndim == 4:
            # (N, 1, T, 12) -> (N, T, 12)
            waveforms = waveforms[:, 0, :, :]
        elif waveforms.ndim == 3:
            # (N, T, 12) - already correct
            pass
        else:
            raise ValueError(f"Unexpected waveform shape: {waveforms.shape}")
        
        # Ensure (N, T, 12) format, then transpose to (N, 12, T) for model
        if waveforms.shape[-1] != 12:
            raise ValueError(f"Expected 12 leads, got {waveforms.shape[-1]}")
        
        # Align metadata with waveforms
        # Assume metadata rows correspond to waveform rows in order
        if len(split_df) != waveforms.shape[0]:
            print(f"[WARNING] Metadata rows ({len(split_df)}) != waveform samples ({waveforms.shape[0]}). Using min.")
            min_len = min(len(split_df), waveforms.shape[0])
            split_df = split_df.iloc[:min_len].reset_index(drop=True)
            waveforms = waveforms[:min_len]
        
        # Apply limit if specified
        if limit is not None and limit > 0:
            limit = min(limit, len(split_df))
            split_df = split_df.iloc[:limit].reset_index(drop=True)
            waveforms = waveforms[:limit]
        
        self.waveforms = waveforms
        self.metadata = split_df
        
        # Extract labels
        # EchoNext uses flag columns (ending in _flag) as binary labels
        # Also check for label_* columns or a 'labels' column as fallback
        label_cols = [col for col in split_df.columns if col.endswith("_flag") or col.startswith("label_") or col == "labels"]
        
        if not label_cols:
            raise ValueError(f"No label columns found in metadata. Available columns: {split_df.columns.tolist()}")
        
        # Handle label format
        if "labels" in split_df.columns:
            # Assume labels are stored as string representations of lists
            labels_list = []
            for lbl_str in split_df["labels"]:
                if isinstance(lbl_str, str):
                    # Parse string representation
                    import ast
                    try:
                        lbl = ast.literal_eval(lbl_str)
                        if isinstance(lbl, list):
                            labels_list.append(lbl)
                        else:
                            labels_list.append([lbl])
                    except:
                        labels_list.append([])
                elif isinstance(lbl_str, (list, np.ndarray)):
                    labels_list.append(list(lbl_str))
                else:
                    labels_list.append([])
            
            # Convert to multi-hot encoding
            if self.mlb is None:
                # Fit on train data
                mlb = MultiLabelBinarizer()
                self.labels = mlb.fit_transform(labels_list).astype(np.float32)
                self.class_names = mlb.classes_.tolist()
                self.mlb = mlb  # Store for potential reuse
            else:
                # Transform with pre-fitted binarizer
                self.labels = self.mlb.transform(labels_list).astype(np.float32)
                self.class_names = self.mlb.classes_.tolist()
        else:
            # Use flag columns or label_* columns as binary indicators
            # Prefer _flag columns (EchoNext format)
            flag_cols = [col for col in label_cols if col.endswith("_flag")]
            if flag_cols:
                label_cols = sorted(flag_cols)
            else:
                # Fallback to label_* columns
                label_cols = sorted([col for col in label_cols if col.startswith("label_")])
            
        self.labels = split_df[label_cols].values.astype(np.float32)
        # Clean up class names (remove _flag suffix)
        self.class_names = [col.replace("_flag", "").replace("label_", "") for col in label_cols]
        
        # Expose boolean labels for sampler (PR5-B)
        self.labels_bool = self.labels.astype(bool)
        
        print(f"[EchoNext] Loaded {len(self)} samples from {split} split")
        print(f"[EchoNext] Labels shape: {self.labels.shape}, classes: {len(self.class_names)}")
    
    def __len__(self) -> int:
        return len(self.metadata)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get waveform and labels for a single sample.
        
        Returns:
            waveform: (12, T) float32 tensor
            labels: (num_classes,) float32 multi-hot vector
        """
        # Load waveform (already in memory or memory-mapped)
        waveform = self.waveforms[idx].astype(np.float32)  # (T, 12)
        
        # Transpose to (12, T) for model input
        waveform = waveform.T  # (12, T)
        
        # Convert to tensor
        waveform = torch.from_numpy(waveform)
        
        # Get labels
        labels = torch.from_numpy(self.labels[idx])
        
        return waveform, labels


def get_echonext_dataloaders(
    dataset_root: str,
    train_split: str = "train",
    val_split: str = "val",
    batch_size: int = 32,
    num_workers: int = 0,
    label_set: str = "cat1",
    limit: Optional[int] = None,
    shuffle_train: bool = True,
    weight_clip_max: float = 5.0,
    use_inverse_sqrt: bool = False,
    sampler_mode: str = "none",  # "none" | "las" | "weighted"
    sampler_gamma: float = 0.5,
    iters_per_epoch: Optional[int] = None,
    seed: int = 42,
) -> Tuple[DataLoader, DataLoader, DataLoader, Optional[torch.Tensor], Optional[dict]]:
    """Create EchoNext dataloaders for train/val/test.
    
    Args:
        dataset_root: Root directory containing EchoNext files
        train_split: Training split name
        val_split: Validation split name
        batch_size: Batch size for all loaders
        num_workers: Number of dataloader workers
        label_set: Label set identifier
        limit: Optional limit on samples (for smoke tests)
        shuffle_train: Whether to shuffle training data
    
    Returns:
        train_loader, val_loader, test_loader, class_weights
        class_weights is None (can be computed from labels if needed)
    """
    # Create train dataset first to determine label space
    train_dataset = EchoNextDataset(
        dataset_root=dataset_root,
        split=train_split,
        label_set=label_set,
        limit=limit,
        mlb=None,  # Fit on train
    )
    
    # Get label space from train dataset (and the fitted binarizer if used)
    num_classes = train_dataset.labels.shape[1]
    mlb = train_dataset.mlb  # May be None if using label_* columns
    
    # Create val dataset with same label space (use fitted binarizer if available)
    val_dataset = EchoNextDataset(
        dataset_root=dataset_root,
        split=val_split,
        label_set=label_set,
        limit=None,  # Don't limit val
        mlb=mlb,  # Use train's binarizer for consistent label space
    )
    
    # Ensure val labels match train label space (should be automatic with mlb, but double-check)
    if val_dataset.labels.shape[1] != num_classes:
        print(f"[WARNING] Val labels shape {val_dataset.labels.shape[1]} != train {num_classes}. This shouldn't happen with shared mlb.")
        # Pad or truncate to match as fallback
        if val_dataset.labels.shape[1] < num_classes:
            padding = np.zeros((len(val_dataset), num_classes - val_dataset.labels.shape[1]), dtype=np.float32)
            val_dataset.labels = np.concatenate([val_dataset.labels, padding], axis=1)
        else:
            val_dataset.labels = val_dataset.labels[:, :num_classes]
    
    # Try to create test dataset (may not exist)
    try:
        test_dataset = EchoNextDataset(
            dataset_root=dataset_root,
            split="test",
            label_set=label_set,
            limit=None,
            mlb=mlb,  # Use same binarizer
        )
        # Ensure test labels match too
        if test_dataset.labels.shape[1] != num_classes:
            if test_dataset.labels.shape[1] < num_classes:
                padding = np.zeros((len(test_dataset), num_classes - test_dataset.labels.shape[1]), dtype=np.float32)
                test_dataset.labels = np.concatenate([test_dataset.labels, padding], axis=1)
            else:
                test_dataset.labels = test_dataset.labels[:, :num_classes]
    except (FileNotFoundError, ValueError):
        print("[WARNING] Test split not found, using validation as test")
        test_dataset = val_dataset
    
    # Create dataloaders with optional class-aware sampling (PR5-B)
    if iters_per_epoch is None or iters_per_epoch == 0:
        # default: roughly len(train)//batch_size to keep epoch semantics
        iters_per_epoch = max(1, len(train_dataset) // batch_size)
    
    if sampler_mode == "las":
        labels_bool = train_dataset.labels_bool  # (N, C) bool array
        batch_sampler = LabelAwareBatchSampler(
            labels_bool=labels_bool,
            batch_size=batch_size,
            iters_per_epoch=iters_per_epoch,
            gamma=sampler_gamma,
            seed=seed,
        )
        train_loader = DataLoader(
            train_dataset,
            batch_sampler=batch_sampler,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
        )
        print(f"[LAS] gamma={sampler_gamma}, iters_per_epoch={iters_per_epoch}, batch_size={batch_size}")
    elif sampler_mode == "weighted":
        # Optional: per-sample weights = sum_c ( y_ic * w_c ), with w_c ∝ 1/freq^gamma
        freq = train_dataset.labels_bool.sum(axis=0).astype(float)
        w_c = (np.maximum(freq, 1.0)) ** (-sampler_gamma)
        w_c = w_c / w_c.sum() * len(w_c)
        Y = train_dataset.labels_bool.astype(float)
        w_i = (Y * w_c).sum(axis=1)
        w_i = np.clip(w_i, 1e-8, None)
        sampler = torch.utils.data.WeightedRandomSampler(
            weights=torch.tensor(w_i, dtype=torch.double),
            num_samples=len(train_dataset),
            replacement=True,
        )
        train_loader = DataLoader(
            train_dataset,
            sampler=sampler,
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
        )
        print(f"[EchoNext] Using Weighted Sampler: gamma={sampler_gamma}")
    else:
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=shuffle_train,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
        )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    
    # Compute class weights (optional, for balanced training)
    # PR5-A: Gentler class weighting with clipping and optional inverse-sqrt
    class_weights = None
    if len(train_dataset.labels) > 0:
        pos_counts = train_dataset.labels.sum(axis=0).astype(np.float32)
        total = len(train_dataset.labels)
        neg_counts = total - pos_counts
        
        # Check for zero-positive classes (edge case)
        zero_pos_mask = pos_counts == 0
        if zero_pos_mask.any():
            print(f"[WARNING] {zero_pos_mask.sum()} classes have zero positives in training set. "
                  f"Setting their weights to clip_max={weight_clip_max}")
        
        # Avoid division by zero: clamp pos to min 1.0
        pos_counts_safe = np.maximum(pos_counts, 1.0)
        neg_counts_safe = np.maximum(neg_counts, 1.0)
        
        # Compute ratio: (neg_counts / pos_counts)
        ratio = neg_counts_safe / pos_counts_safe
        
        # Clamp minimum to 1.0 (no down-weighting of common classes)
        ratio = np.maximum(ratio, 1.0)
        
        # Optional: apply inverse-sqrt for gentler weighting
        if use_inverse_sqrt:
            ratio = np.sqrt(ratio)
        
        # Clip maximum to prevent over-weighting rare classes
        ratio = np.minimum(ratio, weight_clip_max)
        
        # For zero-positive classes, set weight to clip_max
        if zero_pos_mask.any():
            ratio[zero_pos_mask] = weight_clip_max
        
        # Sanity checks: ensure all finite and in bounds
        assert np.isfinite(ratio).all(), f"Non-finite weights detected: {ratio[~np.isfinite(ratio)]}"
        assert (ratio >= 1.0).all(), f"Weights below 1.0: {ratio[ratio < 1.0]}"
        assert (ratio <= weight_clip_max).all(), f"Weights above clip_max: {ratio[ratio > weight_clip_max]}"
        
        class_weights = ratio.astype(np.float32)
        class_weights = torch.from_numpy(class_weights)
        
        method = "inverse-sqrt" if use_inverse_sqrt else "inverse-freq"
        print(f"[EchoNext] Computed class weights ({method}, clip_max={weight_clip_max}): "
              f"min={class_weights.min():.3f}, max={class_weights.max():.3f}, mean={class_weights.mean():.3f}")
        
        # Prepare class frequency metadata for audit
        class_freqs = {
            "pos_counts": pos_counts.tolist(),
            "neg_counts": neg_counts.tolist(),
            "total": int(total),
            "num_classes": int(len(pos_counts)),
            "weight_clip_max": float(weight_clip_max),
            "use_inverse_sqrt": use_inverse_sqrt,
            "final_weights": class_weights.tolist(),
            "zero_pos_classes": np.where(zero_pos_mask)[0].tolist() if zero_pos_mask.any() else [],
        }
    else:
        class_freqs = None
    
    return train_loader, val_loader, test_loader, class_weights, class_freqs


# --- NEW: Label-Aware Sampler for multi-label (PR5-B) --------------------------------

class LabelAwareBatchSampler(torch.utils.data.Sampler):
    """
    Multi-label class-aware batch sampler.
    
    - Computes class frequencies from a boolean label matrix Y (N x C).
    - Chooses classes per item with p(c) ∝ (freq[c])^(-gamma).
    - For each chosen class, draws an index that has that class positive.
    - Deduplicates indices; if exhausted, backfills uniformly.
    
    Note: This is a Sampler that yields batch-sized lists of indices.
    For distributed training compatibility, wrap with BatchSampler or use
    Trainer(use_distributed_sampler=False).
    """
    
    def __init__(
        self,
        labels_bool: np.ndarray,   # (N, C) bool
        batch_size: int,
        iters_per_epoch: int,
        gamma: float = 0.5,
        max_retries_per_item: int = 20,
        seed: int = 42,
    ):
        assert labels_bool.ndim == 2
        self.labels = labels_bool.astype(bool)
        self.N, self.C = self.labels.shape
        self.batch_size = batch_size
        self.iters_per_epoch = iters_per_epoch
        self.gamma = float(gamma)
        self.max_retries_per_item = int(max_retries_per_item)
        self.rng = np.random.default_rng(seed)
        
        # class frequencies
        self.freq = self.labels.sum(axis=0).astype(float)  # (C,)
        # avoid zeros (unseen classes): small epsilon to keep them selectable if they appear later
        eps = 1.0
        freq_safe = np.maximum(self.freq, eps)
        # class sampling probs
        weights = freq_safe ** (-self.gamma)
        self.p_class = weights / weights.sum()
        
        # index pools per class
        self.class_to_indices = [np.flatnonzero(self.labels[:, c]) for c in range(self.C)]
        # for quick uniform fallback
        self.all_indices = np.arange(self.N)
    
    def __len__(self) -> int:
        return self.iters_per_epoch
    
    def __iter__(self):
        for _ in range(self.iters_per_epoch):
            chosen = set()
            # Fill the batch by repeatedly picking a class then an example of that class
            for _k in range(self.batch_size):
                # choose class
                c = int(self.rng.choice(self.C, p=self.p_class))
                pool = self.class_to_indices[c]
                picked = None
                if len(pool) > 0:
                    # try a few times to get a not-yet-used index
                    for _try in range(self.max_retries_per_item):
                        i = int(pool[self.rng.integers(0, len(pool))])
                        if i not in chosen:
                            picked = i
                            break
                if picked is None:
                    # fallback: uniform draw across all indices
                    for _try in range(self.max_retries_per_item):
                        i = int(self.all_indices[self.rng.integers(0, self.N)])
                        if i not in chosen:
                            picked = i
                            break
                chosen.add(picked if picked is not None else int(self.rng.integers(0, self.N)))
            yield list(chosen)

