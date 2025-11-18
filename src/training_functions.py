import os
import torch
import pytorch_lightning as pl
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger
from sklearn.metrics import roc_auc_score, f1_score, multilabel_confusion_matrix, roc_curve, precision_recall_curve, average_precision_score
from proto_models1D import ProtoECGNet1D, prototype_loss1d
from proto_models2D import ProtoECGNet2D, prototype_loss2d
from fusion import FusionProtoClassifier
from push import push_prototypes1d, push_prototypes2d
import torch.nn.functional as F
import itertools
from collections import defaultdict
from losses import AsymmetricLossMultiLabel


def get_class_names(args):
    # For EchoNext adaptation, get class names from the dataset
    if args.training_stage == 'echonext_adapt':
        # Return placeholder - will be set from dataset in ECGTrainer
        # We'll need to pass this differently
        return None  # Will be handled in ECGTrainer.__init__
    
    from ecg_utils import load_label_mappings

    label_mappings = load_label_mappings(
        custom_groups=args.custom_groups,
        prototype_category=int(args.label_set) if args.custom_groups else None
    )

    if args.custom_groups:
        return label_mappings["custom"]
    else:
        return label_mappings[args.label_set]

def seed_everything(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    pl.seed_everything(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

def save_study(study, filename):
    joblib.dump(study, filename)

def load_study(filename):
    return joblib.load(filename)

def save_model_weights(model, job_name, stage, save_dir="./checkpoints", save_weights=True):
    """ Save only model weights after training if save_weights is True. Unnecessary if using PyTorch Lightning checkpointing."""
    if save_weights:
        save_prepath = os.path.join(save_dir, job_name)
        os.makedirs(save_prepath, exist_ok=True)
        save_path = os.path.join(save_prepath, f"{job_name}_{stage}.pth")
        torch.save(model.state_dict(), save_path)
        print(f"Model weights saved to {save_path}")

def load_model_weights(model, checkpoint_path):
    """ Load only model weights (not optimizer state). """
    if checkpoint_path:
        state_dict = torch.load(checkpoint_path, map_location=torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
        model.load_state_dict(state_dict, strict=False)
        print(f"Loaded model weights from {checkpoint_path}")

class ECGTrainer(pl.LightningModule):
    def __init__(self, model, lr=1e-3, l2=0, args=None, class_weights=None,
                 loss_name="bce", asl_gamma_pos=0.0, asl_gamma_neg=4.0, asl_clip=0.05,
                 asl_disable_pos_weight=False, use_class_weights=True):
        super().__init__()
        self.model = model
        self.lr = lr
        self.l2 = l2
        self.args = args
        self.is_proto = isinstance(model, (ProtoECGNet1D, ProtoECGNet2D, FusionProtoClassifier))
        self.class_weights = class_weights.to(self.device) if class_weights is not None else None

        # Create classification criterion (ASL or BCE)
        if loss_name == "asl":
            pos_weight = None
            if use_class_weights and (not asl_disable_pos_weight) and (self.class_weights is not None):
                pos_weight = self.class_weights  # Tensor [C]; keep moderate (e.g., unclipped inverse-freq)
            self.cls_criterion = AsymmetricLossMultiLabel(
                gamma_pos=asl_gamma_pos,
                gamma_neg=asl_gamma_neg,
                clip=asl_clip,
                reduction="mean",
                pos_weight=pos_weight,
            )
            print(f"[ASL] Initialized: gamma_pos={asl_gamma_pos}, gamma_neg={asl_gamma_neg}, clip={asl_clip}, pos_weight={'enabled' if pos_weight is not None else 'disabled'}")
        else:
            # BCE baseline
            self.cls_criterion = None  # Will use F.binary_cross_entropy_with_logits directly

        if self.args.training_stage in ["classifier", "fusion"]:
            if self.cls_criterion is not None:
                self.criterion = lambda logits, y, *_: self.cls_criterion(logits, y) + self.args.l1 * self.compute_l1_loss()
            else:
                self.criterion = lambda logits, y, *_: torch.nn.BCEWithLogitsLoss(pos_weight=self.class_weights.to(logits.device) if class_weights is not None else None)(logits, y.to(logits.device)) + self.args.l1 * self.compute_l1_loss()
        elif self.args.training_stage == "echonext_adapt" and self.args.train_head_only:
            # Head-only training: use classification loss only (no prototype losses)
            if self.cls_criterion is not None:
                self.criterion = lambda logits, y, *_: self.cls_criterion(logits, y)
            else:
                self.criterion = lambda logits, y, *_: torch.nn.BCEWithLogitsLoss(pos_weight=self.class_weights.to(logits.device) if class_weights is not None else None)(logits, y.to(logits.device))
        elif self.args.training_stage == "echonext_adapt" and not self.args.train_head_only:
            # Joint fine-tuning: encoder frozen, prototypes+head trainable
            # Use EchoNext-specific loss with per-sample, per-class masking
            if self.args.dimension == "1D":
                from proto_models1D import prototype_loss1d_echonext
                # Pass cls_criterion to prototype_loss1d_echonext
                cls_crit = self.cls_criterion if self.cls_criterion is not None else None
                self.criterion = lambda logits, y, model, similarity_scores: prototype_loss1d_echonext(
                    logits=logits,
                    y_true=y,
                    model=model,
                    similarity_scores=similarity_scores,
                    lam_clst=self.args.lam_clst,
                    lam_sep=self.args.lam_sep,
                    lam_spars=self.args.lam_spars,
                    lam_div=self.args.lam_div,
                    lam_cnrst=self.args.lam_cnrst,
                    class_weights=self.class_weights,
                    use_contrastive=False,  # Disable contrastive for EchoNext
                    cls_criterion=cls_crit  # Pass ASL criterion if available
                )
            else:
                # 2D version would go here if needed
                raise NotImplementedError("EchoNext adaptation for 2D models not implemented")
        elif self.args.training_stage in ["prototypes", "joint"]:
            if self.args.dimension == "1D":
                self.criterion = lambda logits, y, model, similarity_scores: prototype_loss1d(
                    logits=logits,
                    y_true=y,
                    model=model,
                    similarity_scores=similarity_scores,
                    lam_clst=self.args.lam_clst,
                    lam_sep=self.args.lam_sep,
                    lam_spars=self.args.lam_spars,
                    lam_div=self.args.lam_div,
                    lam_cnrst=self.args.lam_cnrst,
                    class_weights=self.class_weights
                )
            else: 
                self.criterion = lambda logits, y, model, similarity_scores: prototype_loss2d(
                logits=logits,
                y_true=y,
                model=model,
                similarity_scores=similarity_scores,
                lam_clst=self.args.lam_clst,
                lam_sep=self.args.lam_sep,
                lam_spars=self.args.lam_spars,
                lam_div=self.args.lam_div,
                lam_cnrst=self.args.lam_cnrst,
                class_weights=self.class_weights
            )
        else:  # Feature extractor training
            self.criterion = torch.nn.BCEWithLogitsLoss(pos_weight=self.class_weights)
    
        self.train_preds, self.train_labels = [], []
        self.val_preds, self.val_labels = [], []
        self.test_preds = []
        self.test_labels = []
        self.test_probs = []
        
        # For EchoNext, get class names from dataset if available
        if args.training_stage == 'echonext_adapt' and hasattr(args, 'dataset_root'):
            # Try to get from train loader if available (will be set later)
            self.class_names = None  # Will be set from dataset
            # Get num_classes from model
            if hasattr(model, 'num_classes'):
                self.num_classes = model.num_classes
            else:
                # Fallback: try to infer from model output
                self.num_classes = None
        else:
            self.class_names = get_class_names(args)
            self.num_classes = len(self.class_names) if self.class_names else None
        self.save_hyperparameters(ignore=["model"]) 
    
    def compute_l1_loss(self):
        if hasattr(self.model, "prototype_class_identity"):
            l1_mask = 1 - torch.t(self.model.prototype_class_identity).to(self.device)
            l1_penalty = (self.model.classifier.weight * l1_mask).norm(p=1)
            #print(f"[DEBUG] L1 penalty: {l1_penalty.item()}")
            return l1_penalty
        return torch.norm(self.model.classifier.weight, p=1)

    def forward(self, x):
        output = self.model(x)  # Prototype models return both logits and features
        if self.args.training_stage in ["prototypes", "joint"]:
            logits, distances, similarity_scores, = output
        else:
            logits = output
            distances = None
            similarity_scores = None 

        return logits, distances, similarity_scores
    
    def training_step(self, batch):
        x, y = batch
        if isinstance(y, dict):
            y = y["full"] # extract full labels for fusion classifier training

        output = self.model(x)
        
        if self.args.training_stage in ["feature_extractor", "fusion"]:
            logits = output  # Feature extractor only returns logits
            similarity_scores = None
        
        elif self.args.training_stage in ["prototypes", "joint"]:
            logits, _, similarity_scores = output  # Extract similarity scores from the correct index
        elif self.args.training_stage == "echonext_adapt":
            # EchoNext adaptation: always returns (logits, distances, similarity_scores)
            logits, _, similarity_scores = output
        
        elif self.args.training_stage == "classifier":
            logits, _, _ = output  # Classifier only returns logits
            similarity_scores = None

        # Loss computation based on training stage
        if self.args.training_stage == "echonext_adapt" and self.args.train_head_only:
            # Head-only: only BCE loss
            loss = self.criterion(logits, y)
        elif self.args.training_stage == "echonext_adapt" and not self.args.train_head_only:
            # Joint fine-tuning: BCE + masked prototype losses
            loss = self.criterion(logits, y, self.model, similarity_scores)
        else:
            # Standard PTB training
            loss = self.criterion(logits, y, self.model, similarity_scores) if self.is_proto else self.criterion(logits, y)
        
        if torch.isnan(logits).any() or torch.isinf(logits).any():
            print("[DEBUG] Detected NaNs or Infs in logits!")
            print("[DEBUG] Logit stats:", logits.min().item(), logits.max().item(), logits.mean().item())

        probs = torch.sigmoid(logits).detach().cpu().numpy()
        self.train_preds.append(probs)
        self.train_labels.append(y.detach().cpu().numpy())
        self.log('train_loss', loss, prog_bar=True, sync_dist=True)

        return loss

    def on_train_epoch_end(self):
        if len(self.train_preds) == 0:
            return  # Avoid computing if no data

        all_preds = np.concatenate(self.train_preds, axis=0)
        all_labels = np.concatenate(self.train_labels, axis=0)

        auc = self._compute_auc_epoch(all_preds, all_labels)
        f1 = f1_score(all_labels, (all_preds > 0.5).astype(int), average='micro')

        self.log("train_auc", auc, prog_bar=True, sync_dist=True)
        self.log("train_f1", f1, prog_bar=True, sync_dist=True)

        self.train_preds, self.train_labels = [], []  # Reset storage

    def validation_step(self, batch):
        x, y = batch
        if isinstance(y, dict):
            y = y["full"] # extract full labels for fusion classifier training
        output = self.model(x)
        
        if self.args.training_stage in ["feature_extractor", "fusion"]:
            logits = output  # Feature extractor only returns logits
            similarity_scores = None
        
        elif self.args.training_stage in ["prototypes", "joint"]:
            logits, _, similarity_scores = output  # Extract similarity scores from the correct index
        elif self.args.training_stage == "echonext_adapt":
            # EchoNext adaptation: always returns (logits, distances, similarity_scores)
            logits, _, similarity_scores = output
        
        elif self.args.training_stage == "classifier":
            logits, _, _ = output  # Classifier only returns logits
            similarity_scores = None
            
        # Loss computation based on training stage
        if self.args.training_stage == "echonext_adapt" and self.args.train_head_only:
            # Head-only: only BCE loss
            loss = self.criterion(logits, y)
        elif self.args.training_stage == "echonext_adapt" and not self.args.train_head_only:
            # Joint fine-tuning: BCE + masked prototype losses
            loss = self.criterion(logits, y, self.model, similarity_scores)
        else:
            # Standard PTB training
            loss = self.criterion(logits, y, self.model, similarity_scores) if self.is_proto else self.criterion(logits, y)
        
        probs = torch.sigmoid(logits).detach().cpu().numpy()
        self.val_preds.append(probs)
        self.val_labels.append(y.detach().cpu().numpy())
        self.log('val_loss', loss, prog_bar=True, sync_dist=True)

        return loss
    
    def on_validation_epoch_end(self):
        if len(self.val_preds) == 0:
            return  # Avoid computing if no data

        all_preds = np.concatenate(self.val_preds, axis=0)
        all_labels = np.concatenate(self.val_labels, axis=0)

        auc = self._compute_auc_epoch(all_preds, all_labels)
        f1 = f1_score(all_labels, (all_preds > 0.5).astype(int), average='micro')
        
        # Compute AUROC micro/macro and PR-AUC for EchoNext adaptation
        auroc_micro = roc_auc_score(all_labels, all_preds, average='micro')
        # Handle classes with only one label (skip them for macro average)
        try:
            auroc_macro = roc_auc_score(all_labels, all_preds, average='macro')
        except ValueError as e:
            if "Only one class present" in str(e):
                # Compute per-class and skip classes with only one label
                auroc_per_class = []
                for c in range(all_labels.shape[1]):
                    if len(np.unique(all_labels[:, c])) > 1:
                        try:
                            auroc_c = roc_auc_score(all_labels[:, c], all_preds[:, c])
                            auroc_per_class.append(auroc_c)
                        except:
                            pass
                auroc_macro = np.mean(auroc_per_class) if auroc_per_class else 0.0
            else:
                raise
        pr_auc = average_precision_score(all_labels, all_preds, average='micro')

        # Rename val_auc to val_auroc_macro for consistency
        self.log("val_auroc_macro", auroc_macro, prog_bar=True, sync_dist=True)
        self.log("val_f1", f1, prog_bar=True, sync_dist=True)
        self.log("val_auroc_micro", auroc_micro, prog_bar=True, sync_dist=True)
        self.log("val_auprc_micro", pr_auc, prog_bar=False, sync_dist=True)

        self.val_preds, self.val_labels = [], []  # Reset storage

    def test_step(self, batch):
        x, y = batch
        if isinstance(y, dict):
            y = y["full"] # extract full labels for fusion classifier training

        # Run inference
        output = self.model(x)
        
        if self.args.training_stage in ["feature_extractor", "fusion"]:
            logits = output  # Feature extractor only returns logits
            similarity_scores = None
        
        elif self.args.training_stage in ["prototypes", "joint"]:
            logits, _, similarity_scores = output  # Extract similarity scores from the correct index
        elif self.args.training_stage == "echonext_adapt":
            # EchoNext adaptation: always returns (logits, distances, similarity_scores)
            logits, _, similarity_scores = output
        
        elif self.args.training_stage == "classifier":
            logits, _, _ = output  # Classifier only returns logits
            similarity_scores = None
            
        # For head-only training, criterion only takes logits and y
        if self.args.training_stage == "echonext_adapt" and self.args.train_head_only:
            loss = self.criterion(logits, y)
        else:
            loss = self.criterion(logits, y, self.model, similarity_scores) if self.is_proto else self.criterion(logits, y)

        auc = self._compute_auc(logits, y)
        f1 = self._compute_f1(logits, y)
        fmax, _ = self._compute_fmax(logits, y)

        # Convert logits to probabilities and binary predictions
        probs = torch.sigmoid(logits)
        preds = (probs > 0.5).int()

        # Log values
        self.log('test_loss', loss, prog_bar=True, sync_dist=True)
        self.log('test_auc', auc, prog_bar=True, sync_dist=True)
        self.log('test_f1', f1, prog_bar=True, sync_dist=True)
        self.log('test_fmax', fmax, prog_bar=True, sync_dist=True)

        # Store predictions & labels for later analysis
        self.test_preds.append(preds.cpu().numpy())
        self.test_labels.append(y.cpu().numpy())
        self.test_probs.append(probs.cpu().numpy())

        return {
            'test_loss': loss, 
            'test_auc': auc, 
            'test_f1': f1,
            'test_fmax': f1,
        }

    def on_test_epoch_end(self):
        """ Computes full test set confusion matrix & logs it to TensorBoard. """
        all_preds = np.concatenate(self.test_preds, axis=0)
        all_labels = np.concatenate(self.test_labels, axis=0)
        all_probs = np.concatenate(self.test_probs, axis=0)

        # Ensure shape consistency
        assert all_preds.shape == all_labels.shape == all_probs.shape, "Shape mismatch in test outputs"
        
        # Prepare DataFrame
        num_classes = all_preds.shape[1]
        df_dict = {"ID": np.arange(len(all_preds))} 

        # Handle EchoNext case where class_names might be None
        if self.class_names is not None:
            class_labels = self.class_names
        else:
            # Fallback to numeric indices
            class_labels = [f"class_{i}" for i in range(num_classes)]

        for i, class_name in enumerate(class_labels):
            df_dict[f"Label_{class_name}"] = all_labels[:, i]
            df_dict[f"Pred_{class_name}"] = all_preds[:, i]
            df_dict[f"Prob_{class_name}"] = all_probs[:, i]

        # Save as CSV file
        version = self.logger.version if self.logger else '0'
        save_dir = os.path.join(self.args.test_dir, self.args.job_name)
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, f"{self.args.job_name}_test_results_v{version}.csv")
        df = pd.DataFrame(df_dict)
        df.to_csv(save_path, index=False)
        print(f" Saved test predictions to {save_path}")
        
        self.test_preds = []
        self.test_labels = []
        self.test_probs = []

        cm = multilabel_confusion_matrix(all_labels, all_preds)

        # Log confusion matrix, ROC, and PR curves to TensorBoard
        if self.logger:
            fig_cm = self.plot_confusion_matrix(cm)
            fig_roc = self.plot_roc_curve(all_labels, all_probs)
            fig_pr = self.plot_pr_curve(all_labels, all_probs)

            self.logger.experiment.add_figure("Test/Confusion_Matrix", fig_cm, global_step=self.current_epoch)
            self.logger.experiment.add_figure("Test/ROC_Curve", fig_roc, global_step=self.current_epoch)
            self.logger.experiment.add_figure("Test/PR_Curve", fig_pr, global_step=self.current_epoch)

    def plot_confusion_matrix(self, cm):
        num_classes = cm.shape[0]
        fig, axes = plt.subplots(1, num_classes, figsize=(15, 5))
        for i, class_name in enumerate(self.class_names):
            sns.heatmap(cm[i], annot=True, fmt="d", cmap="Blues", ax=axes[i])
            axes[i].set_title(class_name)
            axes[i].set_xlabel("Predicted")
            axes[i].set_ylabel("True")
        plt.tight_layout()
        return fig

    def plot_roc_curve(self, y_true, y_probs):
        num_classes = y_true.shape[1]
        fig, ax = plt.subplots(figsize=(8, 6))
        for i, class_name in enumerate(self.class_names):
            fpr, tpr, _ = roc_curve(y_true[:, i], y_probs[:, i])
            auc_score = roc_auc_score(y_true[:, i], y_probs[:, i])
            ax.plot(fpr, tpr, label=f'{class_name} (AUC = {auc_score:.2f})')
        ax.plot([0, 1], [0, 1], 'k--')
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_title("ROC Curves")
        ax.legend(loc="lower right")
        plt.tight_layout()
        return fig

    def plot_pr_curve(self, y_true, y_probs):
        num_classes = y_true.shape[1]
        fig, ax = plt.subplots(figsize=(8, 6))
        for i, class_name in enumerate(self.class_names):
            precision, recall, _ = precision_recall_curve(y_true[:, i], y_probs[:, i])
            pr_auc = average_precision_score(y_true[:, i], y_probs[:, i])
            ax.plot(recall, precision, label=f'{class_name} (AP = {pr_auc:.2f})')
        ax.set_xlabel("Recall")
        ax.set_ylabel("Precision")
        ax.set_title("Precision-Recall Curves")
        ax.legend(loc="upper right")
        plt.tight_layout()
        return fig
    
    def configure_optimizers(self):
        # Handle freezing for EchoNext adaptation
        # Note: self.model is the ProtoECGNet model, not wrapped
        if hasattr(self.args, 'freeze_encoder') and self.args.freeze_encoder:
            if isinstance(self.model, (ProtoECGNet1D, ProtoECGNet2D)):
                for param in self.model.feature_extractor.parameters():
                    param.requires_grad = False
                print("[VERIFY] Encoder frozen: feature_extractor parameters set to requires_grad=False")
        
        if hasattr(self.args, 'freeze_prototypes') and self.args.freeze_prototypes:
            if isinstance(self.model, (ProtoECGNet1D, ProtoECGNet2D)):
                self.model.prototype_vectors.requires_grad = False
                print("[VERIFY] Prototypes frozen: prototype_vectors set to requires_grad=False")
        
        # Verify freeze/unfreeze state
        if self.args.training_stage == 'echonext_adapt':
            encoder_frozen = all(not p.requires_grad for p in self.model.feature_extractor.parameters())
            prototypes_frozen = not self.model.prototype_vectors.requires_grad if hasattr(self.model, 'prototype_vectors') else True
            head_trainable = any(p.requires_grad for p in self.model.classifier.parameters())
            
            if self.args.train_head_only:
                assert encoder_frozen, "[ERROR] Encoder should be frozen in head-only training!"
                assert prototypes_frozen, "[ERROR] Prototypes should be frozen in head-only training!"
                assert head_trainable, "[ERROR] Head should be trainable in head-only training!"
                print("[VERIFY] Head-only training: ✓ Encoder frozen, ✓ Prototypes frozen, ✓ Head trainable")
            else:
                # Joint fine-tuning: encoder frozen, prototypes+head trainable
                assert encoder_frozen, "[ERROR] Encoder should be frozen in joint fine-tuning!"
                assert not prototypes_frozen, "[ERROR] Prototypes should be trainable in joint fine-tuning!"
                assert head_trainable, "[ERROR] Head should be trainable in joint fine-tuning!"
                print("[VERIFY] Joint fine-tuning: ✓ Encoder frozen, ✓ Prototypes trainable, ✓ Head trainable")
        
        # Collect trainable parameters
        if self.args.training_stage == "prototypes":
            trainable_params = [self.model.prototype_vectors]
        elif self.args.training_stage in ["classifier", "fusion"]:
            trainable_params = list(self.model.classifier.parameters())
        elif self.args.training_stage == "echonext_adapt":
            # For EchoNext adaptation, only train unfrozen parameters
            trainable_params = []
            if isinstance(self.model, (ProtoECGNet1D, ProtoECGNet2D)):
                if not (hasattr(self.args, 'freeze_encoder') and self.args.freeze_encoder):
                    trainable_params.extend(self.model.feature_extractor.parameters())
                if not (hasattr(self.args, 'freeze_prototypes') and self.args.freeze_prototypes):
                    trainable_params.append(self.model.prototype_vectors)
                # Always train classifier
                trainable_params.extend(self.model.classifier.parameters())
            else:
                trainable_params = list(self.model.parameters())
        else:  # Feature extractor and joint training
            trainable_params = list(self.model.parameters())
        
        optimizer = torch.optim.Adam(trainable_params, lr=self.lr, weight_decay=self.l2)

        # Apply scheduler based on the chosen type
        if self.args.scheduler_type == "ReduceLROnPlateau":
            # For EchoNext, monitor val_auroc_micro if available, else val_loss
            monitor_metric = "val_loss"
            if self.args.training_stage == 'echonext_adapt':
                # Try to use val_auroc_micro for ReduceLROnPlateau if available
                monitor_metric = "val_auroc_micro"
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, 
                mode="max" if monitor_metric.startswith("val_auroc") else "min",
                factor=0.5, 
                patience=2,
                min_lr=1e-6,
                verbose=True
            )
            return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler, "monitor": monitor_metric}}
        elif self.args.scheduler_type == "CosineAnnealingLR":
            eta_min = getattr(self.args, 'scheduler_eta_min', 1e-6)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.args.epochs, eta_min=eta_min)
            return {"optimizer": optimizer, "lr_scheduler": scheduler}
        elif self.args.scheduler_type == "CyclicLR":
            base_lr = self.lr * 0.1  # Adjusting base_lr relative to the initial lr
            max_lr = self.lr * 10    # Allowing dynamic cycling of lr
            scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr=base_lr, max_lr=max_lr, step_size_up=2000, mode="triangular")
            return {"optimizer": optimizer, "lr_scheduler": scheduler}
        else:
            return optimizer  # No scheduler if none is chosen

    def _compute_auc(self, logits, y):
        preds = torch.sigmoid(logits).detach().cpu().numpy()
        targets = y.detach().cpu().numpy()

        aucs = []
        for i in range(targets.shape[1]):
            y_true = targets[:, i]
            y_pred = preds[:, i]
            if np.all(y_true == 0) or np.all(y_true == 1):
                continue  # Skip invalid class
            try:
                auc = roc_auc_score(y_true, y_pred)
                aucs.append(auc)
            except Exception as e:
                print(f"[WARNING] Skipping AUC for class {i}: {e}")
                continue
        return np.mean(aucs) if aucs else float('nan')
    
    def _compute_auc_epoch(self, preds, labels):
        """ Computes average AUC across valid classes (ignores all-0 or all-1 classes). """
        aucs = []
        for i in range(labels.shape[1]):
            y_true = labels[:, i]
            y_pred = preds[:, i]
            # Skip if only one class is present (e.g., all 0s or all 1s)
            if np.all(y_true == 0) or np.all(y_true == 1):
                continue
            try:
                auc = roc_auc_score(y_true, y_pred)
                aucs.append(auc)
            except Exception as e:
                print(f"[WARNING] Skipping AUC for class {i}: {e}")
                continue
        return np.mean(aucs) if aucs else float('nan')


    def _compute_f1(self, logits, y):
        preds = (torch.sigmoid(logits) > 0.5).int().detach().cpu().numpy()
        targets = y.detach().cpu().numpy()
        return f1_score(targets, preds, average='micro')

    def _compute_fmax(self, logits, y, thresholds=np.arange(0.0, 1.01, 0.01)):
        probs = torch.sigmoid(logits).detach().cpu().numpy()
        targets = y.detach().cpu().numpy()

        best_f1 = 0.0
        best_threshold = 0.5  

        for t in thresholds:
            preds = (probs > t).astype(int)
            f1 = f1_score(targets, preds, average='micro')
            if f1 > best_f1:
                best_f1 = f1
                best_threshold = t

        return best_f1, best_threshold  


def train_model(base_model, train_loader, val_loader, args, class_weights, trainer=None):
    # For EchoNext, set class names from dataset
    if args.training_stage == 'echonext_adapt' and hasattr(train_loader.dataset, 'class_names'):
        # Temporarily set class names in args for ECGTrainer
        args._echonext_class_names = train_loader.dataset.class_names
        args._echonext_num_classes = len(train_loader.dataset.class_names)
    
    # Extract ASL parameters from args
    loss_name = getattr(args, 'loss', 'bce')
    asl_gamma_pos = getattr(args, 'asl_gamma_pos', 0.0)
    asl_gamma_neg = getattr(args, 'asl_gamma_neg', 4.0)
    asl_clip = getattr(args, 'asl_clip', 0.05)
    asl_disable_pos_weight = getattr(args, 'asl_disable_pos_weight', False)
    use_class_weights = getattr(args, 'use_class_weights', True)
    
    model = ECGTrainer(
        base_model, 
        lr=args.lr, 
        l2=args.l2, 
        args=args, 
        class_weights=class_weights,
        loss_name=loss_name,
        asl_gamma_pos=asl_gamma_pos,
        asl_gamma_neg=asl_gamma_neg,
        asl_clip=asl_clip,
        asl_disable_pos_weight=asl_disable_pos_weight,
        use_class_weights=use_class_weights,
    )
    
    # Set class names from dataset if available
    if args.training_stage == 'echonext_adapt' and hasattr(args, '_echonext_class_names'):
        model.class_names = args._echonext_class_names
        model.num_classes = args._echonext_num_classes
        print(f"[INFO] Set class names from EchoNext dataset: {len(model.class_names)} classes")

    print(f"Starting training for job: {args.job_name}")
    print("Initializing TensorBoard Logger...")

    logger = TensorBoardLogger(args.log_dir, name=args.job_name)
    
    # Configure early stopping based on args
    early_stop_metric = getattr(args, 'early_stop_metric', 'val_loss')
    early_stop_patience = getattr(args, 'early_stop_patience', args.patience)
    
    # Determine checkpoint monitor metric
    if args.training_stage == 'echonext_adapt':
        # For EchoNext, use auroc_micro as default
        checkpoint_monitor = 'val_auroc_micro' if early_stop_metric == 'auroc_micro' else 'val_auroc_macro'
        checkpoint_filename = '{epoch}-{val_auroc_micro:.4f}' if early_stop_metric == 'auroc_micro' else '{epoch}-{val_auroc_macro:.4f}'
    else:
        # For PTB, use val_auroc_macro (renamed from val_auc)
        checkpoint_monitor = 'val_auroc_macro'
        checkpoint_filename = '{epoch}-{val_auroc_macro:.4f}'
    
    checkpoint_callback = ModelCheckpoint(
        dirpath=os.path.join(args.checkpoint_dir, args.job_name),
        filename=checkpoint_filename,
        monitor=checkpoint_monitor,
        mode='max',
        save_top_k=args.save_top_k,
        save_last=True
    )
    
    # Optional: Add second checkpoint for best macro AUROC (if training EchoNext and save_both_checkpoints is True)
    checkpoint_callback_macro = None
    if args.training_stage == 'echonext_adapt' and getattr(args, 'save_both_checkpoints', False):
        checkpoint_callback_macro = ModelCheckpoint(
            dirpath=os.path.join(args.checkpoint_dir, args.job_name, 'best_macro'),
            filename='epoch={epoch}-val_auroc_macro={val_auroc_macro:.4f}',
            monitor='val_auroc_macro',
            mode='max',
            save_top_k=1,
            save_last=False
        )
    
    if early_stop_patience is not None and early_stop_patience > 0:
        if early_stop_metric in ['auroc_micro', 'auroc_macro', 'pr_auc', 'auprc_micro']:
            early_stop_callback = EarlyStopping(
                monitor=f'val_{early_stop_metric}',
                patience=early_stop_patience,
                mode='max'
            )
        else:
            early_stop_callback = EarlyStopping(
                monitor=early_stop_metric,
                patience=early_stop_patience,
                mode='min' if 'loss' in early_stop_metric else 'max'
            )
    else:
        # Fallback: use auroc_macro for PTB, auroc_micro for EchoNext
        fallback_metric = 'val_auroc_micro' if args.training_stage == 'echonext_adapt' else 'val_auroc_macro'
        early_stop_callback = EarlyStopping(monitor=fallback_metric, patience=args.patience, mode='max')
    lr_monitor = LearningRateMonitor(logging_interval='step')
    print("Initializing PyTorch Lightning Trainer...")

    if args.resume_checkpoint:
        print(f"Resuming from checkpoint...")
    else: 
        print(f"Not using a checkpoint...")

    # Use the provided trainer if given, otherwise create a new one 
    if trainer is None:
        # For EchoNext with LAS sampler, disable distributed sampler injection
        # since LabelAwareBatchSampler is a custom batch sampler
        use_distributed_sampler = not (args.training_stage == 'echonext_adapt' and 
                                      hasattr(args, 'sampler_mode') and 
                                      args.sampler_mode == 'las')
        callbacks_list = [checkpoint_callback, early_stop_callback, lr_monitor]
        if checkpoint_callback_macro is not None:
            callbacks_list.append(checkpoint_callback_macro)
        
        trainer = pl.Trainer(
            max_epochs=args.epochs,
            accelerator="gpu" if torch.cuda.is_available() else "cpu",
            devices="auto",  
            strategy="auto",
            logger=logger,
            callbacks=callbacks_list,
            use_distributed_sampler=use_distributed_sampler,
        )
   
    if args.training_stage == "feature_extractor":
        print("Training feature extractor...")
        trainer.fit(model, train_loader, val_loader, ckpt_path="last" if args.resume_checkpoint else None)
        #save_model_weights(model, args.job_name, "feature_extractor", args.checkpoint_dir, args.save_weights)
    
    elif args.training_stage == "prototypes":
        print("Training add-on layers and prototype layer...")
        for param in model.model.feature_extractor.parameters():
            param.requires_grad = False  # Freeze feature extractor
        for param in model.model.classifier.parameters():
            param.requires_grad = False  # Freeze classifier (only train prototype layer)
        trainer.fit(model, train_loader, val_loader, ckpt_path="last" if args.resume_checkpoint else None)
       #save_model_weights(model, args.job_name, "prototypes", args.checkpoint_dir, args.save_weights)

    elif args.training_stage == "joint":
        print("Joint training of feature extractor, add-on layers, and prototype layer (without classifier)...")
        for param in model.model.feature_extractor.parameters():
            param.requires_grad = True  # Unfreeze feature extractor
        for param in model.model.classifier.parameters():
            param.requires_grad = False  # Freeze classifier 
        trainer.fit(model, train_loader, val_loader, ckpt_path="last" if args.resume_checkpoint else None)
        #save_model_weights(model, args.job_name, "joint", args.checkpoint_dir, args.save_weights)
    
    elif args.training_stage == "projection":
        print("Performing prototype projection...")
        proj_dir = os.path.join(args.checkpoint_dir, args.job_name)
        os.makedirs(proj_dir, exist_ok=True)
        if args.dimension == "1D":
            push_prototypes1d(model.model, train_loader, proj_dir, args.label_set, args.job_name, logger, device=args.device, custom_groups=args.custom_groups)
        elif args.dimension == "2D": 
            push_prototypes2d(model.model, train_loader, proj_dir, args.label_set, args.job_name, logger, device=args.device, custom_groups=args.custom_groups)
        save_model_weights(model, args.job_name, "projection", args.checkpoint_dir, args.save_weights)

    elif args.training_stage == "echonext_adapt":
        print("EchoNext adaptation training...")
        # Freezing is handled in configure_optimizers and main.py
        trainer.fit(model, train_loader, val_loader, ckpt_path="last" if args.resume_checkpoint else None)
    
    elif args.training_stage == "classifier":
        print("Fine-tuning classifier (classifier-only training)...")
        for param in model.model.feature_extractor.parameters():
            param.requires_grad = False #Freeze feature extractor
        for param in model.model.add_on_layers.parameters(): 
            param.requires_grad = False #Freeze add-on layers
        with torch.no_grad(): #Freeze prototype layer
            model.model.prototype_vectors = torch.nn.Parameter(model.model.prototype_vectors.detach(), requires_grad=False)
        for param in model.model.classifier.parameters():
            param.requires_grad = True  # Only train classifier
        trainer.fit(model, train_loader, val_loader, ckpt_path="last" if args.resume_checkpoint else None)
        #save_model_weights(model, args.job_name, "classifier", args.checkpoint_dir, args.save_weights) # No need to call since already saving in trainer.fit()
    elif args.training_stage == "fusion":
        print("Training fusion classifier using similarity scores from pretrained models...")

        # Freeze all submodels
        for submodel in [model.model.model1d, model.model.model2d_partial, model.model.model2d_global]:
            submodel.eval()
            for param in submodel.parameters():
                param.requires_grad = False

        # Freeze their prototype vectors explicitly too
        with torch.no_grad():
            model.model.model1d.prototype_vectors = torch.nn.Parameter(
                model.model.model1d.prototype_vectors.detach(), requires_grad=False)
            model.model.model2d_partial.prototype_vectors = torch.nn.Parameter(
                model.model.model2d_partial.prototype_vectors.detach(), requires_grad=False)
            model.model.model2d_global.prototype_vectors = torch.nn.Parameter(
                model.model.model2d_global.prototype_vectors.detach(), requires_grad=False)

        # Ensure classifier head is trainable
        for param in model.model.classifier.parameters():
            param.requires_grad = True

        # Fit the fusion classifier
        trainer.fit(model, train_loader, val_loader, ckpt_path="last" if args.resume_checkpoint else None)

    else:
        raise ValueError("Invalid training stage specified")
    
    hparams = vars(args)
    final_metrics = trainer.callback_metrics
    logger.log_hyperparams(hparams, final_metrics)

    return trainer

def test_model(base_model, test_loader, args, device="cuda", trainer=None):
    model = ECGTrainer(base_model, args=args).to(device)
    logger = TensorBoardLogger(args.log_dir, name=args.job_name)

    if trainer is None:
        trainer = pl.Trainer(
            accelerator="gpu" if torch.cuda.is_available() else "cpu",
            devices="auto",
            logger=logger 
        )
        print("\n--- Evaluating on Test Set ---")
        test_results = trainer.test(model, dataloaders=test_loader)

    else:
        best_ckpt_path = None
        for callback in trainer.callbacks:
            if isinstance(callback, pl.callbacks.ModelCheckpoint) and callback.best_model_path:
                best_ckpt_path = callback.best_model_path
                break
        
        if best_ckpt_path and os.path.exists(best_ckpt_path):
            print("\n--- Evaluating Best Epoch Weights on Test Set ---")
            state_dict = torch.load(best_ckpt_path, map_location=device, weights_only=False)["state_dict"]
            model.load_state_dict(state_dict, strict=False) 
            test_results = trainer.test(model, dataloaders=test_loader)
        else:
            print("\n--- No Best Checkpoint Found, Using Current Model Weights ---")
            test_results = trainer.test(model, dataloaders=test_loader)

    print(f"Test Metrics: {test_results}")

    return test_results  

