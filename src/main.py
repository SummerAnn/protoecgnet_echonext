if __name__ == '__main__':
    import argparse
    import torch
    from training_functions import train_model, test_model, seed_everything, ECGTrainer
    from ecg_utils import get_dataloaders, load_label_mappings
    from backbones import (
        resnet1d18, resnet1d34, resnet1d50, resnet1d101, resnet1d152,  
        resnet18, resnet34, resnet50, resnet101, resnet152
    )
    from proto_models1D import ProtoECGNet1D
    from proto_models2D import ProtoECGNet2D
    from fusion import FusionProtoClassifier, load_fusion_label_mappings, get_fusion_dataloaders

    torch.set_float32_matmul_precision("high")

    def str2bool(v):
        if isinstance(v, bool):
            return v
        if v.lower() in ('yes', 'true', 't', 'y', '1'):
            return True
        elif v.lower() in ('no', 'false', 'f', 'n', '0'):
            return False
        else:
            raise argparse.ArgumentTypeError('Boolean value expected.')

    # Argument parser
    parser = argparse.ArgumentParser()
    parser.add_argument('--job_name', type=str, required=True)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--l2', type=float, default=0)
    parser.add_argument('--l1', type=float, default=1e-4)
    parser.add_argument('--lam_clst', type=float, default=0.004)
    parser.add_argument('--lam_sep', type=float, default=0.0004)
    parser.add_argument('--lam_spars', type=float, default=0)
    parser.add_argument('--lam_div', type=float, default=250)
    parser.add_argument('--lam_cnrst', type=float, default=300)
    parser.add_argument('--proto_dim', type=int, default=512, help='Channel dimension of prototype vectors')
    parser.add_argument('--proto_time_len', type=int, default=3, help='Time dimension of prototype vectors')
    parser.add_argument('--prototype_activation_function', type=str, choices=['log', 'linear'], default='log')
    parser.add_argument('--latent_space_type', type=str, choices=['l2', 'arc'], default='arc')
    parser.add_argument('--add_on_layers_type', type=str, choices=['linear', 'identity', 'other'], default='linear')
    parser.add_argument('--class_specific', type=str2bool, default=True, help='Whether to use class-specific prototypes')
    parser.add_argument('--last_layer_connection_weight', type=float, default=1.0, help='Weight of last layer connection in the model')
    parser.add_argument('--m', type=float, default=0.05, help='Margin for prototype learning') #unused in our paper
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu", type=str, help="Device to use for projection (cuda or cpu)")
    parser.add_argument('--dropout', type=float, default=0)
    parser.add_argument('--checkpoint_dir', type=str, default='/gpfs/data/bbj-lab/users/sethis/experiments/checkpoints')
    parser.add_argument('--log_dir', type=str, default='/gpfs/data/bbj-lab/users/sethis/experiments/logs')
    parser.add_argument('--test_dir', type=str, default='/gpfs/data/bbj-lab/users/sethis/experiments/test_results')
    parser.add_argument('--save_top_k', type=int, default=3)
    parser.add_argument('--patience', type=int, default=10)
    parser.add_argument('--resume_checkpoint', type=str2bool, default=None)
    parser.add_argument('--use_class_weights', type=str2bool, default=True)
    parser.add_argument('--pretrained_weights', type=str, default=None, help='Path to pretrained model weights')
    parser.add_argument('--training_stage', type=str, choices=['feature_extractor', 'prototypes', 'joint', 'projection', 'classifier', 'fusion', 'echonext_adapt'], required=True)
    parser.add_argument('--dimension', type=str, choices=['1D', '2D'], required=True, help='Specify whether the model is 1D or 2D')
    parser.add_argument('--backbone', type=str, choices=[
        'resnet1d18', 'resnet1d34', 'resnet1d50', 'resnet1d101', 'resnet1d152',
        'resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152',
    ], required=True, help='Specify the backbone architecture')
    parser.add_argument('--single_class_prototype_per_class', type=int, default=5)
    parser.add_argument('--joint_prototypes_per_border', type=int, default=0) #not used in our paper
    parser.add_argument('--sampling_rate', type=int, choices=[100, 500], required=True) #we use 100 Hz
    parser.add_argument('--label_set', type=str, choices=['superdiagnostic', 'subdiagnostic', 'all', 'diagnostic', 'form', 'rhythm', '1', '2', '3', '4'], default='superdiagnostic')
    parser.add_argument('--test_model', type=str2bool, default=False, help='Flag to of whether to skip training and just test the model')
    parser.add_argument('--save_weights', type=str2bool, default=True, help='Flag to save model weights after training')
    parser.add_argument('--custom_groups', type=str2bool, default=False, help='Flag to use custom label groupings')
    parser.add_argument('--standardize', type=str2bool, default=False, help='Whether to standardize input ECG signals')
    parser.add_argument('--remove_baseline', type=str2bool, default=True, help='Whether to remove baseline wander from input ECG signals (high-pass filter)')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
    parser.add_argument('--num_workers', type=int, default=0, help='Number of workers for dataloader')

    parser.add_argument('--scheduler_type', type=str, choices=['ReduceLROnPlateau', 'CosineAnnealingLR', 'CyclicLR'],
                    default='ReduceLROnPlateau', help="Type of learning rate scheduler to use")
    parser.add_argument('--scheduler_eta_min', type=float, default=1e-6, help="Minimum LR for CosineAnnealingLR (default: 1e-6)")
    parser.add_argument('--save_both_checkpoints', type=str2bool, default=False, help="Save both best-micro and best-macro checkpoints (EchoNext only)")

    # EchoNext adaptation flags
    parser.add_argument('--dataset_root', type=str, default=None, help='Root directory for dataset (EchoNext or PTB-XL)')
    parser.add_argument('--train_split', type=str, default='train', help='Training split name (train/val/test)')
    parser.add_argument('--val_split', type=str, default='val', help='Validation split name (train/val/test)')
    parser.add_argument('--out_dir', type=str, default=None, help='Output directory for checkpoints and logs')
    parser.add_argument('--freeze_encoder', type=str2bool, default=False, help='Freeze encoder parameters (no gradients)')
    parser.add_argument('--freeze_prototypes', type=str2bool, default=False, help='Freeze prototype parameters (no gradients)')
    parser.add_argument('--train_head_only', type=str2bool, default=False, help='Train only classifier head (freezes encoder and prototypes)')
    parser.add_argument('--lambda_cluster', type=float, default=None, help='Override lam_clst for cluster loss (EchoNext adaptation)')
    parser.add_argument('--lambda_separation', type=float, default=None, help='Override lam_sep for separation loss (EchoNext adaptation)')
    parser.add_argument('--topk_push', type=int, default=5, help='Top-K prototypes to push during training')
    parser.add_argument('--export_encoder_ts', type=str, default=None, help='Export encoder to TorchScript at this path')
    parser.add_argument('--early_stop_metric', type=str, default='val_loss', choices=['val_loss', 'auroc_micro', 'auroc_macro', 'pr_auc'], help='Metric for early stopping')
    parser.add_argument('--early_stop_patience', type=int, default=None, help='Patience for early stopping (None to disable)')
    parser.add_argument('--ckpt', type=str, default=None, help='Checkpoint path to load (alias for --pretrained_weights)')
    parser.add_argument('--weight_clip_max', type=float, default=5.0, help='Maximum value for class weights (PR5-A: gentler weighting)')
    parser.add_argument('--use_inverse_sqrt', type=str2bool, default=False, help='Use inverse-sqrt weighting instead of inverse-freq (PR5-A)')
    parser.add_argument('--sampler_mode', type=str, default='none', choices=['none', 'las', 'weighted'], help='Batch sampler mode (PR5-B: las=Label-Aware Sampler)')
    parser.add_argument('--sampler_gamma', type=float, default=0.5, help='Sampler gamma parameter (PR5-B: higher = more bias toward rare classes)')
    parser.add_argument('--iters_per_epoch', type=int, default=None, help='Iterations per epoch for LAS (PR5-B: default=len(train)//batch_size)')
    # --- Loss selection & ASL knobs (PR5-C) ---
    parser.add_argument("--loss", type=str, choices=["bce", "asl"], default="bce", help="Classification loss: bce or asl (PR5-C)")
    parser.add_argument("--asl_gamma_pos", type=float, default=0.0, help="ASL gamma for positives (PR5-C)")
    parser.add_argument("--asl_gamma_neg", type=float, default=4.0, help="ASL gamma for negatives (PR5-C)")
    parser.add_argument("--asl_clip", type=float, default=0.05, help="ASL probability clip for negatives (PR5-C)")
    parser.add_argument("--asl_disable_pos_weight", action="store_true", help="When using ASL, ignore pos_weight to avoid over-corrections (PR5-C)")

    # Fusion classifier-specific arguments
    parser.add_argument('--fusion_weights1', type=str, default='/gpfs/data/bbj-lab/users/sethis/experiments/checkpoints/cat1_new_redo1_proj1/cat1_new_redo1_proj1_projection.pth', help='Path to pretrained model weights for category 1')
    parser.add_argument('--fusion_weights3', type=str, default='/gpfs/data/bbj-lab/users/sethis/experiments/checkpoints/cat3_2D_redo1_tunejoint1_trial96_proj/cat3_2D_redo1_tunejoint1_trial96_proj_projection.pth', help='Path to pretrained model weights for category 3')
    parser.add_argument('--fusion_weights4', type=str, default='/gpfs/data/bbj-lab/users/sethis/experiments/checkpoints/cat4_2D_redo1_tunejoint1_trial5_proj/cat4_2D_redo1_tunejoint1_trial5_proj_projection.pth', help='Path to pretrained model weights for category 4')
    
    parser.add_argument('--fusion_backbone1', type=str, default='resnet1d18', help='Backbone for 1D rhythm model (category 1)')
    parser.add_argument('--fusion_backbone3', type=str, default='resnet18', help='Backbone for 2D partial morphology model (category 3)')
    parser.add_argument('--fusion_backbone4', type=str, default='resnet18', help='Backbone for 2D global model (category 4)')

    parser.add_argument('--fusion_proto_dim1', type=int, default=512, help='Prototype dimension for 1D rhythm model (category 1)')
    parser.add_argument('--fusion_proto_dim3', type=int, default=512, help='Prototype dimension for 2D partial morphology model (category 3)')
    parser.add_argument('--fusion_proto_dim4', type=int, default=512, help='Prototype dimension for 2D global model (category 4)')

    parser.add_argument('--fusion_single_ppc1', type=int, default=5, help='Single-class prototypes per class for category 1')
    parser.add_argument('--fusion_single_ppc3', type=int, default=18, help='Single-class prototypes per class for category 3')
    parser.add_argument('--fusion_single_ppc4', type=int, default=7, help='Single-class prototypes per class for category 4')

    parser.add_argument('--fusion_joint_ppb1', type=int, default=0, help='Joint prototypes per border for category 1')
    parser.add_argument('--fusion_joint_ppb3', type=int, default=0, help='Joint prototypes per border for category 3')
    parser.add_argument('--fusion_joint_ppb4', type=int, default=0, help='Joint prototypes per border for category 4')
    args = parser.parse_args()

    # Handle --ckpt alias
    if args.ckpt is not None and args.pretrained_weights is None:
        args.pretrained_weights = args.ckpt

    # Handle train_head_only flag
    if args.train_head_only:
        args.freeze_encoder = True
        args.freeze_prototypes = True
        print("[INFO] train_head_only=True: automatically setting freeze_encoder=True and freeze_prototypes=True")

    # Override lambda values if provided
    if args.lambda_cluster is not None:
        args.lam_clst = args.lambda_cluster
    if args.lambda_separation is not None:
        args.lam_sep = args.lambda_separation

    # Set random seed
    print(f"Setting random seed: {args.seed}")
    seed_everything(args.seed)
    
    # Determine number of classes
    if args.training_stage == 'echonext_adapt':
        # For EchoNext, we'll get num_classes from the dataset after loading
        # Set a placeholder for now (will be updated after dataloader creation)
        num_classes = None
        label_mappings = None
        print("[INFO] EchoNext adaptation: num_classes will be determined from dataset")
    else:
        print("Loading label mappings...")
        label_mappings = load_label_mappings(
            custom_groups=args.custom_groups,
            prototype_category=int(args.label_set) if args.custom_groups else None
        )

        if args.custom_groups:
            num_classes = len(label_mappings["custom"])
        else:
            num_classes = len(label_mappings[args.label_set])

    if args.training_stage == "projection":
        return_sample_ids=True
    else: 
        return_sample_ids=False
    
    # Load Data
    if args.training_stage == 'echonext_adapt':
        # Import EchoNext dataloader
        from data.echonext import get_echonext_dataloaders
        print(f"Loading EchoNext data from {args.dataset_root} (num_workers: {args.num_workers})...")
        train_loader, val_loader, test_loader, class_weights, class_freqs = get_echonext_dataloaders(
            dataset_root=args.dataset_root,
            train_split=args.train_split,
            val_split=args.val_split,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            label_set=args.label_set,
            weight_clip_max=args.weight_clip_max,
            use_inverse_sqrt=args.use_inverse_sqrt,
            sampler_mode=args.sampler_mode,
            sampler_gamma=args.sampler_gamma,
            iters_per_epoch=args.iters_per_epoch,
            seed=args.seed,
        )
        
        # Write class frequencies for audit
        if class_freqs is not None and args.out_dir is not None:
            import json
            from pathlib import Path
            out_dir = Path(args.out_dir)
            out_dir.mkdir(parents=True, exist_ok=True)
            freq_path = out_dir / "class_freqs.json"
            with open(freq_path, 'w') as f:
                json.dump(class_freqs, f, indent=2)
            print(f"[INFO] Wrote class frequencies to {freq_path}")
        # Get num_classes from the dataset
        # Access the dataset from the dataloader
        num_classes = train_loader.dataset.labels.shape[1]
        print(f"[INFO] EchoNext dataset has {num_classes} classes")
        
        # Set class weights for EchoNext
        if args.use_class_weights:
            class_wts = class_weights
        else:
            class_wts = None
    elif args.training_stage != 'fusion':
        print(f"Loading data (num_workers: {args.num_workers})...")

        train_loader, val_loader, test_loader, class_weights = get_dataloaders(
            batch_size=args.batch_size, 
            mode=args.dimension,
            sampling_rate=args.sampling_rate, 
            label_set=args.label_set,
            work_num=args.num_workers,
            return_sample_ids=return_sample_ids,
            custom_groups=args.custom_groups, 
            standardize=args.standardize,
            remove_baseline=args.remove_baseline,
        )

        if args.use_class_weights: 
            class_wts = class_weights
        else: 
            class_wts=None

    # Model selection
    print(f"Selecting model: {args.backbone} with dimension {args.dimension} for stage {args.training_stage}...")
    if args.training_stage == "feature_extractor":
        model = eval(args.backbone)(num_classes=num_classes, dropout=args.dropout)  # Load backbone directly
        if args.pretrained_weights: # Load pretrained weights if specified
            state_dict = torch.load(args.pretrained_weights, map_location=torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
            model.load_state_dict(state_dict, strict=False)
            print(f"Loaded pretrained weights from {args.pretrained_weights}")
    elif args.training_stage != 'fusion':
        if args.dimension == '1D':
            # For EchoNext adaptation, we may have different num_classes than the checkpoint
            # Use strict=False for EchoNext adaptation (different num_classes than PTB checkpoint)
            # This allows classifier and prototype_class_identity to be reinitialized
            load_strict = not (args.training_stage == 'echonext_adapt')
            model = ProtoECGNet1D(num_classes=num_classes, single_class_prototype_per_class=args.single_class_prototype_per_class, 
                                  joint_prototypes_per_border=args.joint_prototypes_per_border, proto_dim=args.proto_dim, 
                                  backbone=args.backbone, prototype_activation_function=args.prototype_activation_function, 
                                  latent_space_type=args.latent_space_type, add_on_layers_type=args.add_on_layers_type, 
                                  class_specific=args.class_specific, last_layer_connection_weight=args.last_layer_connection_weight, 
                                  m=args.m, dropout=args.dropout, custom_groups=args.custom_groups, label_set = args.label_set, 
                                  pretrained_weights=args.pretrained_weights, load_strict=load_strict)
        elif args.dimension == '2D':
            model = ProtoECGNet2D(num_classes=num_classes, single_class_prototype_per_class=args.single_class_prototype_per_class, 
                                  joint_prototypes_per_border=args.joint_prototypes_per_border, proto_dim=args.proto_dim, 
                                  proto_time_len=args.proto_time_len,backbone=args.backbone, prototype_activation_function=args.prototype_activation_function, 
                                  latent_space_type=args.latent_space_type, add_on_layers_type=args.add_on_layers_type, 
                                  class_specific=args.class_specific, last_layer_connection_weight=args.last_layer_connection_weight, 
                                  m=args.m, dropout=args.dropout, custom_groups=args.custom_groups, label_set = args.label_set, pretrained_weights=args.pretrained_weights)
        else:
            raise ValueError(f"Unsupported model dimension: {args.dimension}")

    # Apply freezing if requested
    if args.freeze_encoder or args.freeze_prototypes:
        if isinstance(model, (ProtoECGNet1D, ProtoECGNet2D)):
            if args.freeze_encoder:
                print("[INFO] Freezing encoder parameters...")
                for param in model.feature_extractor.parameters():
                    param.requires_grad = False
            if args.freeze_prototypes:
                print("[INFO] Freezing prototype parameters...")
                model.prototype_vectors.requires_grad = False
        else:
            print("[WARNING] Freezing flags only apply to ProtoECGNet models")

    # Train or test model
    if args.test_model:
        test_model(model, test_loader, args=args)
    elif args.training_stage == "projection":
        trainer = train_model(model, train_loader, val_loader, args, class_wts)
    elif args.training_stage == "fusion":
        label_map = load_fusion_label_mappings()

        num_classes1 = len(label_map["1"])
        num_classes3 = len(label_map["3"])
        num_classes4 = len(label_map["4"])


        print("Loading pretrained 1D and 2D models for fusion...")
        
        print(f"Creating category 1 model with backbone {args.fusion_backbone1}...")
        model_1d = ProtoECGNet1D(
            num_classes=num_classes1,
            single_class_prototype_per_class=args.fusion_single_ppc1,
            joint_prototypes_per_border=args.fusion_joint_ppb1,
            proto_dim=args.fusion_proto_dim1,
            backbone=args.fusion_backbone1,
            prototype_activation_function=args.prototype_activation_function,
            latent_space_type=args.latent_space_type,
            add_on_layers_type=args.add_on_layers_type,
            class_specific=args.class_specific,
            last_layer_connection_weight=args.last_layer_connection_weight,
            m=args.m,
            dropout=args.dropout,
            custom_groups=True,
            label_set="1",
            pretrained_weights=args.fusion_weights1
        )

        print(f"1D model loaded. Creating category 3 model with backbone {args.fusion_backbone3}...")

        model_2d_partial = ProtoECGNet2D(
            num_classes=num_classes3,
            single_class_prototype_per_class=args.fusion_single_ppc3,
            joint_prototypes_per_border=args.fusion_joint_ppb3,
            proto_dim=args.fusion_proto_dim3,
            proto_time_len=args.proto_time_len,
            backbone=args.fusion_backbone3,
            prototype_activation_function=args.prototype_activation_function,
            latent_space_type=args.latent_space_type,
            add_on_layers_type=args.add_on_layers_type,
            class_specific=args.class_specific,
            last_layer_connection_weight=args.last_layer_connection_weight,
            m=args.m,
            dropout=args.dropout,
            custom_groups=True,
            label_set="3",
            pretrained_weights=args.fusion_weights3
        )

        print(f"2D partial model loaded. Creating category 4 model with backbone {args.fusion_backbone4}...")
        model_2d_global = ProtoECGNet2D(
            num_classes=num_classes4,
            single_class_prototype_per_class=args.fusion_single_ppc4,
            joint_prototypes_per_border=args.fusion_joint_ppb4,
            proto_dim=args.fusion_proto_dim4,
            proto_time_len=32,
            backbone=args.fusion_backbone4,
            prototype_activation_function=args.prototype_activation_function,
            latent_space_type=args.latent_space_type,
            add_on_layers_type=args.add_on_layers_type,
            class_specific=args.class_specific,
            last_layer_connection_weight=args.last_layer_connection_weight,
            m=args.m,
            dropout=args.dropout,
            custom_groups=True,
            label_set="4",
            pretrained_weights=args.fusion_weights4
        )

        print(f"Loaded all three models for fusion {args.fusion_backbone3}...")
        for m in [model_1d, model_2d_partial, model_2d_global]:
                m.eval()
                for param in m.parameters():
                    param.requires_grad = False

        model = FusionProtoClassifier(model_1d, model_2d_partial, model_2d_global, num_classes=71)
        print(f"Fusion classifier initialized. Getting dataloaders...")
        train_loader, val_loader, test_loader, class_weights = get_fusion_dataloaders(args, return_sample_ids=return_sample_ids)

        print(f"Got dataloaders. Starting fusion classifier training...")
        trainer = train_model(model, train_loader, val_loader, args, class_weights)

        print(f"Training complete. Beginning fusion classifier testing...")
        test_model(model, test_loader, args=args, trainer=trainer)
    else:
        trainer = train_model(model, train_loader, val_loader, args, class_wts)
        test_model(model, test_loader, args=args, trainer=trainer)

    # Export encoder to TorchScript if requested
    if args.export_encoder_ts is not None:
        from model.export import export_encoder_to_torchscript
        print(f"[INFO] Exporting encoder to TorchScript: {args.export_encoder_ts}")
        export_encoder_to_torchscript(model, args.export_encoder_ts, device=args.device)
        print(f"[OK] Encoder exported to {args.export_encoder_ts}")
    
