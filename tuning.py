#!/usr/bin/env python3
"""
Hyperparameter tuning script for EchoNext Stage B adaptation.
Runs a grid/random search over key hyperparameters with 30 epochs each.
"""
import os
import sys
import json
import subprocess
import itertools
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple

# Default paths
DEFAULT_DATASET_ROOT = "/opt/gpudata/ecg/echonext-v1.0.0"
DEFAULT_CKPT = "/opt/gpudata/summereunann/ptbxl_weights/cat1.ckpt"
DEFAULT_BASE_OUT_DIR = "results/hyperparameter_tuning"
DEFAULT_SEED = 42

# Hyperparameter search space
HYPERPARAMETER_SPACE = {
    # Learning rate (critical for convergence)
    'lr': [1e-5, 5e-5, 1e-4, 5e-4],
    
    # Lambda cluster (how much to push prototypes toward positives)
    'lambda_cluster': [0.6, 0.8, 1.0, 1.2],
    
    # Lambda separation (how much to push prototypes away from negatives)
    'lambda_separation': [0.05, 0.1, 0.15, 0.2],
    
    # Batch size (affects gradient stability)
    'batch_size': [64, 128],
    
    # Scheduler type (affects LR decay)
    'scheduler_type': ['CosineAnnealingLR', 'ReduceLROnPlateau'],
}

# Fixed configuration (not tuned)
FIXED_CONFIG = {
    'training_stage': 'echonext_adapt',
    'dataset_root': DEFAULT_DATASET_ROOT,
    'train_split': 'train',
    'val_split': 'val',
    'dimension': '1D',
    'backbone': 'resnet1d18',
    'label_set': '1',
    'custom_groups': 'true',
    'single_class_prototype_per_class': 5,
    'proto_dim': 512,
    'freeze_encoder': 'true',
    'freeze_prototypes': 'false',
    'loss': 'bce',
    'epochs': 30,  # Full 30 epochs
    'seed': DEFAULT_SEED,
    'sampling_rate': 100,
    'use_class_weights': 'true',
    'early_stop_metric': 'auroc_micro',
    'early_stop_patience': 7,  # Allow more patience for 30 epochs
    'scheduler_eta_min': 1e-6,
    'save_both_checkpoints': 'true',
    'num_workers': 4,
    # Use writable directories (avoid /gpfs permission issues)
    'checkpoint_dir': None,  # Will be set per-trial in build_training_command
    'log_dir': None,  # Will be set per-trial in build_training_command
    'test_dir': None,  # Will be set per-trial in build_training_command
}


def generate_configurations(mode: str = 'grid', n_trials: int = 20) -> List[Dict]:
    """
    Generate hyperparameter configurations.
    
    Args:
        mode: 'grid' for full grid search, 'random' for random search
        n_trials: Number of trials for random search (ignored for grid)
    
    Returns:
        List of configuration dictionaries
    """
    if mode == 'grid':
        # Full grid search
        keys = list(HYPERPARAMETER_SPACE.keys())
        values = list(HYPERPARAMETER_SPACE.values())
        configs = []
        for combo in itertools.product(*values):
            config = dict(zip(keys, combo))
            configs.append(config)
        return configs
    else:
        # Random search
        import random
        random.seed(DEFAULT_SEED)
        configs = []
        for _ in range(n_trials):
            config = {}
            for key, values in HYPERPARAMETER_SPACE.items():
                config[key] = random.choice(values)
            configs.append(config)
        return configs


def config_to_job_name(config: Dict, trial_id: int) -> str:
    """Generate a unique job name from configuration."""
    lr_str = f"lr{config['lr']:.0e}".replace('e-0', 'e-').replace('+', '')
    lc_str = f"lc{config['lambda_cluster']:.1f}".replace('.', 'p')
    ls_str = f"ls{config['lambda_separation']:.2f}".replace('.', 'p')
    bs_str = f"bs{config['batch_size']}"
    sch_str = config['scheduler_type'][:4].lower()  # 'cosi' or 'red'
    return f"tune{trial_id:03d}_{lr_str}_{lc_str}_{ls_str}_{bs_str}_{sch_str}"


def build_training_command(config: Dict, trial_id: int, ckpt: str, 
                          base_out_dir: str, gpu_id: int = 2) -> List[str]:
    """Build the training command for a given configuration."""
    job_name = config_to_job_name(config, trial_id)
    out_dir = os.path.join(base_out_dir, job_name)
    
    # Create per-trial directories
    trial_checkpoint_dir = os.path.join(out_dir, 'checkpoints')
    trial_log_dir = os.path.join(out_dir, 'logs')
    trial_test_dir = os.path.join(out_dir, 'test_results')
    
    cmd = [
        'python3', 'src/main.py',
        '--job_name', job_name,
        '--ckpt', ckpt,
        '--out_dir', out_dir,
        '--checkpoint_dir', trial_checkpoint_dir,
        '--log_dir', trial_log_dir,
        '--test_dir', trial_test_dir,
    ]
    
    # Add fixed config (skip None values and paths we set explicitly)
    for key, value in FIXED_CONFIG.items():
        if value is None or key in ['checkpoint_dir', 'log_dir', 'test_dir']:
            continue
        cmd.extend([f'--{key}', str(value)])
    
    # Add hyperparameters from config
    for key, value in config.items():
        if key == 'lambda_cluster':
            cmd.extend(['--lambda_cluster', str(value)])
            cmd.extend(['--lam_clst', str(value)])  # Also set alias
        elif key == 'lambda_separation':
            cmd.extend(['--lambda_separation', str(value)])
            cmd.extend(['--lam_sep', str(value)])  # Also set alias
        else:
            cmd.extend([f'--{key}', str(value)])
    
    # Set CUDA device
    env = os.environ.copy()
    env['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
    
    return cmd, env, job_name, out_dir


def extract_results(out_dir: str, job_name: str) -> Dict:
    """Extract final results from training output."""
    results = {
        'job_name': job_name,
        'out_dir': out_dir,
        'status': 'unknown',
        'best_macro_auroc': None,
        'best_micro_auroc': None,
        'final_macro_auroc': None,
        'final_micro_auroc': None,
        'epochs_completed': None,
        'best_epoch': None,
    }
    
    # Try to find training log
    log_path = os.path.join(out_dir, 'training.log')
    if not os.path.exists(log_path):
        # Try alternative location
        log_path = os.path.join(out_dir, f'{job_name}.log')
    
    if os.path.exists(log_path):
        try:
            import re
            with open(log_path, 'r') as f:
                log_content = f.read()
            
            # Find best checkpoints
            best_micro_match = re.search(r'best.*micro.*?(\d+\.\d+)', log_content, re.IGNORECASE)
            best_macro_match = re.search(r'best.*macro.*?(\d+\.\d+)', log_content, re.IGNORECASE)
            
            # Find final epoch metrics
            epoch_matches = list(re.finditer(
                r'Epoch (\d+).*?val_auroc_macro=([\d.]+).*?val_auroc_micro=([\d.]+)',
                log_content
            ))
            if epoch_matches:
                last_match = epoch_matches[-1]
                results['epochs_completed'] = int(last_match.group(1))
                # Strip trailing periods before converting to float
                macro_str = last_match.group(2).rstrip('.')
                micro_str = last_match.group(3).rstrip('.')
                results['final_macro_auroc'] = float(macro_str)
                results['final_micro_auroc'] = float(micro_str)
            
            # Try to find best checkpoint files
            checkpoint_dir = os.path.join(out_dir, 'checkpoints', job_name)
            if os.path.exists(checkpoint_dir):
                # Find best micro checkpoint
                best_micro_ckpts = [f for f in os.listdir(checkpoint_dir) 
                                  if 'micro' in f and f.endswith('.ckpt')]
                if best_micro_ckpts:
                    # Extract epoch from filename
                    epoch_match = re.search(r'epoch=(\d+)', best_micro_ckpts[0])
                    if epoch_match:
                        results['best_epoch'] = int(epoch_match.group(1))
                
                # Try to read from checkpoint metadata if available
                best_macro_dir = os.path.join(checkpoint_dir, 'best_macro')
                if os.path.exists(best_macro_dir):
                    best_macro_ckpts = [f for f in os.listdir(best_macro_dir) 
                                      if f.endswith('.ckpt')]
                    if best_macro_ckpts:
                        # Extract AUROC from filename if available
                        auroc_match = re.search(r'auroc_macro=([\d.]+)', best_macro_ckpts[0])
                        if auroc_match:
                            results['best_macro_auroc'] = float(auroc_match.group(1))
            
            results['status'] = 'completed' if results['epochs_completed'] else 'running'
        except Exception as e:
            results['status'] = f'error_extracting: {str(e)}'
    else:
        results['status'] = 'no_log'
    
    return results


def run_trial(config: Dict, trial_id: int, ckpt: str, base_out_dir: str, 
             gpu_id: int = 2, dry_run: bool = False) -> Dict:
    """Run a single training trial."""
    print(f"\n{'='*80}")
    print(f"Trial {trial_id}: {config_to_job_name(config, trial_id)}")
    print(f"{'='*80}")
    print(f"Configuration:")
    for key, value in config.items():
        print(f"  {key}: {value}")
    print()
    
    cmd, env, job_name, out_dir = build_training_command(config, trial_id, ckpt, 
                                                        base_out_dir, gpu_id)
    
    if dry_run:
        print("DRY RUN - Would execute:")
        print(" ".join(cmd))
        print(f"CUDA_VISIBLE_DEVICES={env.get('CUDA_VISIBLE_DEVICES')}")
        return {'status': 'dry_run', 'job_name': job_name}
    
    # Create output directory
    os.makedirs(out_dir, exist_ok=True)
    
    # Save configuration
    config_path = os.path.join(out_dir, 'config.json')
    with open(config_path, 'w') as f:
        json.dump({**FIXED_CONFIG, **config}, f, indent=2)
    
    # Run training
    log_path = os.path.join(out_dir, 'training.log')
    print(f"Running training (GPU {gpu_id})...")
    print(f"Log: {log_path}")
    print(f"Command: {' '.join(cmd)}\n")
    
    try:
        with open(log_path, 'w') as log_file:
            process = subprocess.Popen(
                cmd,
                env=env,
                stdout=log_file,
                stderr=subprocess.STDOUT,
                cwd=os.path.dirname(os.path.abspath(__file__))
            )
            process.wait()
            
        if process.returncode == 0:
            print(f"Trial {trial_id} completed successfully")
        else:
            print(f"WARNING: Trial {trial_id} failed with return code {process.returncode}")
    except Exception as e:
        print(f"ERROR: Trial {trial_id} crashed: {str(e)}")
        return {'status': 'crashed', 'error': str(e), 'job_name': job_name}
    
    # Extract results
    results = extract_results(out_dir, job_name)
    results.update(config)
    return results


def main():
    parser = argparse.ArgumentParser(description='Hyperparameter tuning for EchoNext Stage B')
    parser.add_argument('--mode', type=str, default='grid', choices=['grid', 'random'],
                       help='Search mode: grid or random')
    parser.add_argument('--n_trials', type=int, default=20,
                       help='Number of trials for random search (ignored for grid)')
    parser.add_argument('--gpu_id', type=int, default=2,
                       help='GPU ID to use (default: 2)')
    parser.add_argument('--ckpt', type=str, default=DEFAULT_CKPT,
                       help=f'Checkpoint path (default: {DEFAULT_CKPT})')
    parser.add_argument('--base_out_dir', type=str, default=DEFAULT_BASE_OUT_DIR,
                       help=f'Base output directory (default: {DEFAULT_BASE_OUT_DIR})')
    parser.add_argument('--start_trial', type=int, default=0,
                       help='Start trial ID (for resuming)')
    parser.add_argument('--dry_run', action='store_true',
                       help='Dry run (print commands without executing)')
    parser.add_argument('--results_json', type=str, default=None,
                       help='Path to save results JSON (default: base_out_dir/results.json)')
    
    args = parser.parse_args()
    
    # Generate configurations
    configs = generate_configurations(args.mode, args.n_trials)
    print(f"\n{'='*80}")
    print(f"Hyperparameter Tuning")
    print(f"{'='*80}")
    print(f"Mode: {args.mode}")
    print(f"Total trials: {len(configs)}")
    print(f"GPU: {args.gpu_id}")
    print(f"Epochs per trial: {FIXED_CONFIG['epochs']}")
    print(f"Output directory: {args.base_out_dir}")
    print(f"{'='*80}\n")
    
    # Create output directory
    os.makedirs(args.base_out_dir, exist_ok=True)
    
    # Run trials
    all_results = []
    for trial_id, config in enumerate(configs[args.start_trial:], start=args.start_trial):
        result = run_trial(config, trial_id, args.ckpt, args.base_out_dir, 
                          args.gpu_id, args.dry_run)
        all_results.append(result)
        
        # Save intermediate results
        results_path = args.results_json or os.path.join(args.base_out_dir, 'results.json')
        with open(results_path, 'w') as f:
            json.dump(all_results, f, indent=2)
        
        if args.dry_run and trial_id >= args.start_trial + 4:
            print("\n... (stopping after 5 trials in dry run mode)")
            break
    
    # Final summary
    print(f"\n{'='*80}")
    print(f"Tuning Complete")
    print(f"{'='*80}")
    print(f"Total trials: {len(all_results)}")
    
    if not args.dry_run:
        # Find best configuration
        valid_results = [r for r in all_results if r.get('final_macro_auroc') is not None]
        if valid_results:
            best = max(valid_results, key=lambda x: x.get('final_macro_auroc', 0))
            print(f"\nBest Configuration (by Macro AUROC):")
            print(f"  Job: {best['job_name']}")
            print(f"  Macro AUROC: {best['final_macro_auroc']:.4f}")
            print(f"  Micro AUROC: {best['final_micro_auroc']:.4f}")
            best_config = {k: v for k, v in best.items() if k in HYPERPARAMETER_SPACE}
            print(f"  Config: {json.dumps(best_config, indent=2)}")
        
        results_path = args.results_json or os.path.join(args.base_out_dir, 'results.json')
        print(f"\nResults saved to: {results_path}")


if __name__ == '__main__':
    main()

