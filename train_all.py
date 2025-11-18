"""
Script to train multiple models in sequence.
Useful for comparing different architectures.
"""
import subprocess
import sys
from pathlib import Path
import json
import time


# Define experiments
EXPERIMENTS = [
    # Phase 2: Baseline CNNs
    {
        'name': 'Baseline - Simple CNN',
        'model_type': 'baseline',
        'model_name': 'simple',
        'epochs': 30,
        'batch_size': 32,
        'lr': 0.001
    },
    {
        'name': 'Baseline - Improved CNN',
        'model_type': 'baseline',
        'model_name': 'improved',
        'epochs': 30,
        'batch_size': 32,
        'lr': 0.001
    },
    
    # Phase 3: Transfer Learning
    {
        'name': 'Transfer Learning - ResNet50',
        'model_type': 'transfer',
        'model_name': 'resnet50',
        'epochs': 25,
        'batch_size': 32,
        'lr': 0.0001,
        'pretrained': True
    },
    {
        'name': 'Transfer Learning - EfficientNet-B0',
        'model_type': 'transfer',
        'model_name': 'efficientnet_b0',
        'epochs': 25,
        'batch_size': 32,
        'lr': 0.0001,
        'pretrained': True
    },
    {
        'name': 'Transfer Learning - ConvNeXt Tiny',
        'model_type': 'transfer',
        'model_name': 'convnext_tiny',
        'epochs': 25,
        'batch_size': 32,
        'lr': 0.0001,
        'pretrained': True
    },
    
    # Phase 4: Transformers
    {
        'name': 'Transformer - ViT Base',
        'model_type': 'transformer',
        'model_name': 'vit_base_patch16_224',
        'epochs': 25,
        'batch_size': 16,
        'lr': 0.00005,
        'pretrained': True
    },
    {
        'name': 'Transformer - DeiT3 Base',
        'model_type': 'transformer',
        'model_name': 'deit3_base_patch16_224',
        'epochs': 25,
        'batch_size': 16,
        'lr': 0.00005,
        'pretrained': True
    },
    {
        'name': 'Transformer - Swin Transformer',
        'model_type': 'transformer',
        'model_name': 'swin_base_patch4_window7_224',
        'epochs': 25,
        'batch_size': 16,
        'lr': 0.00005,
        'pretrained': True
    },
    
    # Phase 4 (SOTA): 2024-2025 Models
    {
        'name': 'SOTA 2025 - EVA-CLIP-02 Base',
        'model_type': 'sota',
        'model_name': 'evaclip02_base',
        'epochs': 20,
        'batch_size': 16,
        'lr': 0.00003,
        'pretrained': True
    },
    {
        'name': 'SOTA 2025 - SigLIP',
        'model_type': 'sota',
        'model_name': 'siglip',
        'epochs': 20,
        'batch_size': 16,
        'lr': 0.00003,
        'pretrained': True
    }
]


def run_experiment(exp, dataset='smoke', device='cuda'):
    """Run a single experiment."""
    print(f"\n{'='*80}")
    print(f"Starting Experiment: {exp['name']}")
    print(f"Dataset: {dataset}")
    print(f"{'='*80}\n")
    
    # Build command
    cmd = [
        sys.executable,
        'train.py',
        '--model-type', exp['model_type'],
        '--model-name', exp['model_name'],
        '--dataset', dataset,
        '--epochs', str(exp['epochs']),
        '--batch-size', str(exp['batch_size']),
        '--lr', str(exp['lr']),
        '--device', device
    ]
    
    if exp.get('pretrained', False):
        cmd.append('--pretrained')
    
    if exp.get('freeze_backbone', False):
        cmd.append('--freeze-backbone')
    
    if exp.get('dropout'):
        cmd.extend(['--dropout', str(exp['dropout'])])
    
    # Run experiment
    start_time = time.time()
    try:
        result = subprocess.run(cmd, check=True)
        elapsed_time = time.time() - start_time
        
        print(f"\n✓ Experiment completed in {elapsed_time/60:.2f} minutes")
        return True, elapsed_time
    
    except subprocess.CalledProcessError as e:
        print(f"\n✗ Experiment failed with error: {e}")
        return False, 0


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Train multiple models')
    parser.add_argument('--dataset', type=str, default='smoke',
                       choices=['fire', 'smoke'],
                       help='Dataset to use')
    parser.add_argument('--device', type=str, default='cuda',
                       choices=['cuda', 'cpu'],
                       help='Device to use')
    parser.add_argument('--experiments', type=str, nargs='+',
                       help='Specific experiments to run (indices or names)')
    args = parser.parse_args()
    
    # Filter experiments if specified
    if args.experiments:
        selected_experiments = []
        for exp_id in args.experiments:
            try:
                idx = int(exp_id)
                if 0 <= idx < len(EXPERIMENTS):
                    selected_experiments.append(EXPERIMENTS[idx])
            except ValueError:
                # Try to match by name
                for exp in EXPERIMENTS:
                    if exp_id.lower() in exp['name'].lower():
                        selected_experiments.append(exp)
                        break
        experiments = selected_experiments
    else:
        experiments = EXPERIMENTS
    
    if not experiments:
        print("No experiments selected!")
        return
    
    print(f"\n{'='*80}")
    print(f"Training Pipeline")
    print(f"{'='*80}")
    print(f"Dataset: {args.dataset}")
    print(f"Device: {args.device}")
    print(f"Number of experiments: {len(experiments)}")
    print(f"{'='*80}\n")
    
    # List experiments
    print("Experiments to run:")
    for i, exp in enumerate(experiments):
        print(f"  {i+1}. {exp['name']}")
    print()
    
    # Run experiments
    results = []
    total_start_time = time.time()
    
    for i, exp in enumerate(experiments):
        print(f"\n[{i+1}/{len(experiments)}] Running: {exp['name']}")
        success, elapsed_time = run_experiment(exp, args.dataset, args.device)
        
        results.append({
            'experiment': exp['name'],
            'success': success,
            'time': elapsed_time
        })
    
    total_time = time.time() - total_start_time
    
    # Summary
    print(f"\n{'='*80}")
    print("TRAINING SUMMARY")
    print(f"{'='*80}")
    print(f"Total time: {total_time/60:.2f} minutes ({total_time/3600:.2f} hours)")
    print(f"\nResults:")
    
    successful = 0
    for result in results:
        status = "✓" if result['success'] else "✗"
        time_str = f"{result['time']/60:.2f} min" if result['success'] else "Failed"
        print(f"  {status} {result['experiment']:<40} {time_str}")
        if result['success']:
            successful += 1
    
    print(f"\nSuccessful: {successful}/{len(results)}")
    print(f"{'='*80}\n")
    
    # Save summary
    summary_path = Path('results') / f'training_summary_{args.dataset}.json'
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(summary_path, 'w') as f:
        json.dump({
            'dataset': args.dataset,
            'total_time': total_time,
            'results': results
        }, f, indent=2)
    
    print(f"Summary saved to {summary_path}")


if __name__ == '__main__':
    main()

