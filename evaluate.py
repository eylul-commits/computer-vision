#Evaluation and comparison script for trained models.
import argparse
import json
import torch
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

from src.utils.visualization import compare_models


def load_results(results_dir: str = 'models'):
    """Load all experiment results."""
    results_dir = Path(results_dir)
    
    all_results = {}
    
    # Find all result files
    for config_file in results_dir.glob('*_config.json'):
        experiment_name = config_file.stem.replace('_config', '')
        
        # Load config
        with open(config_file, 'r') as f:
            config = json.load(f)
        
        # Check if test results exist
        test_results_file = results_dir / f"{experiment_name}_test_results.json"
        if test_results_file.exists():
            with open(test_results_file, 'r') as f:
                test_results = json.load(f)
            
            all_results[experiment_name] = {
                'config': config,
                'test_results': test_results,
                'metrics': test_results['metrics']
            }
    
    return all_results


def create_comparison_table(results: dict, output_file: str = 'results/model_comparison.csv'):
    """Create a comparison table of all models."""
    
    data = []
    for exp_name, exp_data in results.items():
        config = exp_data['config']
        metrics = exp_data['metrics']
        
        row = {
            'Experiment': exp_name,
            'Model Type': config['model_type'],
            'Model Name': config['model_name'],
            'Dataset': config['dataset'],
            'Accuracy': metrics['accuracy'],
            'Precision': metrics['precision'],
            'Recall': metrics['recall'],
            'F1-Score': metrics['f1'],
        }
        
        if 'roc_auc' in metrics and metrics['roc_auc'] is not None:
            row['ROC-AUC'] = metrics['roc_auc']
        
        data.append(row)
    
    # Create DataFrame
    df = pd.DataFrame(data)
    
    # Sort by accuracy
    df = df.sort_values('Accuracy', ascending=False)
    
    # Save to CSV
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    
    print(f"\nComparison table saved to {output_path}")
    
    return df


def print_comparison_table(df: pd.DataFrame):
    """Print formatted comparison table."""
    
    print("\n" + "="*120)
    print("MODEL COMPARISON")
    print("="*120)
    
    # Set pandas display options
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', None)
    pd.set_option('display.max_colwidth', None)
    
    print(df.to_string(index=False))
    print("="*120 + "\n")


def create_comparison_plots(results: dict, output_dir: str = 'figures/comparison'):
    """Create comparison plots for all models."""
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Extract metrics
    metrics_dict = {name: data['metrics'] for name, data in results.items()}
    
    # Plot accuracy comparison
    compare_models(
        metrics_dict,
        metric='accuracy',
        save_path=str(output_dir / 'accuracy_comparison.png')
    )
    
    # Plot F1 comparison
    compare_models(
        metrics_dict,
        metric='f1',
        save_path=str(output_dir / 'f1_comparison.png')
    )
    
    # Plot precision comparison
    compare_models(
        metrics_dict,
        metric='precision',
        save_path=str(output_dir / 'precision_comparison.png')
    )
    
    # Plot recall comparison
    compare_models(
        metrics_dict,
        metric='recall',
        save_path=str(output_dir / 'recall_comparison.png')
    )
    
    print(f"Comparison plots saved to {output_dir}")


def find_best_model(results: dict):
    """Find the best performing model."""
    
    best_acc = 0
    best_model = None
    
    for exp_name, exp_data in results.items():
        acc = exp_data['metrics']['accuracy']
        if acc > best_acc:
            best_acc = acc
            best_model = exp_name
    
    return best_model, best_acc


def main():
    parser = argparse.ArgumentParser(description='Evaluate and compare trained models')
    parser.add_argument('--results-dir', type=str, default='models',
                       help='Directory containing model results')
    parser.add_argument('--output-dir', type=str, default='results',
                       help='Directory to save comparison results')
    parser.add_argument('--dataset', type=str, default=None,
                       help='Filter by dataset')
    args = parser.parse_args()
    
    print(f"\n{'='*80}")
    print("MODEL EVALUATION AND COMPARISON")
    print(f"{'='*80}\n")
    
    # Load results
    print(f"Loading results from {args.results_dir}...")
    results = load_results(args.results_dir)
    
    if not results:
        print("No results found!")
        return
    
    # Filter by dataset if specified
    if args.dataset:
        results = {
            name: data for name, data in results.items()
            if data['config']['dataset'] == args.dataset
        }
        print(f"Filtered to {args.dataset} dataset: {len(results)} experiments")
    
    print(f"Found {len(results)} experiments")
    
    # Create comparison table
    df = create_comparison_table(
        results,
        output_file=f"{args.output_dir}/model_comparison.csv"
    )
    
    # Print comparison table
    print_comparison_table(df)
    
    # Create comparison plots
    create_comparison_plots(
        results,
        output_dir=f"{args.output_dir}/comparison"
    )
    
    # Find best model
    best_model, best_acc = find_best_model(results)
    
    print(f"\n{'='*80}")
    print("BEST MODEL")
    print(f"{'='*80}")
    print(f"Model: {best_model}")
    print(f"Accuracy: {best_acc:.4f}")
    
    # Print best model details
    best_results = results[best_model]
    print(f"\nDetails:")
    print(f"  Model Type: {best_results['config']['model_type']}")
    print(f"  Model Name: {best_results['config']['model_name']}")
    print(f"  Dataset: {best_results['config']['dataset']}")
    print(f"  Precision: {best_results['metrics']['precision']:.4f}")
    print(f"  Recall: {best_results['metrics']['recall']:.4f}")
    print(f"  F1-Score: {best_results['metrics']['f1']:.4f}")
    
    if 'roc_auc' in best_results['metrics'] and best_results['metrics']['roc_auc']:
        print(f"  ROC-AUC: {best_results['metrics']['roc_auc']:.4f}")
    
    print(f"{'='*80}\n")
    
    # Save best model info
    best_model_info = {
        'best_model': best_model,
        'accuracy': best_acc,
        'config': best_results['config'],
        'metrics': best_results['metrics']
    }
    
    best_model_path = Path(args.output_dir) / 'best_model.json'
    with open(best_model_path, 'w') as f:
        json.dump(best_model_info, f, indent=2)
    
    print(f"Best model info saved to {best_model_path}")


if __name__ == '__main__':
    main()

