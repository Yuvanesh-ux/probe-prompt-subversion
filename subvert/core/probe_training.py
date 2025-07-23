"""
Train linear probes on collected activation data.
"""

import torch
import json
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, classification_report
import pickle
from typing import Dict, Tuple
import argparse


def load_activation_data(file_path: str) -> Dict:
    """Load activation data saved by collect.py"""
    return torch.load(file_path, map_location='cpu')


def prepare_probe_data(data: Dict) -> Tuple[np.ndarray, np.ndarray]:
    """
    Prepare activation data for probe training.
    Returns X (features) and y (labels).
    """
    activations = data['activations']
    labels = data['labels'] 
    X = activations.numpy()
    print(f"Raw activations shape: {X.shape}")
    if len(X.shape) == 3:
        if X.shape[1] == 1:
            # Shape is (n_samples, 1, hidden_dim) - squeeze middle dimension
            print(f"3D activations with singleton middle dim: {X.shape} -> {X.squeeze(1).shape}")
            X = X.squeeze(1)
        else:
            print(f"3D activations with sequence dim: {X.shape} -> {X[:, -1, :].shape}")
            X = X[:, -1, :]  # Take last token
    if len(X.shape) != 2:
        raise ValueError(f"Expected 2D activations after processing, got shape {X.shape}")
    y = np.array(labels, dtype=int)
    print(f"Prepared data: {X.shape} features, {len(y)} samples")
    print(f"Label distribution: {np.sum(y)} positive, {len(y) - np.sum(y)} negative")
    return X, y


def train_probe(X: np.ndarray, y: np.ndarray, 
               test_size: float = 0.2, random_state: int = 42) -> Dict:
    """
    Train logistic regression probe with train/validation split.
    Returns dictionary with probe, metrics, and data splits.
    """
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    probe = LogisticRegression(
        solver='liblinear', 
        C=1.0, 
        max_iter=1000, 
        random_state=random_state
    )
    print("Training probe...")
    probe.fit(X_train, y_train)
    train_preds = probe.predict_proba(X_train)[:, 1]
    val_preds = probe.predict_proba(X_val)[:, 1]
    train_auroc = roc_auc_score(y_train, train_preds)
    val_auroc = roc_auc_score(y_val, val_preds)
    print(f"Training AUROC: {train_auroc:.4f}")
    print(f"Validation AUROC: {val_auroc:.4f}")
    val_pred_classes = probe.predict(X_val)
    report = classification_report(y_val, val_pred_classes, output_dict=True)
    return {
        'probe': probe,
        'train_auroc': train_auroc,
        'val_auroc': val_auroc,
        'classification_report': report,
        'X_train': X_train,
        'X_val': X_val,
        'y_train': y_train,
        'y_val': y_val,
        'feature_dim': X.shape[1]
    }


def evaluate_probe_on_data(probe: LogisticRegression, data: Dict) -> Dict:
    """
    Evaluate trained probe on new data (e.g., deployment data).
    Returns evaluation metrics.
    """
    X, y = prepare_probe_data(data)
    pred_probs = probe.predict_proba(X)[:, 1]
    pred_classes = probe.predict(X)
    auroc = roc_auc_score(y, pred_probs)
    report = classification_report(y, pred_classes, output_dict=True)
    print(f"AUROC: {auroc:.4f}")
    return {
        'auroc': auroc,
        'classification_report': report,
        'predictions': pred_probs.tolist(),
        'true_labels': y.tolist(),
        'n_samples': len(y)
    }


def main():
    parser = argparse.ArgumentParser(description="Train probe on activation data")
    parser.add_argument("--train-data", required=True, 
                       help="Training activation data (.pt file)")
    parser.add_argument("--output", required=True,
                       help="Output file for trained probe (.pkl)")
    parser.add_argument("--eval-data", 
                       help="Optional evaluation data (.pt file)")
    parser.add_argument("--results", 
                       help="Optional results output (.json)")
    args = parser.parse_args()
    print(f"Loading training data from {args.train_data}")
    train_data = load_activation_data(args.train_data)
    X, y = prepare_probe_data(train_data)
    results = train_probe(X, y)
    print(f"Saving probe to {args.output}")
    with open(args.output, 'wb') as f:
        pickle.dump({
            'probe': results['probe'],
            'train_auroc': results['train_auroc'],
            'val_auroc': results['val_auroc'],
            'feature_dim': results['feature_dim'],
            'metadata': {
                'system_prompt': train_data['system_prompt'],
                'hook_point': train_data['hook_point'],
                'n_train_prompts': train_data['n_prompts']
            }
        }, f)
    if args.eval_data:
        print(f"\nEvaluating on {args.eval_data}")
        eval_data = load_activation_data(args.eval_data)
        eval_results = evaluate_probe_on_data(results['probe'], eval_data)
        auroc_drop = results['val_auroc'] - eval_results['auroc']
        print(f"AUROC drop: {auroc_drop:.4f} ({results['val_auroc']:.4f} -> {eval_results['auroc']:.4f})")
        if args.results:
            full_results = {
                'train_auroc': results['train_auroc'],
                'val_auroc': results['val_auroc'],
                'eval_auroc': eval_results['auroc'],
                'auroc_drop': auroc_drop,
                'success': auroc_drop >= 0.30,
                'train_metadata': {
                    'system_prompt': train_data['system_prompt'],
                    'hook_point': train_data['hook_point'],
                    'n_prompts': train_data['n_prompts']
                },
                'eval_metadata': {
                    'system_prompt': eval_data['system_prompt'],
                    'hook_point': eval_data['hook_point'],
                    'n_prompts': eval_data['n_prompts']
                }
            }
            with open(args.results, 'w') as f:
                json.dump(full_results, f, indent=2)
            print(f"Results saved to {args.results}")


if __name__ == "__main__":
    main()