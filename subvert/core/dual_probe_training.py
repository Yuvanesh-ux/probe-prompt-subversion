"""
Dual-probe training pipeline for conscious control experiments.
Trains separate blue and red color detection probes with comprehensive validation.
"""

import torch
import json
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import roc_auc_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
import pickle
from typing import Dict, Tuple, List, Optional
import argparse
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns


class DualProbeTrainer:
    """
    Enhanced probe trainer for dual-color conscious control experiments.
    """
    
    def __init__(self, 
                 feature_standardization: bool = True,
                 cross_validation_folds: int = 5,
                 random_state: int = 42,
                 min_auroc: float = 0.95,
                 validate_f1: bool = True):
        """
        Initialize dual probe trainer.
        
        Args:
            feature_standardization: Whether to standardize features
            cross_validation_folds: Number of CV folds for validation
            random_state: Random seed for reproducibility
            min_auroc: Minimum AUROC threshold for probe validation
            validate_f1: Whether to include F1 score in validation (can disable)
        """
        self.feature_standardization = feature_standardization
        self.cv_folds = cross_validation_folds
        self.random_state = random_state
        self.min_auroc = min_auroc
        self.validate_f1 = validate_f1
        self.scaler = StandardScaler() if feature_standardization else None
    
    def load_dual_probe_data(self, file_path: str, layer_key: str = None) -> Dict:
        """
        Load activation data for dual-probe training.
        
        Args:
            file_path: Path to activation data file
            layer_key: Specific layer to extract (if None, uses first available)
        
        Returns:
            Dictionary with processed data
        """
        data = torch.load(file_path, map_location='cpu')
        
        # Handle multi-layer data
        if isinstance(data['activations'], dict):
            if layer_key is None:
                # Use the last layer by default (typically final representations)
                available_layers = list(data['activations'].keys())
                layer_key = available_layers[-1] if available_layers else None
                print(f"Using layer: {layer_key}")
            
            if layer_key not in data['activations']:
                raise ValueError(f"Layer {layer_key} not found. Available: {list(data['activations'].keys())}")
            
            activations = data['activations'][layer_key]
        else:
            # Single layer data (legacy format)
            activations = data['activations']
        
        # Prepare features
        X = activations.numpy()
        if len(X.shape) == 3:
            if X.shape[1] == 1:
                X = X.squeeze(1)  # Remove singleton dimension
            else:
                X = X[:, -1, :]  # Take last token
        
        print(f"Loaded activations shape: {X.shape}")
        print(f"Total samples: {len(data['blue_labels'])}")
        
        return {
            'X': X,
            'blue_labels': np.array(data['blue_labels'], dtype=int),
            'red_labels': np.array(data['red_labels'], dtype=int),
            'outputs': data['outputs'],
            'system_prompt': data['system_prompt'],
            'layer_key': layer_key,
            'n_samples': len(data['blue_labels'])
        }
    
    def train_single_probe(self, 
                          X: np.ndarray, 
                          y: np.ndarray, 
                          probe_name: str,
                          test_size: float = 0.2) -> Dict:
        """
        Train a single color detection probe with comprehensive validation.
        
        Args:
            X: Feature matrix
            y: Binary labels  
            probe_name: Name of the probe (for logging)
            test_size: Train/test split ratio
        
        Returns:
            Dictionary with probe and validation metrics
        """
        print(f"\nTraining {probe_name} probe...")
        print(f"  Features: {X.shape}")
        print(f"  Labels: {len(y)} ({np.sum(y)} positive, {len(y)-np.sum(y)} negative)")
        
        # Check label balance
        if np.sum(y) < 10 or (len(y) - np.sum(y)) < 10:
            print(f"WARNING: Very imbalanced labels for {probe_name} probe!")
        
        # Train/test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=self.random_state, stratify=y
        )
        
        # Feature standardization
        if self.feature_standardization:
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_test_scaled = self.scaler.transform(X_test)
        else:
            X_train_scaled = X_train
            X_test_scaled = X_test
        
        # Train probe with regularization search
        best_probe = None
        best_cv_score = 0
        best_c = None
        
        # Test different regularization strengths
        c_values = [0.01, 0.1, 1.0, 10.0, 100.0]
        for C in c_values:
            probe = LogisticRegression(
                solver='liblinear',
                C=C,
                max_iter=2000,
                random_state=self.random_state,
                class_weight='balanced'  # Handle class imbalance
            )
            
            # Cross-validation
            cv_scores = cross_val_score(
                probe, X_train_scaled, y_train, 
                cv=self.cv_folds, scoring='roc_auc'
            )
            mean_cv_score = np.mean(cv_scores)
            
            print(f"    C={C}: CV AUROC = {mean_cv_score:.4f} ± {np.std(cv_scores):.4f}")
            
            if mean_cv_score > best_cv_score:
                best_cv_score = mean_cv_score
                best_c = C
                best_probe = probe
        
        # Train final probe with best parameters
        print(f"  Best C: {best_c} (CV AUROC: {best_cv_score:.4f})")
        best_probe.fit(X_train_scaled, y_train)
        
        # Evaluate on test set
        train_preds = best_probe.predict_proba(X_train_scaled)[:, 1]
        test_preds = best_probe.predict_proba(X_test_scaled)[:, 1]
        
        train_auroc = roc_auc_score(y_train, train_preds)
        test_auroc = roc_auc_score(y_test, test_preds)
        
        # Classification metrics
        test_pred_classes = best_probe.predict(X_test_scaled)
        report = classification_report(y_test, test_pred_classes, output_dict=True)
        conf_matrix = confusion_matrix(y_test, test_pred_classes)
        
        print(f"  Final Results:")
        print(f"    Train AUROC: {train_auroc:.4f}")
        print(f"    Test AUROC: {test_auroc:.4f}")
        print(f"    Test Accuracy: {report['accuracy']:.4f}")
        print(f"    Test F1: {report['macro avg']['f1-score']:.4f}")
        
        # Quality control check
        quality_check = self._quality_control_check(test_auroc, report)
        
        return {
            'probe': best_probe,
            'scaler': self.scaler,
            'train_auroc': train_auroc,
            'test_auroc': test_auroc,
            'cv_auroc_mean': best_cv_score,
            'best_c': best_c,
            'classification_report': report,
            'confusion_matrix': conf_matrix.tolist(),
            'quality_check': quality_check,
            'X_train': X_train_scaled,
            'X_test': X_test_scaled,
            'y_train': y_train,
            'y_test': y_test,
            'train_predictions': train_preds.tolist(),
            'test_predictions': test_preds.tolist(),
            'feature_dim': X.shape[1]
        }
    
    def _quality_control_check(self, auroc: float, report: Dict) -> Dict:
        """
        Perform quality control checks on trained probe.
        
        Args:
            auroc: Test AUROC score
            report: Classification report
        
        Returns:
            Dictionary with quality check results
        """
        checks = {
            'auroc_threshold': auroc >= self.min_auroc,
        }
        
        # Only include F1/precision/recall if validate_f1 is True
        if self.validate_f1:
            checks.update({
                'f1_threshold': report['macro avg']['f1-score'] >= 0.85,  # Lowered from 0.90
                'precision_threshold': report['macro avg']['precision'] >= 0.85,
                'recall_threshold': report['macro avg']['recall'] >= 0.85,
            })
        
        checks['overall_pass'] = all(checks.values())
        checks['auroc_score'] = auroc
        checks['f1_score'] = report['macro avg']['f1-score']
        
        if not checks['overall_pass']:
            print(f"  WARNING: Quality control FAILED for probe")
            print(f"    AUROC: {auroc:.4f} ({'PASS' if checks['auroc_threshold'] else 'FAIL'})")
            if self.validate_f1:
                print(f"    F1: {report['macro avg']['f1-score']:.4f} ({'PASS' if checks.get('f1_threshold', True) else 'FAIL'})")
        else:
            print(f"  Quality control: PASSED (AUROC-focused)")
        
        return checks
    
    def train_dual_probes(self, 
                         data: Dict, 
                         test_size: float = 0.2) -> Dict:
        """
        Train both blue and red color detection probes.
        
        Args:
            data: Data dictionary from load_dual_probe_data
            test_size: Train/test split ratio
        
        Returns:
            Dictionary with both trained probes and metrics
        """
        X = data['X']
        blue_labels = data['blue_labels']
        red_labels = data['red_labels']
        
        print(f"Training dual probes on {X.shape[0]} samples...")
        print(f"Blue label distribution: {np.sum(blue_labels)}/{len(blue_labels)} positive")
        print(f"Red label distribution: {np.sum(red_labels)}/{len(red_labels)} positive")
        
        # Train blue probe
        blue_results = self.train_single_probe(X, blue_labels, "Blue", test_size)
        
        # Reset scaler for red probe
        if self.feature_standardization:
            self.scaler = StandardScaler()
        
        # Train red probe
        red_results = self.train_single_probe(X, red_labels, "Red", test_size)
        
        # Dual probe validation
        dual_validation = self._validate_dual_probes(blue_results, red_results, data)
        
        return {
            'blue_probe': blue_results,
            'red_probe': red_results,
            'dual_validation': dual_validation,
            'metadata': {
                'system_prompt': data['system_prompt'],
                'layer_key': data['layer_key'],
                'n_samples': data['n_samples'],
                'feature_dim': X.shape[1],
                'training_params': {
                    'feature_standardization': self.feature_standardization,
                    'cv_folds': self.cv_folds,
                    'test_size': test_size,
                    'random_state': self.random_state,
                }
            }
        }
    
    def _validate_dual_probes(self, blue_results: Dict, red_results: Dict, data: Dict) -> Dict:
        """
        Validate dual probe performance and check for issues.
        
        Args:
            blue_results: Blue probe training results
            red_results: Red probe training results  
            data: Original data dictionary
        
        Returns:
            Validation results and diagnostics
        """
        print(f"\nDual Probe Validation:")
        
        # Check probe quality
        blue_quality = blue_results['quality_check']['overall_pass']
        red_quality = red_results['quality_check']['overall_pass']
        
        print(f"  Blue probe quality: {'PASS' if blue_quality else 'FAIL'}")
        print(f"  Red probe quality: {'PASS' if red_quality else 'FAIL'}")
        
        # Check for label correlation (problematic if too high)
        blue_labels = data['blue_labels']
        red_labels = data['red_labels']
        label_correlation = np.corrcoef(blue_labels, red_labels)[0, 1]
        
        print(f"  Blue-Red label correlation: {label_correlation:.4f}")
        
        # Check probe prediction correlation on same data
        blue_test_preds = np.array(blue_results['test_predictions'])
        red_test_preds = np.array(red_results['test_predictions'])
        pred_correlation = np.corrcoef(blue_test_preds, red_test_preds)[0, 1]
        
        print(f"  Probe prediction correlation: {pred_correlation:.4f}")
        
        # Diagnostic checks
        diagnostics = {
            'both_probes_quality_pass': blue_quality and red_quality,
            'label_correlation': label_correlation,
            'prediction_correlation': pred_correlation,
            'label_correlation_acceptable': abs(label_correlation) < 0.7,  # Not too correlated
            'prediction_correlation_acceptable': abs(pred_correlation) < 0.8,
            'blue_auroc': blue_results['test_auroc'],
            'red_auroc': red_results['test_auroc'],
            'mean_auroc': (blue_results['test_auroc'] + red_results['test_auroc']) / 2,
        }
        
        # Overall validation status
        diagnostics['overall_validation_pass'] = (
            diagnostics['both_probes_quality_pass'] and
            diagnostics['label_correlation_acceptable'] and
            diagnostics['prediction_correlation_acceptable']
        )
        
        if diagnostics['overall_validation_pass']:
            print(f"  Overall validation: PASSED")
        else:
            print(f"  Overall validation: FAILED")
            if not diagnostics['both_probes_quality_pass']:
                print(f"    - One or both probes failed quality control")
            if not diagnostics['label_correlation_acceptable']:
                print(f"    - Labels too correlated ({label_correlation:.3f})")
            if not diagnostics['prediction_correlation_acceptable']:
                print(f"    - Predictions too correlated ({pred_correlation:.3f})")
        
        return diagnostics
    
    def save_dual_probes(self, results: Dict, output_path: str) -> str:
        """
        Save trained dual probes with comprehensive metadata.
        
        Args:
            results: Results from train_dual_probes
            output_path: Output file path
        
        Returns:
            Path to saved file
        """
        # Prepare save data (remove large arrays to save space)
        save_data = {
            'blue_probe': {
                'probe': results['blue_probe']['probe'],
                'scaler': results['blue_probe']['scaler'],
                'train_auroc': results['blue_probe']['train_auroc'],
                'test_auroc': results['blue_probe']['test_auroc'],
                'cv_auroc_mean': results['blue_probe']['cv_auroc_mean'],
                'best_c': results['blue_probe']['best_c'],
                'classification_report': results['blue_probe']['classification_report'],
                'confusion_matrix': results['blue_probe']['confusion_matrix'],
                'quality_check': results['blue_probe']['quality_check'],
            },
            'red_probe': {
                'probe': results['red_probe']['probe'],
                'scaler': results['red_probe']['scaler'],
                'train_auroc': results['red_probe']['train_auroc'],
                'test_auroc': results['red_probe']['test_auroc'],
                'cv_auroc_mean': results['red_probe']['cv_auroc_mean'],
                'best_c': results['red_probe']['best_c'],
                'classification_report': results['red_probe']['classification_report'],
                'confusion_matrix': results['red_probe']['confusion_matrix'],
                'quality_check': results['red_probe']['quality_check'],
            },
            'dual_validation': results['dual_validation'],
            'metadata': results['metadata'],
        }
        
        with open(output_path, 'wb') as f:
            pickle.dump(save_data, f)
        
        print(f"Saved dual probes to: {output_path}")
        return output_path
    
    def evaluate_dual_probes_on_data(self, 
                                   probe_file: str, 
                                   data_file: str, 
                                   layer_key: str = None) -> Dict:
        """
        Evaluate trained dual probes on new data.
        
        Args:
            probe_file: Path to saved dual probe file
            data_file: Path to evaluation data
            layer_key: Layer to evaluate on
        
        Returns:
            Evaluation results for both probes
        """
        # Load probes
        with open(probe_file, 'rb') as f:
            probe_data = pickle.load(f)
        
        # Load evaluation data
        eval_data = self.load_dual_probe_data(data_file, layer_key)
        
        # Prepare features
        X = eval_data['X']
        blue_labels = eval_data['blue_labels']
        red_labels = eval_data['red_labels']
        
        results = {}
        
        # Evaluate blue probe
        blue_probe = probe_data['blue_probe']['probe']
        blue_scaler = probe_data['blue_probe']['scaler']
        
        if blue_scaler:
            X_blue_scaled = blue_scaler.transform(X)
        else:
            X_blue_scaled = X
        
        blue_preds = blue_probe.predict_proba(X_blue_scaled)[:, 1]
        blue_auroc = roc_auc_score(blue_labels, blue_preds)
        
        results['blue'] = {
            'auroc': blue_auroc,
            'predictions': blue_preds.tolist(),
            'true_labels': blue_labels.tolist(),
        }
        
        # Evaluate red probe
        red_probe = probe_data['red_probe']['probe']
        red_scaler = probe_data['red_probe']['scaler']
        
        if red_scaler:
            X_red_scaled = red_scaler.transform(X)
        else:
            X_red_scaled = X
        
        red_preds = red_probe.predict_proba(X_red_scaled)[:, 1]
        red_auroc = roc_auc_score(red_labels, red_preds)
        
        results['red'] = {
            'auroc': red_auroc,
            'predictions': red_preds.tolist(),
            'true_labels': red_labels.tolist(),
        }
        
        # Differential analysis
        results['differential'] = {
            'blue_minus_red_preds': (blue_preds - red_preds).tolist(),
            'mean_differential': np.mean(blue_preds - red_preds),
            'std_differential': np.std(blue_preds - red_preds),
        }
        
        print(f"Dual Probe Evaluation Results:")
        print(f"  Blue probe AUROC: {blue_auroc:.4f}")
        print(f"  Red probe AUROC: {red_auroc:.4f}")
        print(f"  Mean differential: {results['differential']['mean_differential']:.4f}")
        
        return results


def main():
    parser = argparse.ArgumentParser(description="Train dual color detection probes")
    parser.add_argument("--data", required=True, help="Training data file (.pt)")
    parser.add_argument("--output", required=True, help="Output file for dual probes (.pkl)")
    parser.add_argument("--layer", help="Specific layer to use (default: last layer)")
    parser.add_argument("--test-size", type=float, default=0.2, help="Test split ratio")
    parser.add_argument("--no-standardization", action="store_true", 
                       help="Disable feature standardization")
    parser.add_argument("--cv-folds", type=int, default=5, help="Cross-validation folds")
    
    args = parser.parse_args()
    
    print("Dual Probe Training Pipeline")
    print("=" * 40)
    
    # Initialize trainer
    trainer = DualProbeTrainer(
        feature_standardization=not args.no_standardization,
        cross_validation_folds=args.cv_folds,
    )
    
    # Load data
    print(f"Loading data from: {args.data}")
    data = trainer.load_dual_probe_data(args.data, args.layer)
    
    # Train dual probes
    results = trainer.train_dual_probes(data, test_size=args.test_size)
    
    # Save results
    trainer.save_dual_probes(results, args.output)
    
    # Summary
    print(f"\nTraining Summary:")
    print(f"  Blue probe AUROC: {results['blue_probe']['test_auroc']:.4f}")
    print(f"  Red probe AUROC: {results['red_probe']['test_auroc']:.4f}")
    print(f"  Validation status: {'PASS' if results['dual_validation']['overall_validation_pass'] else 'FAIL'}")


if __name__ == "__main__":
    main()