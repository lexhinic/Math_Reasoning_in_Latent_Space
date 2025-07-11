import os
import json
import logging
from typing import Dict, List, Tuple, Any, Optional
import torch
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

from models.reasoning_model import ReasoningModel
from baseline.Math_Reasoning_in_Latent_Space.lean.data.ori.dataset import MultiStepReasoningDataset
from data.formula_parser import LeanFormulaParser
from baseline.Math_Reasoning_in_Latent_Space.lean.data.ori.lean_interface import LeanRewriteEngine, LeanTheorem
import sys 
sys.path.append('/home/stu4/formal_reasoning/baseline/Math_Reasoning_in_Latent_Space/')
from baseline.Math_Reasoning_in_Latent_Space.lean.config import config

logger = logging.getLogger(__name__)

class MultiStepEvaluator:
    """Evaluator for multi-step reasoning capabilities"""
    
    def __init__(self,
                 model: ReasoningModel,
                 parser: LeanFormulaParser,
                 device: str = "cuda"):
        """
        Initialize evaluator
        
        Args:
            model: Trained reasoning model
            parser: Formula parser
            device: Device to run evaluation on
        """
        self.model = model.to(device)
        self.model.eval()
        self.parser = parser
        self.device = device
        
    def evaluate_multi_step_reasoning(self,
                                    test_theorems: List[LeanTheorem],
                                    max_steps: int = 9) -> Dict[str, Any]:
        """
        Evaluate multi-step reasoning performance
        
        Args:
            test_theorems: Test theorems to evaluate on
            max_steps: Maximum number of reasoning steps
            
        Returns:
            Dictionary containing evaluation results
        """
        results = {
            'step_results': {},
            'overall_metrics': {},
            'detailed_results': []
        }
        
        logger.info(f"Evaluating multi-step reasoning up to {max_steps} steps")
        
        # Generate evaluation data for each step
        for num_steps in range(max_steps + 1):
            logger.info(f"Evaluating {num_steps} step reasoning")
            
            step_results = self._evaluate_step_reasoning(test_theorems, num_steps)
            results['step_results'][num_steps] = step_results
            
            logger.info(f"Step {num_steps} - AUC: {step_results['auc']:.4f}, "
                       f"Accuracy: {step_results['accuracy']:.4f}")
        
        # Compute overall metrics
        results['overall_metrics'] = self._compute_overall_metrics(results['step_results'])
        
        return results
    
    def _evaluate_step_reasoning(self, 
                                test_theorems: List[LeanTheorem],
                                num_steps: int) -> Dict[str, Any]:
        """
        Evaluate reasoning for a specific number of steps
        
        Args:
            test_theorems: Test theorems
            num_steps: Number of reasoning steps
            
        Returns:
            Evaluation results for this step count
        """
        if num_steps == 0:
            return self._evaluate_direct_prediction(test_theorems)
        else:
            return self._evaluate_propagated_prediction(test_theorems, num_steps)
    
    def _evaluate_direct_prediction(self, test_theorems: List[LeanTheorem]) -> Dict[str, Any]:
        """
        Evaluate direct rewrite prediction (0 steps)
        
        Args:
            test_theorems: Test theorems
            
        Returns:
            Evaluation results
        """
        all_predictions = []
        all_targets = []
        
        with torch.no_grad():
            for i, target_theorem in enumerate(tqdm(test_theorems, desc="Direct prediction")):
                # Get all possible parameters for this theorem
                parameter_theorems = test_theorems[:i]  # Only use earlier theorems
                
                if not parameter_theorems:
                    continue
                
                # Convert to graphs
                target_graph = self.parser.formula_to_graph(target_theorem.statement).to(self.device)
                
                for param_theorem in parameter_theorems:
                    param_graph = self.parser.formula_to_graph(param_theorem.statement).to(self.device)
                    
                    # Create batch
                    target_batch = target_graph.clone()
                    target_batch.batch = torch.zeros(target_graph.x.size(0), dtype=torch.long, device=self.device)
                    
                    param_batch = param_graph.clone()
                    param_batch.batch = torch.zeros(param_graph.x.size(0), dtype=torch.long, device=self.device)
                    
                    # Forward pass
                    success_logits, _ = self.model(target_batch, param_batch)
                    prediction = torch.sigmoid(success_logits).item()
                    
                    # For evaluation, we need ground truth - this would come from actual rewrite attempts
                    # For now, we'll use a simplified simulation
                    target = self._simulate_rewrite_success(target_theorem, param_theorem)
                    
                    all_predictions.append(prediction)
                    all_targets.append(target)
        
        # Compute metrics
        if all_predictions and all_targets:
            auc = roc_auc_score(all_targets, all_predictions)
            accuracy = np.mean((np.array(all_predictions) > 0.5) == np.array(all_targets))
            
            return {
                'auc': auc,
                'accuracy': accuracy,
                'num_samples': len(all_predictions),
                'predictions': all_predictions,
                'targets': all_targets
            }
        else:
            return {
                'auc': 0.0,
                'accuracy': 0.0,
                'num_samples': 0,
                'predictions': [],
                'targets': []
            }
    
    def _evaluate_propagated_prediction(self, 
                                      test_theorems: List[LeanTheorem],
                                      num_steps: int) -> Dict[str, Any]:
        """
        Evaluate prediction after multiple reasoning steps in latent space
        
        Args:
            test_theorems: Test theorems
            num_steps: Number of reasoning steps
            
        Returns:
            Evaluation results
        """
        all_predictions = []
        all_targets = []
        l2_distances = []
        
        with torch.no_grad():
            for theorem_idx in tqdm(range(len(test_theorems)), desc=f"{num_steps}-step reasoning"):
                # Generate a reasoning chain
                chain = self._generate_reasoning_chain(test_theorems, theorem_idx, num_steps)
                
                if not chain:
                    continue
                
                # Get initial embedding
                initial_formula = chain[0]['formula']
                initial_graph = self.parser.formula_to_graph(initial_formula).to(self.device)
                initial_batch = initial_graph.clone()
                initial_batch.batch = torch.zeros(initial_graph.x.size(0), dtype=torch.long, device=self.device)
                
                initial_embedding = self.model.encode_target(
                    initial_batch.x, initial_batch.edge_index, initial_batch.batch
                )
                
                # Collect parameter embeddings
                parameter_embeddings = []
                for step in chain[1:]:
                    param_graph = self.parser.formula_to_graph(step['parameter']).to(self.device)
                    param_batch = param_graph.clone()
                    param_batch.batch = torch.zeros(param_graph.x.size(0), dtype=torch.long, device=self.device)
                    
                    param_embedding = self.model.encode_parameter(
                        param_batch.x, param_batch.edge_index, param_batch.batch
                    )
                    parameter_embeddings.append(param_embedding)
                
                # Stack parameter embeddings
                param_stack = torch.stack(parameter_embeddings, dim=1)  # [1, num_steps, embedding_dim]
                
                # Perform reasoning in latent space
                final_predicted_embedding = self.model.reason_in_latent_space(
                    initial_embedding, param_stack
                )
                
                # Get true final embedding
                final_formula = chain[-1]['formula']
                final_graph = self.parser.formula_to_graph(final_formula).to(self.device)
                final_batch = final_graph.clone()
                final_batch.batch = torch.zeros(final_graph.x.size(0), dtype=torch.long, device=self.device)
                
                true_final_embedding = self.model.encode_target(
                    final_batch.x, final_batch.edge_index, final_batch.batch
                )
                
                # Compute L2 distance
                l2_dist = F.mse_loss(final_predicted_embedding, true_final_embedding).item()
                l2_distances.append(l2_dist)
                
                # Evaluate rewrite success prediction using predicted embedding
                test_parameters = test_theorems[:theorem_idx]
                
                for param_theorem in test_parameters[:min(len(test_parameters), 10)]:  # Limit for efficiency
                    param_graph = self.parser.formula_to_graph(param_theorem.statement).to(self.device)
                    param_batch = param_graph.clone()
                    param_batch.batch = torch.zeros(param_graph.x.size(0), dtype=torch.long, device=self.device)
                    
                    param_embedding = self.model.encode_parameter(
                        param_batch.x, param_batch.edge_index, param_batch.batch
                    )
                    
                    # Predict success using predicted embedding
                    prediction = self.model.evaluate_rewrite_success(
                        final_predicted_embedding, param_embedding
                    ).item()
                    
                    # Get ground truth (simplified simulation)
                    target = self._simulate_rewrite_success_from_formula(final_formula, param_theorem.statement)
                    
                    all_predictions.append(prediction)
                    all_targets.append(target)
        
        # Compute metrics
        if all_predictions and all_targets:
            auc = roc_auc_score(all_targets, all_predictions)
            accuracy = np.mean((np.array(all_predictions) > 0.5) == np.array(all_targets))
            avg_l2_distance = np.mean(l2_distances) if l2_distances else 0.0
            
            return {
                'auc': auc,
                'accuracy': accuracy,
                'avg_l2_distance': avg_l2_distance,
                'num_samples': len(all_predictions),
                'predictions': all_predictions,
                'targets': all_targets,
                'l2_distances': l2_distances
            }
        else:
            return {
                'auc': 0.0,
                'accuracy': 0.0,
                'avg_l2_distance': 0.0,
                'num_samples': 0,
                'predictions': [],
                'targets': [],
                'l2_distances': []
            }
    
    def _generate_reasoning_chain(self, 
                                theorems: List[LeanTheorem],
                                start_idx: int,
                                num_steps: int) -> List[Dict[str, str]]:
        """
        Generate a reasoning chain starting from a theorem
        
        Args:
            theorems: Available theorems
            start_idx: Index of starting theorem
            num_steps: Number of reasoning steps
            
        Returns:
            List of reasoning steps
        """
        if start_idx >= len(theorems):
            return []
        
        chain = [{'formula': theorems[start_idx].statement, 'parameter': None}]
        current_formula = theorems[start_idx].statement
        
        # Randomly select parameters for each step
        available_params = theorems[:start_idx]
        
        if len(available_params) < num_steps:
            return []
        
        selected_params = np.random.choice(available_params, num_steps, replace=False)
        
        for param_theorem in selected_params:
            # Simulate applying the parameter (simplified)
            new_formula = self._simulate_rewrite_application(current_formula, param_theorem.statement)
            
            chain.append({
                'formula': new_formula,
                'parameter': param_theorem.statement
            })
            
            current_formula = new_formula
        
        return chain
    
    def _simulate_rewrite_success(self, target_theorem: LeanTheorem, param_theorem: LeanTheorem) -> float:
        """
        Simulate rewrite success (placeholder implementation)
        
        Args:
            target_theorem: Target theorem
            param_theorem: Parameter theorem
            
        Returns:
            Simulated success probability (0 or 1)
        """
        # This is a simplified simulation - in practice, you'd use actual Lean rewriting
        # For now, we'll use some heuristics based on formula structure
        
        target_tokens = set(target_theorem.statement.split())
        param_tokens = set(param_theorem.statement.split())
        
        # Simple heuristic: if they share common tokens, more likely to succeed
        intersection = target_tokens.intersection(param_tokens)
        
        if len(intersection) > 2:
            return 1.0 if np.random.random() > 0.3 else 0.0
        elif len(intersection) > 0:
            return 1.0 if np.random.random() > 0.7 else 0.0
        else:
            return 1.0 if np.random.random() > 0.9 else 0.0
    
    def _simulate_rewrite_success_from_formula(self, formula: str, parameter: str) -> float:
        """
        Simulate rewrite success from formula strings
        
        Args:
            formula: Target formula string
            parameter: Parameter formula string
            
        Returns:
            Simulated success (0 or 1)
        """
        # Similar heuristic as above
        formula_tokens = set(formula.split())
        param_tokens = set(parameter.split())
        
        intersection = formula_tokens.intersection(param_tokens)
        
        if len(intersection) > 2:
            return 1.0 if np.random.random() > 0.3 else 0.0
        elif len(intersection) > 0:
            return 1.0 if np.random.random() > 0.7 else 0.0
        else:
            return 1.0 if np.random.random() > 0.9 else 0.0
    
    def _simulate_rewrite_application(self, formula: str, parameter: str) -> str:
        """
        Simulate applying a rewrite parameter to a formula
        
        Args:
            formula: Current formula
            parameter: Rewrite parameter
            
        Returns:
            Modified formula (simplified simulation)
        """
        # Very simplified simulation - just add some variation
        # In practice, this would be done by actual Lean rewriting
        
        tokens = formula.split()
        if len(tokens) > 3:
            # Randomly modify part of the formula
            idx = np.random.randint(1, len(tokens) - 1)
            tokens[idx] = f"rewritten_{tokens[idx]}"
        
        return " ".join(tokens)
    
    def _compute_overall_metrics(self, step_results: Dict[int, Dict[str, Any]]) -> Dict[str, Any]:
        """
        Compute overall evaluation metrics across all steps
        
        Args:
            step_results: Results for each step count
            
        Returns:
            Overall metrics
        """
        aucs = [results['auc'] for results in step_results.values() if results['num_samples'] > 0]
        accuracies = [results['accuracy'] for results in step_results.values() if results['num_samples'] > 0]
        
        l2_distances = []
        for step, results in step_results.items():
            if step > 0 and 'l2_distances' in results:
                l2_distances.extend(results['l2_distances'])
        
        return {
            'mean_auc': np.mean(aucs) if aucs else 0.0,
            'std_auc': np.std(aucs) if aucs else 0.0,
            'mean_accuracy': np.mean(accuracies) if accuracies else 0.0,
            'std_accuracy': np.std(accuracies) if accuracies else 0.0,
            'mean_l2_distance': np.mean(l2_distances) if l2_distances else 0.0,
            'std_l2_distance': np.std(l2_distances) if l2_distances else 0.0,
            'num_step_evaluations': len(aucs)
        }
    
    def plot_results(self, results: Dict[str, Any], save_dir: str):
        """
        Plot evaluation results
        
        Args:
            results: Evaluation results
            save_dir: Directory to save plots
        """
        os.makedirs(save_dir, exist_ok=True)
        
        # Plot AUC vs number of steps
        steps = list(results['step_results'].keys())
        aucs = [results['step_results'][step]['auc'] for step in steps]
        accuracies = [results['step_results'][step]['accuracy'] for step in steps]
        
        plt.figure(figsize=(12, 5))
        
        # AUC plot
        plt.subplot(1, 2, 1)
        plt.plot(steps, aucs, 'bo-', label='AUC')
        plt.xlabel('Number of Reasoning Steps')
        plt.ylabel('AUC')
        plt.title('AUC vs Number of Reasoning Steps')
        plt.grid(True)
        plt.legend()
        
        # Accuracy plot
        plt.subplot(1, 2, 2)
        plt.plot(steps, accuracies, 'ro-', label='Accuracy')
        plt.xlabel('Number of Reasoning Steps')
        plt.ylabel('Accuracy')
        plt.title('Accuracy vs Number of Reasoning Steps')
        plt.grid(True)
        plt.legend()
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'step_performance.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        # Plot L2 distances
        steps_with_l2 = [step for step in steps if step > 0 and 'avg_l2_distance' in results['step_results'][step]]
        l2_distances = [results['step_results'][step]['avg_l2_distance'] for step in steps_with_l2]
        
        if l2_distances:
            plt.figure(figsize=(8, 6))
            plt.plot(steps_with_l2, l2_distances, 'go-')
            plt.xlabel('Number of Reasoning Steps')
            plt.ylabel('Average L2 Distance')
            plt.title('Embedding Quality vs Number of Reasoning Steps')
            plt.grid(True)
            plt.savefig(os.path.join(save_dir, 'l2_distances.png'), dpi=300, bbox_inches='tight')
            plt.close()
        
        # Plot ROC curves for different steps
        plt.figure(figsize=(10, 8))
        colors = plt.cm.viridis(np.linspace(0, 1, len(steps)))
        
        for i, step in enumerate(steps):
            step_result = results['step_results'][step]
            if step_result['num_samples'] > 0:
                fpr, tpr, _ = roc_curve(step_result['targets'], step_result['predictions'])
                plt.plot(fpr, tpr, color=colors[i], label=f'Step {step} (AUC={step_result["auc"]:.3f})')
        
        plt.plot([0, 1], [0, 1], 'k--', alpha=0.5, label='Random')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curves for Different Reasoning Steps')
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(save_dir, 'roc_curves.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Evaluation plots saved to {save_dir}")
    
    def save_results(self, results: Dict[str, Any], save_path: str):
        """
        Save evaluation results to file
        
        Args:
            results: Evaluation results
            save_path: Path to save results
        """
        # Convert numpy arrays to lists for JSON serialization
        serializable_results = {}
        
        for key, value in results.items():
            if key == 'step_results':
                serializable_results[key] = {}
                for step, step_result in value.items():
                    serializable_results[key][step] = {}
                    for metric, metric_value in step_result.items():
                        if isinstance(metric_value, (list, np.ndarray)):
                            serializable_results[key][step][metric] = list(metric_value)
                        else:
                            serializable_results[key][step][metric] = metric_value
            else:
                serializable_results[key] = value
        
        with open(save_path, 'w') as f:
            json.dump(serializable_results, f, indent=2)
        
        logger.info(f"Evaluation results saved to {save_path}")

def run_evaluation(model: ReasoningModel,
                  test_theorems: List[LeanTheorem],
                  parser: LeanFormulaParser,
                  save_dir: str,
                  max_steps: int = 9) -> Dict[str, Any]:
    """
    Run complete evaluation pipeline
    
    Args:
        model: Trained reasoning model
        test_theorems: Test theorems
        parser: Formula parser
        save_dir: Directory to save results
        max_steps: Maximum number of reasoning steps
        
    Returns:
        Evaluation results
    """
    os.makedirs(save_dir, exist_ok=True)
    
    evaluator = MultiStepEvaluator(model, parser, config.device)
    
    logger.info("Starting multi-step reasoning evaluation")
    results = evaluator.evaluate_multi_step_reasoning(test_theorems, max_steps)
    
    # Save results
    evaluator.save_results(results, os.path.join(save_dir, 'evaluation_results.json'))
    
    # Plot results
    evaluator.plot_results(results, save_dir)
    
    # Print summary
    logger.info("Evaluation Summary:")
    logger.info(f"Mean AUC across steps: {results['overall_metrics']['mean_auc']:.4f} ± {results['overall_metrics']['std_auc']:.4f}")
    logger.info(f"Mean Accuracy across steps: {results['overall_metrics']['mean_accuracy']:.4f} ± {results['overall_metrics']['std_accuracy']:.4f}")
    
    if results['overall_metrics']['mean_l2_distance'] > 0:
        logger.info(f"Mean L2 distance: {results['overall_metrics']['mean_l2_distance']:.4f} ± {results['overall_metrics']['std_l2_distance']:.4f}")
    
    return results