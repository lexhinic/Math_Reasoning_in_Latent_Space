# evaluation/metrics.py
import numpy as np
import torch
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
from tqdm import tqdm
from data_processing.parser import HOLLightParser

def compute_roc_auc(y_true, y_pred):
    """
    Compute ROC curve and AUC.
    
    Args:
        y_true: True labels
        y_pred: Predicted scores
        
    Returns:
        fpr: False positive rate
        tpr: True positive rate
        auc_score: Area under the ROC curve
    """
    fpr, tpr, _ = roc_curve(y_true, y_pred)
    auc_score = auc(fpr, tpr)
    
    return fpr, tpr, auc_score

def evaluate_multi_step_reasoning(model, datasets, device, max_steps=9):
    """
    Evaluate multi-step reasoning in latent space.
    
    Args:
        model: Model to evaluate
        datasets: List of datasets, one for each rewrite step
        device: Device to use for evaluation
        max_steps: Maximum number of rewrite steps
        
    Returns:
        results: Dictionary of evaluation results
    """
    model.eval()
    parser = HOLLightParser()
    
    results = {
        "true_fpr": [],
        "true_tpr": [],
        "true_auc": [],
        "one_step_fpr": [],
        "one_step_tpr": [],
        "one_step_auc": [],
        "multi_step_fpr": [],
        "multi_step_tpr": [],
        "multi_step_auc": [],
        "l2_distances_one_step": [],
        "l2_distances_multi_step": [],
        "true_success_probs": [],
        "one_step_success_probs": [],
        "multi_step_success_probs": [],
        "all_success_labels": []
    }
    
    # For each step
    for step in range(min(max_steps + 1, len(datasets))):
        print(f"Evaluating step {step}")
        
        # Get the dataset for this step
        dataset = datasets[step]
        
        # Compute true embeddings
        true_embeddings = []
        all_success_labels = []
        
        for t_statement, p_statement, success, result in tqdm(dataset, desc=f"Computing true embeddings (step {step})"):
            # Parse the theorems
            t_nodes, t_edges, t_types = parser.parse_formula(t_statement)
            
            # Convert to tensors and move to device
            t_nodes = t_nodes.to(device)
            t_edges = t_edges.to(device)
            t_types = t_types.to(device)
            
            # Compute embeddings
            with torch.no_grad():
                theorem_embed = model.theorem_embed(t_nodes, t_edges, t_types)
                true_embeddings.append(theorem_embed.cpu())
            
            all_success_labels.append(success)
        
        # True evaluation
        true_success_probs = []
        
        for i, (t_statement, p_statement, success, result) in tqdm(enumerate(dataset), desc=f"True evaluation (step {step})"):
            # Parse the parameter
            p_nodes, p_edges, p_types = parser.parse_formula(p_statement)
            
            # Convert to tensors and move to device
            p_nodes = p_nodes.to(device)
            p_edges = p_edges.to(device)
            p_types = p_types.to(device)
            
            # Compute embeddings
            with torch.no_grad():
                theorem_embed = true_embeddings[i].to(device)
                param_embed = model.param_embed(p_nodes, p_edges, p_types)
                
                # Predict rewrite success
                success_prob = model.predict_rewrite_success(theorem_embed.unsqueeze(0), 
                                                           param_embed.unsqueeze(0))
                
                true_success_probs.append(success_prob.item())
        
        # Compute ROC and AUC for true evaluation
        true_fpr, true_tpr, true_auc = compute_roc_auc(all_success_labels, true_success_probs)
        
        results["true_fpr"].append(true_fpr)
        results["true_tpr"].append(true_tpr)
        results["true_auc"].append(true_auc)
        results["true_success_probs"].append(true_success_probs)
        results["all_success_labels"].append(all_success_labels)
        
        # One-step evaluation (predicting embeddings directly)
        if step > 0:
            one_step_embeddings = []
            one_step_success_probs = []
            
            # For each example in the previous step's dataset
            for i, (prev_t, prev_p, prev_success, t_statement) in tqdm(enumerate(datasets[step-1]), 
                                                                     desc=f"One-step evaluation (step {step})"):
                if prev_success:
                    # Parse the theorems
                    t_nodes, t_edges, t_types = parser.parse_formula(prev_t)
                    p_nodes, p_edges, p_types = parser.parse_formula(prev_p)
                    
                    # Convert to tensors and move to device
                    t_nodes = t_nodes.to(device)
                    t_edges = t_edges.to(device)
                    t_types = t_types.to(device)
                    p_nodes = p_nodes.to(device)
                    p_edges = p_edges.to(device)
                    p_types = p_types.to(device)
                    
                    # Compute embeddings
                    with torch.no_grad():
                        theorem_embed = model.theorem_embed(t_nodes, t_edges, t_types)
                        param_embed = model.param_embed(p_nodes, p_edges, p_types)
                        
                        # Predict result embedding
                        result_embed = model.predict_result_embedding(
                            theorem_embed.unsqueeze(0), 
                            param_embed.unsqueeze(0)
                        )
                        
                        one_step_embeddings.append(result_embed[0].cpu())
            
            # Evaluate the predicted embeddings on the current step's dataset
            for i, (t_statement, p_statement, success, result) in enumerate(dataset):
                if i < len(one_step_embeddings):
                    # Parse the parameter
                    p_nodes, p_edges, p_types = parser.parse_formula(p_statement)
                    
                    # Convert to tensors and move to device
                    p_nodes = p_nodes.to(device)
                    p_edges = p_edges.to(device)
                    p_types = p_types.to(device)
                    
                    # Compute embeddings
                    with torch.no_grad():
                        theorem_embed = one_step_embeddings[i].to(device)
                        param_embed = model.param_embed(p_nodes, p_edges, p_types)
                        
                        # Predict rewrite success
                        success_prob = model.predict_rewrite_success(
                            theorem_embed.unsqueeze(0), 
                            param_embed.unsqueeze(0)
                        )
                        
                        one_step_success_probs.append(success_prob.item())
            
            # Compute ROC and AUC for one-step evaluation
            if one_step_success_probs:
                labels = all_success_labels[:len(one_step_success_probs)]
                one_step_fpr, one_step_tpr, one_step_auc = compute_roc_auc(labels, one_step_success_probs)
                
                results["one_step_fpr"].append(one_step_fpr)
                results["one_step_tpr"].append(one_step_tpr)
                results["one_step_auc"].append(one_step_auc)
                results["one_step_success_probs"].append(one_step_success_probs)
                
                # Compute L2 distances for one-step evaluation
                l2_distances = []
                
                for i in range(min(len(true_embeddings), len(one_step_embeddings))):
                    l2_dist = torch.norm(true_embeddings[i] - one_step_embeddings[i]).item()
                    l2_distances.append(l2_dist)
                
                results["l2_distances_one_step"].append(l2_distances)
        
        # Multi-step evaluation (propagating in latent space)
        if step > 0:
            multi_step_embeddings = []
            multi_step_success_probs = []
            
            # Initialize with the true embeddings of the first step
            if step == 1:
                prev_embeddings = [true_embeddings[0].clone() for _ in range(len(datasets[0]))]
            
            # For each example in the previous step's dataset
            for i, (prev_t, prev_p, prev_success, t_statement) in tqdm(enumerate(datasets[step-1]), 
                                                                     desc=f"Multi-step evaluation (step {step})"):
                if prev_success and i < len(prev_embeddings):
                    # Parse the parameter
                    p_nodes, p_edges, p_types = parser.parse_formula(prev_p)
                    
                    # Convert to tensors and move to device
                    p_nodes = p_nodes.to(device)
                    p_edges = p_edges.to(device)
                    p_types = p_types.to(device)
                    
                    # Compute embeddings
                    with torch.no_grad():
                        theorem_embed = prev_embeddings[i].to(device)
                        param_embed = model.param_embed(p_nodes, p_edges, p_types)
                        
                        # Predict result embedding
                        result_embed = model.predict_result_embedding(
                            theorem_embed.unsqueeze(0), 
                            param_embed.unsqueeze(0)
                        )
                        
                        multi_step_embeddings.append(result_embed[0].cpu())
            
            # Update previous embeddings for the next step
            prev_embeddings = multi_step_embeddings
            
            # Evaluate the predicted embeddings on the current step's dataset
            for i, (t_statement, p_statement, success, result) in enumerate(dataset):
                if i < len(multi_step_embeddings):
                    # Parse the parameter
                    p_nodes, p_edges, p_types = parser.parse_formula(p_statement)
                    
                    # Convert to tensors and move to device
                    p_nodes = p_nodes.to(device)
                    p_edges = p_edges.to(device)
                    p_types = p_types.to(device)
                    
                    # Compute embeddings
                    with torch.no_grad():
                        theorem_embed = multi_step_embeddings[i].to(device)
                        param_embed = model.param_embed(p_nodes, p_edges, p_types)
                        
                        # Predict rewrite success
                        success_prob = model.predict_rewrite_success(
                            theorem_embed.unsqueeze(0), 
                            param_embed.unsqueeze(0)
                        )
                        
                        multi_step_success_probs.append(success_prob.item())
            
            # Compute ROC and AUC for multi-step evaluation
            if multi_step_success_probs:
                labels = all_success_labels[:len(multi_step_success_probs)]
                multi_step_fpr, multi_step_tpr, multi_step_auc = compute_roc_auc(labels, multi_step_success_probs)
                
                results["multi_step_fpr"].append(multi_step_fpr)
                results["multi_step_tpr"].append(multi_step_tpr)
                results["multi_step_auc"].append(multi_step_auc)
                results["multi_step_success_probs"].append(multi_step_success_probs)
                
                # Compute L2 distances for multi-step evaluation
                l2_distances = []
                
                for i in range(min(len(true_embeddings), len(multi_step_embeddings))):
                    l2_dist = torch.norm(true_embeddings[i] - multi_step_embeddings[i]).item()
                    l2_distances.append(l2_dist)
                
                results["l2_distances_multi_step"].append(l2_distances)
    
    return results