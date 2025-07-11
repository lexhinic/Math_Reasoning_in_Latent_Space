import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import seaborn as sns

def plot_roc_curves(results, step=9):
    """
    Plot ROC curves for different evaluation methods.
    
    Args:
        results: Dictionary of evaluation results
        step: Rewrite step to plot
        
    Returns:
        fig: Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Plot true evaluation ROC curve
    if step < len(results["true_auc"]):
        ax.plot(
            results["true_fpr"][step],
            results["true_tpr"][step],
            label=f"True (AUC = {results['true_auc'][step]:.3f})",
            color="blue"
        )
    
    # Plot one-step evaluation ROC curve
    if step > 0 and step-1 < len(results["one_step_auc"]):
        ax.plot(
            results["one_step_fpr"][step-1],
            results["one_step_tpr"][step-1],
            label=f"One Step (AUC = {results['one_step_auc'][step-1]:.3f})",
            color="green"
        )
    
    # Plot multi-step evaluation ROC curve
    if step > 0 and step-1 < len(results["multi_step_auc"]):
        ax.plot(
            results["multi_step_fpr"][step-1],
            results["multi_step_tpr"][step-1],
            label=f"Multi Step (AUC = {results['multi_step_auc'][step-1]:.3f})",
            color="red"
        )
    
    # Plot random baseline
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Random Baseline")
    
    # Set labels and title
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(f"ROC Curves for Rewrite Step {step}")
    
    # Add legend
    ax.legend()
    
    # Set axis limits
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    
    return fig

def plot_auc_vs_steps(results, max_steps=9):
    """
    Plot AUC versus rewrite steps.
    
    Args:
        results: Dictionary of evaluation results
        max_steps: Maximum number of rewrite steps
        
    Returns:
        fig: Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Plot true evaluation AUC
    ax.plot(
        range(len(results["true_auc"])),
        results["true_auc"],
        label="True",
        color="blue",
        marker="o"
    )
    
    # Plot one-step evaluation AUC
    if results["one_step_auc"]:
        one_step_auc = [0] + results["one_step_auc"]
        ax.plot(
            range(len(one_step_auc)),
            one_step_auc,
            label="One Step",
            color="green",
            marker="s"
        )
    
    # Plot multi-step evaluation AUC
    if results["multi_step_auc"]:
        multi_step_auc = [0] + results["multi_step_auc"]
        ax.plot(
            range(len(multi_step_auc)),
            multi_step_auc,
            label="Multi Step",
            color="red",
            marker="^"
        )
    
    # Set labels and title
    ax.set_xlabel("Rewrite Steps")
    ax.set_ylabel("Area Under ROC Curve")
    ax.set_title("AUC versus Rewrite Steps")
    
    # Add legend
    ax.legend()
    
    # Set axis limits
    ax.set_xlim([0, max_steps])
    ax.set_ylim([0.5, 1.0])
    
    # Add grid
    ax.grid(True)
    
    return fig

def plot_l2_distances(results, max_steps=9):
    """
    Plot L2 distances versus rewrite steps.
    
    Args:
        results: Dictionary of evaluation results
        max_steps: Maximum number of rewrite steps
        
    Returns:
        fig: Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Compute mean L2 distances for one-step evaluation
    if results["l2_distances_one_step"]:
        mean_l2_one_step = [np.mean(distances) for distances in results["l2_distances_one_step"]]
        
        # Plot mean L2 distances
        ax.plot(
            range(1, len(mean_l2_one_step) + 1),
            mean_l2_one_step,
            label="One Step",
            color="green",
            marker="s"
        )
    
    # Compute mean L2 distances for multi-step evaluation
    if results["l2_distances_multi_step"]:
        mean_l2_multi_step = [np.mean(distances) for distances in results["l2_distances_multi_step"]]
        
        # Plot mean L2 distances
        ax.plot(
            range(1, len(mean_l2_multi_step) + 1),
            mean_l2_multi_step,
            label="Multi Step",
            color="red",
            marker="^"
        )
    
    # Plot random baseline
    # Assuming the random baseline is the mean distance between random embeddings
    random_baseline = np.sqrt(2 * 1024)  # For 1024-dimensional embeddings
    ax.axhline(y=random_baseline, color="gray", linestyle="--", label="Random Baseline")
    
    # Set labels and title
    ax.set_xlabel("Rewrite Steps")
    ax.set_ylabel("Mean L2 Distance")
    ax.set_title("Mean L2 Distance versus Rewrite Steps")
    
    # Add legend
    ax.legend()
    
    # Set axis limits
    ax.set_xlim([1, max_steps])
    ax.set_ylim([0, None])
    
    # Add grid
    ax.grid(True)
    
    return fig

def plot_histogram_of_scores(success_probs, success_labels, step=1):
    """
    Plot histogram of rewrite success prediction scores.
    
    Args:
        success_probs: Array of predicted success probabilities
        success_labels: Array of true success labels
        step: Rewrite step
        
    Returns:
        fig: Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Convert to logits for better visualization
    success_logits = [np.log(p / (1 - p)) if 0 < p < 1 else (-40 if p <= 0 else 40) for p in success_probs]
    
    # Separate positive and negative examples
    pos_logits = [logit for logit, label in zip(success_logits, success_labels) if label == 1]
    neg_logits = [logit for logit, label in zip(success_logits, success_labels) if label == 0]
    
    # Plot histograms
    bins = np.linspace(-40, 40, 41)
    ax.hist(
        pos_logits,
        bins=bins,
        alpha=0.7,
        label="Successful Rewrites",
        color="green",
        density=True
    )
    
    ax.hist(
        neg_logits,
        bins=bins,
        alpha=0.7,
        label="Failed Rewrites",
        color="red",
        density=True
    )
    
    # Set labels and title
    ax.set_xlabel("Score Logit")
    ax.set_ylabel("Ratio of Scores")
    ax.set_title(f"Histogram of Prediction Scores (Step {step})")
    
    # Add legend
    ax.legend()
    
    return fig

def plot_embedding_visualization(embeddings, labels, title="Embedding Visualization"):
    """
    Visualize embeddings using t-SNE.
    
    Args:
        embeddings: List of embedding tensors
        labels: List of labels for coloring
        title: Plot title
        
    Returns:
        fig: Matplotlib figure
    """
    # Convert to numpy array
    embeddings_np = np.stack([e.numpy() for e in embeddings])
    
    # Apply t-SNE
    tsne = TSNE(n_components=2, random_state=42)
    embeddings_2d = tsne.fit_transform(embeddings_np)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Create scatter plot
    scatter = ax.scatter(
        embeddings_2d[:, 0],
        embeddings_2d[:, 1],
        c=labels,
        cmap="viridis",
        alpha=0.7,
        s=50
    )
    
    # Add colorbar
    plt.colorbar(scatter, ax=ax, label="Label")
    
    # Set labels and title
    ax.set_xlabel("t-SNE Dimension 1")
    ax.set_ylabel("t-SNE Dimension 2")
    ax.set_title(title)
    
    return fig