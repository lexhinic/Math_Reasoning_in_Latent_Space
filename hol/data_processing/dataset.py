# data_processing/dataset.py
import os
import random
import pickle
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from tqdm import tqdm
from .parser import HOLLightParser
from .rewrite_executor import HOLLightRewriter, SimulatedHOLLightRewriter

class HOLRewriteDataset(Dataset):
    """
    Dataset for training and evaluating the rewrite prediction model.
    Contains pairs of theorems and parameters, with labels indicating
    whether the rewrite was successful.
    """
    
    def __init__(self, config, split="train"):
        """
        Initialize the dataset.
        
        Args:
            config: Configuration object
            split: Dataset split ("train", "val", or "test")
        """
        self.config = config
        self.split = split
        
        # Initialize parser and rewriter
        self.parser = HOLLightParser()
        if config.USE_SIMULATED_HOLLIGHTREWRITER:
            self.rewriter = SimulatedHOLLightRewriter(timeout=config.TIMEOUT)
        else:
            self.rewriter = HOLLightRewriter(config.HOL_LIGHT_PATH, timeout=config.TIMEOUT)
        
        # Load the theorem database
        self.theorems = self._load_theorems()
        
        # Generate or load the rewrite examples
        self.examples = self._generate_examples()
        
        if self.split == "train":
            start_idx = 0
            end_idx = self.config.TRAIN_SPLIT
        elif self.split == "val":
            start_idx = self.config.TRAIN_SPLIT
            end_idx = self.config.TRAIN_SPLIT + self.config.VAL_SPLIT
        else:  # test
            start_idx = self.config.TRAIN_SPLIT + self.config.VAL_SPLIT
            end_idx = self.config.TRAIN_SPLIT + self.config.VAL_SPLIT + self.config.TEST_SPLIT
        self.examples = self.examples[start_idx:end_idx]

#        self.examples = [("(a + 0) + b)", "!(x:num). x + 0 = x", 1, "a + b"), ("(a + b)", "!(x:num). x + 0 = x", 0, None), ("(a * (b * 1))", "!(x:num). x * 1 = x", 1, "a * b"), ("(a * b)", "!(x:num). x * 1 = x", 0, None), ("~~P", "!(p:bool). ~~p = p", 1, "P"), ("~P", "!(p:bool). ~~p = p", 0, None)]

    def _load_theorems(self):
        """Load theorems from the HOList database."""
        theorems = []
        
        with open(os.path.join(self.config.HOLIST_DB_PATH, "theorems.txt"), "r") as f:
            for i, line in enumerate(f):
                # Parse the theorem
                theorem_name, theorem_statement = line.strip().split(":", 1)
                theorems.append((theorem_name, theorem_statement))
        
        return theorems
    
    def _generate_examples(self):
        """
        Generate rewrite examples by trying all possible rewrites.
        Caches results for efficiency.
        """
        # Check if examples are already cached
        cache_path = os.path.join(self.config.HOLIST_DB_PATH, f"{self.split}_examples.pkl")
        txt_cache_path = os.path.join(self.config.HOLIST_DB_PATH, f"{self.split}_examples.txt")
        
        if os.path.exists(cache_path):
            print(f"Loading cached examples from {cache_path}")
            with open(cache_path, "rb") as f:
                return pickle.load(f)
        
        print(f"Generating examples for {self.split} split")
        examples = []
        
        # Generate all pairs (T, P) where P occurs before T in the database
        for i, (t_name, t_statement) in tqdm(enumerate(self.theorems)):
            for j in range(i):
                p_name, p_statement = self.theorems[j]
                
                # Execute the rewrite
                success, result = self.rewriter.execute_rewrite(t_statement, p_statement)
                
                if success:
                    # If the rewrite was successful and changed the theorem
                    examples.append((t_statement, p_statement, 1, result))
                else:
                    # If the rewrite failed
                    examples.append((t_statement, p_statement, 0, None))
        
        # Cache the examples
        with open(cache_path, "wb") as f:
            pickle.dump(examples, f)
            
        # Also save in human-readable txt format
        print(f"Saving human-readable examples to {txt_cache_path}")
        with open(txt_cache_path, "w", encoding="utf-8") as f:
            f.write(f"Generated examples for {self.split} split\n")
            f.write(f"Total examples: {len(examples)}\n")
            f.write("=" * 80 + "\n\n")
            
            # Count successful and failed examples
            successful_count = sum(1 for ex in examples if ex[2] == 1)
            failed_count = len(examples) - successful_count
            
            f.write(f"Statistics:\n")
            f.write(f"Successful rewrites: {successful_count}\n")
            f.write(f"Failed rewrites: {failed_count}\n")
            f.write(f"Success rate: {successful_count / len(examples) * 100:.2f}%\n")
            f.write("=" * 80 + "\n\n")
            
            # Write examples grouped by success/failure
            f.write("SUCCESSFUL REWRITES:\n")
            f.write("-" * 40 + "\n")
            successful_examples = [ex for ex in examples if ex[2] == 1]
            for idx, (t_statement, p_statement, success, result) in enumerate(successful_examples):
                f.write(f"Example {idx + 1}:\n")
                f.write(f"  Theorem (T): {t_statement}\n")
                f.write(f"  Parameter (P): {p_statement}\n")
                f.write(f"  Result (R): {result}\n")
                f.write(f"  Success: {success}\n")
                f.write("\n")
                
            f.write("\n" + "=" * 80 + "\n\n")
            f.write("FAILED REWRITES:\n")
            f.write("-" * 40 + "\n")
            failed_examples = [ex for ex in examples if ex[2] == 0]
            for idx, (t_statement, p_statement, success, result) in enumerate(failed_examples):
                f.write(f"Example {idx + 1}:\n")
                f.write(f"  Theorem (T): {t_statement}\n")
                f.write(f"  Parameter (P): {p_statement}\n")
                f.write(f"  Result: None (failed)\n")
                f.write(f"  Success: {success}\n")
                f.write("\n")
        
        return examples
    
    def __len__(self):
        """Return the number of examples."""
        return len(self.examples)
    
    def __getitem__(self, idx):
        """Get an example by index."""
        t_statement, p_statement, success, result = self.examples[idx]
        
        # Parse the theorems into graph representations
        t_nodes, t_edges, t_types = self.parser.parse_formula(t_statement)
        p_nodes, p_edges, p_types = self.parser.parse_formula(p_statement)
        
        # If the rewrite was successful, also parse the result
        if success:
            r_nodes, r_edges, r_types = self.parser.parse_formula(result)
        else:
            r_nodes, r_edges, r_types = None, None, None
        
        return {
            "t_statement": t_statement,
            "p_statement": p_statement,
            "t_nodes": t_nodes,
            "t_edges": t_edges,
            "t_types": t_types,
            "p_nodes": p_nodes,
            "p_edges": p_edges,
            "p_types": p_types,
            "success": torch.tensor(success, dtype=torch.float32),
            "result": result,
            "r_nodes": r_nodes,
            "r_edges": r_edges,
            "r_types": r_types
        }

def collate_fn(batch):
    """
    Custom collate function for batching examples.
    Ensures each batch has both positive and negative examples.
    """
    # Filter out None values
    batch = [b for b in batch if b is not None]
    if not batch:
        return None
    
    # Separate the examples by rewrite success
    successful = [ex for ex in batch if ex["success"].item() == 1]
    failed = [ex for ex in batch if ex["success"].item() == 0]
    
    # Ensure that we have at least one successful example in the batch
    if not successful:
        return None
    
    # Group successful examples by theorem
    t_successful = {}
    for ex in successful:
        t_key = ex["t_statement"]
        if t_key not in t_successful:
            t_successful[t_key] = []
        t_successful[t_key].append(ex)
    
    # Select one successful example per theorem
    selected = []
    for t_key, examples in t_successful.items():
        selected.append(random.choice(examples))
    
    # Add negative examples (failed rewrites)
    if failed:
        num_neg = min(len(selected) * 15, len(failed))
        selected.extend(random.sample(failed, num_neg))
    
    # Create the batch
    batch_dict = {
        "t_statement": [ex["t_statement"] for ex in selected],
        "p_statement": [ex["p_statement"] for ex in selected],
        "t_nodes": [ex["t_nodes"] for ex in selected],
        "t_edges": [ex["t_edges"] for ex in selected],
        "t_types": [ex["t_types"] for ex in selected],
        "p_nodes": [ex["p_nodes"] for ex in selected],
        "p_edges": [ex["p_edges"] for ex in selected],
        "p_types": [ex["p_types"] for ex in selected],
        "success": torch.stack([ex["success"] for ex in selected]),
        "result": [ex["result"] for ex in selected],
    }
    
    # Only include result data for successful rewrites
    r_nodes = [ex["r_nodes"] for ex in selected if ex["r_nodes"] is not None]
    r_edges = [ex["r_edges"] for ex in selected if ex["r_edges"] is not None]
    r_types = [ex["r_types"] for ex in selected if ex["r_types"] is not None]
    
    if r_nodes:
        batch_dict["r_nodes"] = r_nodes
        batch_dict["r_edges"] = r_edges
        batch_dict["r_types"] = r_types
    
    return batch_dict

def get_dataloader(config, split="train", batch_size=None, shuffle=None):
    """
    Get a dataloader for the specified split.
    
    Args:
        config: Configuration object
        split: Dataset split ("train", "val", or "test")
        batch_size: Batch size (default: config.BATCH_SIZE)
        shuffle: Whether to shuffle the data (default: True for train, False otherwise)
        
    Returns:
        DataLoader object
    """
    dataset = HOLRewriteDataset(config, split)
    
    if batch_size is None:
        batch_size = config.BATCH_SIZE
    
    if shuffle is None:
        shuffle = (split == "train")
    
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collate_fn,
        num_workers=4
    )

def generate_multi_step_datasets(config, max_steps=9):
    """
    Generate datasets for multi-step reasoning evaluation.
    Each dataset contains theorems that are the result of multiple rewrite steps.
    
    Args:
        config: Configuration object
        max_steps: Maximum number of rewrite steps
        
    Returns:
        List of datasets, one for each rewrite step
    """
    # Initialize the rewriter and parser
    if config.USE_SIMULATED_HOLLIGHTREWRITER:
        rewriter = SimulatedHOLLightRewriter(timeout=config.TIMEOUT)
    else:
        rewriter = HOLLightRewriter(config.HOL_LIGHT_PATH, timeout=config.TIMEOUT)
    parser = HOLLightParser()
    
    # Load validation theorems
    val_theorems = []
    
    with open(os.path.join(config.HOLIST_DB_PATH, "theorems.txt"), "r") as f:
        for i, line in enumerate(f):
            if i >= config.TRAIN_SPLIT and i < config.TRAIN_SPLIT + config.VAL_SPLIT:
                theorem_name, theorem_statement = line.strip().split(":", 1)
                val_theorems.append((theorem_name, theorem_statement))
    
    # Generate datasets for each step
    datasets = []
    
    # Initial dataset (step 0)
    step0_data = []
    for t_name, t_statement in val_theorems:
        for p_name, p_statement in val_theorems:
            if t_name != p_name:
                success, result = rewriter.execute_rewrite(t_statement, p_statement)
                step0_data.append((t_statement, p_statement, success, result))
    
    datasets.append(step0_data)
    
    # For each subsequent step
    current_theorems = [t_statement for _, t_statement in val_theorems]
    
    for step in range(1, max_steps + 1):
        print(f"Generating dataset for step {step}")
        next_data = []
        next_theorems = []
        
        # For each theorem in the current step
        for t_statement in tqdm(current_theorems):
            # Try rewriting with each parameter
            for p_name, p_statement in val_theorems:
                success, result = rewriter.execute_rewrite(t_statement, p_statement)
                
                if success:
                    next_data.append((t_statement, p_statement, success, result))
                    if result not in next_theorems:
                        next_theorems.append(result)
                else:
                    next_data.append((t_statement, p_statement, success, None))
        
        datasets.append(next_data)
        
        # Update current theorems for next step
        if next_theorems:
            # Limit the number of theorems to keep the dataset manageable
            if len(next_theorems) > len(val_theorems):
                next_theorems = random.sample(next_theorems, len(val_theorems))
            current_theorems = next_theorems
        else:
            # If no successful rewrites, use the original theorems again
            print(f"No successful rewrites at step {step}, reusing original theorems")
            current_theorems = [t_statement for _, t_statement in val_theorems]
    
    return datasets