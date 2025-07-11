"""
Dataset classes for mathematical reasoning in latent space
"""
import os
import json
import pickle
import random
from typing import List, Tuple, Dict, Any, Optional
import torch
from torch.utils.data import Dataset, DataLoader
from torch_geometric.data import Data, Batch
import numpy as np
from .lean_interface import LeanTheorem, RewriteResult, generate_rewrite_pairs
from ..formula_parser import LeanFormulaParser
import sys 
sys.path.append('/home/stu4/formal_reasoning/baseline/Math_Reasoning_in_Latent_Space/')
from baseline.Math_Reasoning_in_Latent_Space.lean.config import config

class RewriteDataset(Dataset):
    """Dataset for rewrite prediction and embedding learning"""
    
    def __init__(self, 
                 theorems: List[LeanTheorem],
                 rewrite_pairs: List[Tuple[str, str, bool, Optional[str]]],
                 parser: LeanFormulaParser,
                 mode: str = "train"):
        """
        Initialize the rewrite dataset
        
        Args:
            theorems: List of theorems
            rewrite_pairs: List of (target, parameter, success, result) tuples
            parser: Formula parser for converting to graphs
            mode: Dataset mode ("train", "val", "test")
        """
        self.theorems = theorems
        self.rewrite_pairs = rewrite_pairs
        self.parser = parser
        self.mode = mode
        
        # Filter successful pairs for embedding learning
        self.successful_pairs = [
            (target, param, result) for target, param, success, result 
            in rewrite_pairs if success and result is not None
        ]
        
        # Create theorem name to index mapping
        self.theorem_to_idx = {
            theorem.name: idx for idx, theorem in enumerate(theorems)
        }
    
    def __len__(self) -> int:
        return len(self.rewrite_pairs)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        Get a single item from the dataset
        
        Args:
            idx: Index of the item
            
        Returns:
            Dictionary containing the data item
        """
        target, parameter, success, result = self.rewrite_pairs[idx]
        
        # Convert formulas to graphs
        target_graph = self.parser.formula_to_graph(target)
        param_graph = self.parser.formula_to_graph(parameter)
        
        item = {
            'target_formula': target,
            'parameter_formula': parameter,
            'target_graph': target_graph,
            'parameter_graph': param_graph,
            'success': torch.tensor(success, dtype=torch.float),
            'index': idx
        }
        
        # Add result graph for successful rewrites
        if success and result is not None:
            result_graph = self.parser.formula_to_graph(result)
            item['result_graph'] = result_graph
            item['result_formula'] = result
        
        return item
    
    def get_negative_samples(self, target_idx: int, num_negatives: int = 15) -> List[int]:
        """
        Get negative samples for a given target theorem
        
        Args:
            target_idx: Index of the target theorem
            num_negatives: Number of negative samples to return
            
        Returns:
            List of indices for negative samples
        """
        # Find all parameters that don't successfully rewrite the target
        negative_indices = []
        target_formula = self.theorems[target_idx].statement
        
        for i, (target, param, success, _) in enumerate(self.rewrite_pairs):
            if target == target_formula and not success:
                negative_indices.append(i)
        
        # Sample random negatives if we don't have enough
        if len(negative_indices) < num_negatives:
            all_indices = list(range(len(self.rewrite_pairs)))
            additional_negatives = random.sample(
                [i for i in all_indices if i not in negative_indices], 
                min(num_negatives - len(negative_indices), len(all_indices) - len(negative_indices))
            )
            negative_indices.extend(additional_negatives)
        
        return random.sample(negative_indices, min(num_negatives, len(negative_indices)))

class MultiStepReasoningDataset(Dataset):
    """Dataset for multi-step reasoning evaluation"""
    
    def __init__(self, 
                 base_theorems: List[LeanTheorem],
                 rewrite_chains: List[List[Tuple[str, str]]],
                 parser: LeanFormulaParser,
                 max_steps: int = 9):
        """
        Initialize the multi-step reasoning dataset
        
        Args:
            base_theorems: Starting theorems
            rewrite_chains: Chains of (formula, parameter) pairs
            parser: Formula parser
            max_steps: Maximum number of reasoning steps
        """
        self.base_theorems = base_theorems
        self.rewrite_chains = rewrite_chains
        self.parser = parser
        self.max_steps = max_steps
        
        # Build chains of different lengths
        self.step_datasets = {}
        for step in range(max_steps + 1):
            self.step_datasets[step] = self._build_step_dataset(step)
    
    def _build_step_dataset(self, num_steps: int) -> List[Dict[str, Any]]:
        """
        Build dataset for a specific number of steps
        
        Args:
            num_steps: Number of rewrite steps
            
        Returns:
            List of data items for this step count
        """
        step_data = []
        
        for chain in self.rewrite_chains:
            if len(chain) > num_steps:
                # Extract the first num_steps from the chain
                sub_chain = chain[:num_steps+1]  # +1 because we need the final state
                
                item = {
                    'initial_formula': sub_chain[0][0],
                    'rewrite_sequence': [param for _, param in sub_chain[:-1]],
                    'final_formula': sub_chain[-1][0],
                    'num_steps': num_steps
                }
                
                step_data.append(item)
        
        return step_data
    
    def get_step_dataset(self, num_steps: int) -> List[Dict[str, Any]]:
        """
        Get dataset for a specific number of steps
        
        Args:
            num_steps: Number of steps
            
        Returns:
            Dataset for the specified number of steps
        """
        return self.step_datasets.get(num_steps, [])
    
    def __len__(self) -> int:
        return sum(len(dataset) for dataset in self.step_datasets.values())

def collate_rewrite_batch(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Custom collate function for rewrite dataset batches
    
    Args:
        batch: List of data items
        
    Returns:
        Batched data
    """
    # Separate the different types of data
    target_graphs = [item['target_graph'] for item in batch]
    param_graphs = [item['parameter_graph'] for item in batch]
    successes = torch.stack([item['success'] for item in batch])
    
    # Batch the graphs
    target_batch = Batch.from_data_list(target_graphs)
    param_batch = Batch.from_data_list(param_graphs)
    
    batched_data = {
        'target_batch': target_batch,
        'parameter_batch': param_batch,
        'success': successes,
        'target_formulas': [item['target_formula'] for item in batch],
        'parameter_formulas': [item['parameter_formula'] for item in batch],
        'indices': torch.tensor([item['index'] for item in batch])
    }
    
    # Add result graphs for successful rewrites
    result_graphs = []
    result_formulas = []
    has_results = []
    
    for item in batch:
        if 'result_graph' in item:
            result_graphs.append(item['result_graph'])
            result_formulas.append(item['result_formula'])
            has_results.append(True)
        else:
            # Create dummy graph for failed rewrites
            dummy_graph = Data(x=torch.zeros((1, 2)), edge_index=torch.empty((2, 0), dtype=torch.long))
            result_graphs.append(dummy_graph)
            result_formulas.append("")
            has_results.append(False)
    
    batched_data['result_batch'] = Batch.from_data_list(result_graphs)
    batched_data['result_formulas'] = result_formulas
    batched_data['has_results'] = torch.tensor(has_results, dtype=torch.bool)
    
    return batched_data

def create_data_loaders(theorems: List[LeanTheorem],
                       rewrite_pairs: List[Tuple[str, str, bool, Optional[str]]],
                       parser: LeanFormulaParser) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create train, validation, and test data loaders
    
    Args:
        theorems: List of all theorems
        rewrite_pairs: List of all rewrite pairs
        parser: Formula parser
        
    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    # Split the data
    random.shuffle(rewrite_pairs)
    
    train_size = int(len(rewrite_pairs) * config.data.train_split)
    val_size = int(len(rewrite_pairs) * config.data.val_split)
    
    train_pairs = rewrite_pairs[:train_size]
    val_pairs = rewrite_pairs[train_size:train_size + val_size]
    test_pairs = rewrite_pairs[train_size + val_size:]
    
    # Create datasets
    train_dataset = RewriteDataset(theorems, train_pairs, parser, mode="train")
    val_dataset = RewriteDataset(theorems, val_pairs, parser, mode="val")
    test_dataset = RewriteDataset(theorems, test_pairs, parser, mode="test")
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.model.batch_size,
        shuffle=True,
        collate_fn=collate_rewrite_batch,
        num_workers=config.num_workers
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.data.eval_batch_size,
        shuffle=False,
        collate_fn=collate_rewrite_batch,
        num_workers=config.num_workers
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.data.eval_batch_size,
        shuffle=False,
        collate_fn=collate_rewrite_batch,
        num_workers=config.num_workers
    )
    
    return train_loader, val_loader, test_loader

def save_processed_data(data: Dict[str, Any], save_path: str):
    """
    Save processed data to disk
    
    Args:
        data: Dictionary containing the processed data
        save_path: Path to save the data
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    with open(save_path, 'wb') as f:
        pickle.dump(data, f)

def load_processed_data(load_path: str) -> Dict[str, Any]:
    """
    Load processed data from disk
    
    Args:
        load_path: Path to load the data from
        
    Returns:
        Dictionary containing the loaded data
    """
    with open(load_path, 'rb') as f:
        return pickle.load(f)