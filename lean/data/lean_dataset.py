import os
import json
import pickle
import random
from typing import List, Tuple, Dict, Any, Optional
import torch
from torch.utils.data import Dataset, DataLoader
from torch_geometric.data import Data, Batch
import numpy as np
from rwpair import ReWritePair, ReWriteChain
from formula_parser import LeanFormulaParser
import sys 
sys.path.append('/home/stu4/formal_reasoning/baseline/Math_Reasoning_in_Latent_Space/')
from baseline.Math_Reasoning_in_Latent_Space.lean.config import config

class ReWritePairDataset(Dataset):
    '''
    Dataset for rewriting pairs.
    '''
    def __init__(self, rw_pairs: List[ReWritePair]):
        self.data = [rw_pair.get_rewrite_pair() for rw_pair in rw_pairs]
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        ori_thm, rw_tac, success, rw_result = self.data[idx].values()
        ori_graph = LeanFormulaParser().formula_to_graph(ori_thm)
        tac_graph = LeanFormulaParser().formula_to_graph(rw_tac)
        
        item = {
            "ori_thm": ori_thm,
            "rw_tac": rw_tac,
            "rw_result": rw_result,
            "success": torch.tensor(success, dtype=torch.float),
            "ori_graph": ori_graph,
            "tac_graph": tac_graph,
        }
        
        if success:
            item["rw_graph"] = LeanFormulaParser().formula_to_graph(rw_result)
        
        return item
    
class ReWriteChainDataset(Dataset):
    '''
    Dataset for rewriting chains.
    '''
    def __init__(self, rw_chains: List[ReWriteChain]):
        self.data = [rw_chain.get_rewrite_chain() for rw_chain in rw_chains]
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        ori_thm, rw_tacs, rw_results = self.data[index].values()
        ori_graph = LeanFormulaParser().formula_to_graph(ori_thm)
        tac_graphs = [LeanFormulaParser().formula_to_graph(tac) for tac in rw_tacs]
        rw_graphs = [LeanFormulaParser().formula_to_graph(result) for result in rw_results]
        
        item = {
            "ori_thm": ori_thm,
            "rw_tacs": rw_tacs,
            "rw_results": rw_results,
            "ori_graph": ori_graph,
            "tac_graphs": tac_graphs,
            "rw_graphs": rw_graphs,
        }
        
        return item 
    
    
    
    
    