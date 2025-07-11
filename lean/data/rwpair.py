import os 
from lean_dojo import * 
import json 
from datetime import datetime 
import random 
from typing import List, Tuple 
from tqdm import tqdm, trange 
from lean_interaction import run_theorem_with_a_tac, process_result, build_theorem_from_str

class ReWritePair:
    """
    The class for rewriting pairs.
    Args:
        ori_thm (str): The original theorem.
        rw_tac (str): The rewrite tactic.
        success (int): 0 for failure, 1 for success.
        rw_result (str): The result of the rewrite.
    """
    def __init__(self, ori_thm, rw_tac, success=0, rw_result=''):
        self.ori_thm = ori_thm
        self.rw_tac = rw_tac
        self.success = success
        self.rw_result = rw_result
        
    def get_rewrite_pair(self):
        """
        Returns the rewrite pair.
        Dict: {"ori_thm": str, "rw_tac"": str, "success": int (0 for failure, 1 for success), "rw_result": str}
        """
        return {
            "ori_thm": self.ori_thm,
            "rw_tac": self.rw_tac,
            "success": self.success,
            "rw_result": self.rw_result
        }
        
class ReWriteChain:
    '''
    The class for rewriting chains.
    Args:
        ori_thm (str): The original theorem.
        rw_tacs (List[str]): The list of rewrite tactics.
        rw_results (List[str]): The list of results for each rewrite tactic.
    '''
    def __init__(self, ori_thm, rw_tacs, rw_results):
        self.ori_thm = ori_thm
        self.rw_tacs = rw_tacs
        self.rw_results = rw_results
        self.lens = len(rw_tacs)
        
    def get_rewrite_chain(self):
        """
        Returns the rewrite chain.
        Dict: {"ori_thm": str, "rw_tacs": List[str], "rw_results": List[str], "lens": int}
        """
        return {
            "ori_thm": self.ori_thm,
            "rw_tacs": self.rw_tacs,
            "rw_results": self.rw_results,
            "lens": self.lens
        }
        