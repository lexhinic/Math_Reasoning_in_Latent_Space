import os 
from lean_dojo import * 
import json 
from datetime import datetime 
import random 
from typing import List, Tuple 
from tqdm import tqdm, trange 

def get_tactic_state(state):
    '''
    Extracts the tactic state from a given state object.
    Args:
        state (TacticState): The state object to extract the tactic state from.
    Returns:
        str: The tactic state as a string.
    '''
    if isinstance(state, TacticState):
        ts = state.pp
    else:
        ts = state.unsolved_tactic_state
    return ts

def run_theorem_with_a_tac(theorem, tac):
    '''
    Runs a theorem with a given tactic.
    Args:
        theorem (Theorem): The theorem to run.
        tac (str): The tactic to apply.
    Returns:
        result (TatacticResult: ProofFinished | LeanError | ProofGivenUp | TanticState): The result of running the tactic on the theorem.
    '''
    with Dojo(theorem) as (dojo, init_state):
        result = dojo.run_tac(init_state, tac)
        return result 
    
def process_result(result):
    '''
    Processes the result of running a tactic on a theorem.
    Args:
        result (TatacticResult: ProofFinished | LeanError | ProofGivenUp | TacticState): The result of running a tactic on a theorem.
    Returns:
        str: A string representation of the result.
    '''
    if isinstance(result, ProofFinished):
        return "Proof finished successfully."
    if isinstance(result, LeanError):
        return f"Lean error: {result.error}"
    if isinstance(result, ProofGivenUp):
        return "Proof was given up."
    if isinstance(result, TacticState):
        return result.pp
    
def extract_theorem_from_tacticstate(state):
    '''
    Extracts the theorem from a given tactic state.
    Args:
        state (TacticState): The tactic state to extract the theorem from.
    Returns:
        str: The theorem as a string.
    '''
    if not isinstance(state, TacticState):
        raise ValueError("Expected a TacticState object.")
    
        
        
    
repo = LeanGitRepo(
    "https://github.com/leanprover-community/mathlib4",
    "29dcec074de168ac2bf835a77ef68bbe069194c5",
)
theorem = Theorem(repo, "Lean4Example.lean", "example")
print(run_theorem_with_a_tac(theorem, "rw [add_assoc, add_comm b, ←add_assoc]"))
    




