"""
Interface for interacting with Lean theorems and tactics
Adapted for LeanDojo from the original HOL Light implementation
"""
import json
import time
import logging
from typing import List, Dict, Tuple, Optional, Any, Union
from dataclasses import dataclass
from pathlib import Path
import traceback

from lean_dojo import LeanGitRepo, Theorem, Pos, TacticState, LeanFile
from lean_dojo.interaction import Dojo, DojoInitRequest, ProofFinished, DojoHardTimeoutError, DojoCrashError

logger = logging.getLogger(__name__)

@dataclass
class RewriteResult:
    """Result of a rewrite operation"""
    success: bool
    original_formula: str
    parameter: str
    result_formula: Optional[str] = None
    execution_time: float = 0.0
    error_message: Optional[str] = None
    intermediate_states: List[str] = None

    def __post_init__(self):
        if self.intermediate_states is None:
            self.intermediate_states = []

class LeanRewriteEngine:
    """Engine for performing rewrites in Lean using LeanDojo"""
    
    def __init__(self, repo_path: str, max_time: float = 10.0, timeout: int = 300):
        """
        Initialize the Lean rewrite engine
        
        Args:
            repo_path: Path to the Lean repository
            max_time: Maximum time allowed for each rewrite operation
            timeout: Timeout for Dojo operations in seconds
        """
        self.repo_path = Path(repo_path)
        self.max_time = max_time
        self.timeout = timeout
        self.repo = None
        self.dojo = None
        
        try:
            self.repo = LeanGitRepo(self.repo_path, commit=None)
        except Exception as e:
            logger.error(f"Failed to initialize LeanGitRepo: {e}")
            raise
        
    def __enter__(self):
        """Context manager entry"""
        try:
            self.dojo = Dojo(self.repo, timeout=self.timeout)
            return self
        except Exception as e:
            logger.error(f"Failed to initialize Dojo: {e}")
            raise
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        if self.dojo:
            try:
                self.dojo.close()
            except Exception as e:
                logger.warning(f"Error closing Dojo: {e}")
    
    def rewrite_with_theorem(self, target_theorem: Union[str, Theorem], 
                           rewrite_theorem: str, 
                           context_theorems: List[str] = None) -> RewriteResult:
        """
        Attempt to rewrite target using rewrite_theorem
        
        Args:
            target_theorem: The theorem to be rewritten (name or Theorem object)
            rewrite_theorem: The theorem name to use for rewriting
            context_theorems: Additional theorem names for context
            
        Returns:
            RewriteResult containing the outcome
        """
        start_time = time.time()
        
        try:
            # Get the target theorem object
            if isinstance(target_theorem, str):
                theorem = self._find_theorem_by_name(target_theorem)
                if theorem is None:
                    return RewriteResult(
                        success=False,
                        original_formula=target_theorem,
                        parameter=rewrite_theorem,
                        error_message=f"Theorem '{target_theorem}' not found",
                        execution_time=time.time() - start_time
                    )
            else:
                theorem = target_theorem
            
            # Initialize the proof state
            req = DojoInitRequest(theorem, timeout=self.timeout)
            result = self.dojo.init(req)
            
            if isinstance(result, ProofFinished):
                return RewriteResult(
                    success=False,
                    original_formula=str(theorem.statement),
                    parameter=rewrite_theorem,
                    error_message="Theorem is already proven",
                    execution_time=time.time() - start_time
                )
            
            # Apply the rewrite tactic
            rewrite_result = self._execute_rewrite(result, rewrite_theorem, context_theorems)
            rewrite_result.execution_time = time.time() - start_time
            rewrite_result.original_formula = str(theorem.statement)
            rewrite_result.parameter = rewrite_theorem
            
            return rewrite_result
                
        except (DojoHardTimeoutError, DojoCrashError) as e:
            execution_time = time.time() - start_time
            logger.error(f"Dojo error in rewrite operation: {e}")
            
            return RewriteResult(
                success=False,
                original_formula=str(target_theorem) if isinstance(target_theorem, Theorem) else target_theorem,
                parameter=rewrite_theorem,
                execution_time=execution_time,
                error_message=f"Dojo error: {str(e)}"
            )
        
        except Exception as e:
            execution_time = time.time() - start_time
            logger.error(f"Error in rewrite operation: {e}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            
            return RewriteResult(
                success=False,
                original_formula=str(target_theorem) if isinstance(target_theorem, Theorem) else target_theorem,
                parameter=rewrite_theorem,
                execution_time=execution_time,
                error_message=str(e)
            )
    
    def _find_theorem_by_name(self, theorem_name: str) -> Optional[Theorem]:
        """
        Find a theorem by name in the repository
        
        Args:
            theorem_name: Name of the theorem to find
            
        Returns:
            Theorem object if found, None otherwise
        """
        try:
            # Get all theorems from the repository
            theorems = self.repo.get_theorems()
            
            for theorem in theorems:
                if theorem.full_name == theorem_name or theorem.name == theorem_name:
                    return theorem
            
            # If not found by exact match, try partial match
            for theorem in theorems:
                if theorem_name in theorem.full_name:
                    return theorem
                    
            return None
            
        except Exception as e:
            logger.error(f"Error finding theorem '{theorem_name}': {e}")
            return None
    
    def _execute_rewrite(self, state: TacticState, 
                        rewrite_theorem: str, 
                        context_theorems: List[str] = None) -> RewriteResult:
        """
        Execute the actual rewrite operation in Lean
        
        Args:
            state: The initial tactic state
            rewrite_theorem: The theorem to use for rewriting
            context_theorems: Additional theorems for context
            
        Returns:
            RewriteResult with the outcome
        """
        try:
            intermediate_states = [str(state)]
            
            # Prepare the rewrite tactic
            tactic = f"rw [{rewrite_theorem}]"
            
            # Try various rewrite strategies
            strategies = [
                f"rw [{rewrite_theorem}]",
                f"rw [← {rewrite_theorem}]",  # Reverse rewrite
                f"simp only [{rewrite_theorem}]",
                f"rw [{rewrite_theorem}] <;> simp",
            ]
            
            # Add context theorems if provided
            if context_theorems:
                for ctx_thm in context_theorems:
                    strategies.append(f"rw [{ctx_thm}, {rewrite_theorem}]")
                    strategies.append(f"rw [{rewrite_theorem}, {ctx_thm}]")
            
            for strategy in strategies:
                try:
                    result = self.dojo.run_tac(state, strategy, timeout=self.max_time)
                    
                    if isinstance(result, ProofFinished):
                        return RewriteResult(
                            success=True,
                            result_formula="⊢ (proof completed)",
                            intermediate_states=intermediate_states + [str(result)]
                        )
                    
                    elif hasattr(result, 'state') and result.state is not None:
                        new_state_str = str(result.state)
                        intermediate_states.append(new_state_str)
                        
                        # Check if the state actually changed
                        if new_state_str != str(state):
                            new_formula = self._extract_formula_from_state(result.state)
                            return RewriteResult(
                                success=True,
                                result_formula=new_formula,
                                intermediate_states=intermediate_states
                            )
                    
                except Exception as tactic_error:
                    logger.debug(f"Tactic '{strategy}' failed: {tactic_error}")
                    continue
            
            # All strategies failed
            return RewriteResult(
                success=False,
                error_message="All rewrite strategies failed",
                intermediate_states=intermediate_states
            )
            
        except Exception as e:
            logger.error(f"Error executing rewrite: {e}")
            return RewriteResult(
                success=False,
                error_message=str(e)
            )
    
    def _extract_formula_from_state(self, state: TacticState) -> str:
        """
        Extract the formula from a Lean tactic state
        
        Args:
            state: The tactic state
            
        Returns:
            String representation of the formula
        """
        try:
            if hasattr(state, 'goals') and state.goals:
                # Get the first goal's target
                first_goal = state.goals[0]
                if hasattr(first_goal, 'target'):
                    return str(first_goal.target)
                elif hasattr(first_goal, 'type'):
                    return str(first_goal.type)
            
            # Fallback to string representation
            return str(state)
            
        except Exception as e:
            logger.warning(f"Error extracting formula from state: {e}")
            return str(state)

class LeanTheorem:
    """Wrapper for Lean theorems with enhanced functionality"""
    
    def __init__(self, name: str, statement: str, proof: str = ""):
        """
        Initialize a Lean theorem
        
        Args:
            name: Name of the theorem
            statement: The theorem statement
            proof: The proof (if available)
        """
        self.name = name
        self.statement = statement
        self.proof = proof
    
    def __str__(self) -> str:
        return f"LeanTheorem({self.name}: {self.statement})"
    
    def __repr__(self) -> str:
        return self.__str__()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation"""
        return {
            "name": self.name,
            "statement": self.statement,
            "proof": self.proof,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'LeanTheorem':
        """Create from dictionary representation"""
        return cls(
            name=data.get("name", ""),
            statement=data.get("statement", ""),
            proof=data.get("proof", ""),
        )
    
    @classmethod
    def from_leandojo_theorem(cls, theorem: Theorem) -> 'LeanTheorem':
        """Create from LeanDojo Theorem object"""
        return cls(
            name=theorem.name,
            statement=str(theorem.statement),
            proof=str(theorem.proof) if hasattr(theorem, 'proof') and theorem.proof else "",
        )

def load_leandojo_theorems(repo_path: str, max_theorems: int = None) -> List[LeanTheorem]:
    """
    Load theorems from LeanDojo repository
    
    Args:
        repo_path: Path to the Lean repository
        max_theorems: Maximum number of theorems to load (None for all)
        
    Returns:
        List of LeanTheorem objects
    """
    theorems = []
    
    try:
        repo = LeanGitRepo(repo_path)
        lean_theorems = repo.get_theorems()
        
        count = 0
        for theorem in lean_theorems:
            if max_theorems and count >= max_theorems:
                break
                
            try:
                lean_theorem = LeanTheorem.from_leandojo_theorem(theorem)
                theorems.append(lean_theorem)
                count += 1
                
            except Exception as e:
                logger.warning(f"Error processing theorem {theorem.name}: {e}")
                continue
                
        logger.info(f"Loaded {len(theorems)} theorems from {repo_path}")
        
    except Exception as e:
        logger.error(f"Error loading theorems from {repo_path}: {e}")
        logger.error(f"Traceback: {traceback.format_exc()}")
    
    return theorems

def generate_rewrite_pairs(theorems: List[LeanTheorem], 
                          repo_path: str,
                          max_pairs: int = 1000,
                          timeout: int = 300) -> List[Tuple[str, str, bool, Optional[str]]]:
    """
    Generate all possible rewrite pairs from a list of theorems
    
    Args:
        theorems: List of theorems to generate pairs from
        repo_path: Path to the Lean repository
        max_pairs: Maximum number of pairs to generate
        timeout: Timeout for each rewrite operation
        
    Returns:
        List of tuples (target, parameter, success, result)
    """
    pairs = []
    
    try:
        with LeanRewriteEngine(repo_path, timeout=timeout) as rewrite_engine:
            pair_count = 0
            
            for i, target_theorem in enumerate(theorems):
                if pair_count >= max_pairs:
                    break
                    
                for j, param_theorem in enumerate(theorems):
                    if pair_count >= max_pairs:
                        break
                        
                    # Only consider parameters that come before the target
                    if j < i:
                        try:
                            result = rewrite_engine.rewrite_with_theorem(
                                target_theorem.name, 
                                param_theorem.name
                            )
                            
                            pairs.append((
                                target_theorem.statement,
                                param_theorem.statement,
                                result.success,
                                result.result_formula
                            ))
                            
                            pair_count += 1
                            
                            if pair_count % 100 == 0:
                                logger.info(f"Generated {pair_count} rewrite pairs")
                                
                        except Exception as e:
                            logger.warning(f"Error generating pair for {target_theorem.name}, {param_theorem.name}: {e}")
                            continue
    
    except Exception as e:
        logger.error(f"Error in generate_rewrite_pairs: {e}")
        logger.error(f"Traceback: {traceback.format_exc()}")
    
    logger.info(f"Generated {len(pairs)} total rewrite pairs")
    return pairs