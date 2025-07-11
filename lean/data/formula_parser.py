import re
import networkx as nx
from typing import Dict, List, Tuple, Any, Optional, Set, Union
from dataclasses import dataclass, field
import torch
from torch_geometric.data import Data
import logging

logger = logging.getLogger(__name__)

@dataclass
class FormulaNode:
    """Represents a node in the formula's abstract syntax tree"""
    node_type: str
    value: str
    children: List['FormulaNode'] = field(default_factory=list)
    node_id: int = -1
    position: Optional[Tuple[int, int]] = None  # (start, end) positions in original text
    attributes: Dict[str, Any] = field(default_factory=dict)
    
    def add_child(self, child: 'FormulaNode'):
        """Add a child node"""
        self.children.append(child)
    
    def is_leaf(self) -> bool:
        """Check if this is a leaf node"""
        return len(self.children) == 0

class LeanFormulaParser:
    """Enhanced parser for converting Lean formulas to graph representations"""
    
    def __init__(self, max_nodes: int = 1000):
        """Initialize the parser with Lean-specific syntax rules"""
        self.node_type_vocab = {}
        self.value_vocab = {}
        self.next_node_id = 0
        self.max_nodes = max_nodes
        
        # Enhanced Lean 4 syntax patterns with priorities
        self.lean_patterns = {
            # Quantifiers (highest priority)
            r'∀': ('FORALL', 10),
            r'∃': ('EXISTS', 10),
            r'∃!': ('EXISTS_UNIQUE', 10),
            
            # Logical operators
            r'→': ('IMPLIES', 6),
            r'↔': ('IFF', 7),
            r'∧': ('AND', 4),
            r'∨': ('OR', 5),
            r'¬': ('NOT', 8),
            
            # Equality and comparisons
            r'=': ('EQ', 3),
            r'≠': ('NE', 3),
            r'<': ('LT', 3),
            r'≤': ('LE', 3),
            r'>': ('GT', 3),
            r'≥': ('GE', 3),
            r'∣': ('DIVIDES', 3),
            r'∈': ('MEMBER', 3),
            r'∉': ('NOT_MEMBER', 3),
            r'⊆': ('SUBSET', 3),
            r'⊂': ('PROPER_SUBSET', 3),
            
            # Arithmetic operators
            r'\+': ('PLUS', 2),
            r'-': ('MINUS', 2),
            r'\*': ('MULT', 1),
            r'/': ('DIV', 1),
            r'\^': ('POW', 0),
            r'%': ('MOD', 1),
            
            # Function arrows and composition
            r'∘': ('COMPOSE', 1),
            r'⁻¹': ('INVERSE', 0),
            
            # Set operations
            r'∪': ('UNION', 2),
            r'∩': ('INTER', 2),
            r'\\': ('DIFF', 2),
            r'△': ('SYMDIFF', 2),
            
            # Special symbols
            r'⊤': ('TOP', 9),
            r'⊥': ('BOT', 9),
            r'∅': ('EMPTY', 9),
            r'ℕ': ('NAT', 9),
            r'ℤ': ('INT', 9),
            r'ℚ': ('RAT', 9),
            r'ℝ': ('REAL', 9),
            r'ℂ': ('COMPLEX', 9),
        }
        
        # Keywords and special constructs
        self.lean_keywords = {
            'if', 'then', 'else', 'match', 'with', 'fun', 'λ', 'let', 'in',
            'have', 'show', 'by', 'calc', 'sorry', 'admit', 'trivial',
            'True', 'False', 'Prop', 'Type', 'Sort'
        }
        
        # Initialize built-in vocabularies
        self._init_vocabularies()
    
    def _init_vocabularies(self):
        """Initialize vocabularies with common Lean constructs"""
        # Node types
        base_types = [
            'FORALL', 'EXISTS', 'EXISTS_UNIQUE', 'IMPLIES', 'IFF', 'AND', 'OR', 'NOT',
            'EQ', 'NE', 'LT', 'LE', 'GT', 'GE', 'PLUS', 'MINUS', 'MULT', 'DIV', 'POW',
            'VARIABLE', 'CONSTANT', 'APPLICATION', 'LAMBDA', 'LET', 'IF_THEN_ELSE',
            'MATCH', 'OPERATOR', 'KEYWORD', 'TYPE', 'PROP', 'SORT', 'EMPTY'
        ]
        
        for i, node_type in enumerate(base_types):
            self.node_type_vocab[node_type] = i
        
        # Common values
        base_values = [
            '', '0', '1', '2', 'x', 'y', 'z', 'n', 'm', 'f', 'g', 'h',
            'True', 'False', 'Prop', 'Type', 'ℕ', 'ℤ', 'ℚ', 'ℝ', 'sorry'
        ]
        
        for i, value in enumerate(base_values):
            self.value_vocab[value] = i
    
    def parse_formula(self, formula: str) -> FormulaNode:
        """
        Parse a Lean formula string into an AST
        
        Args:
            formula: The formula string to parse
            
        Returns:
            Root node of the parsed AST
        """
        try:
            # Reset node counter for each formula
            self.next_node_id = 0
            
            # Preprocess the formula
            processed_formula = self._preprocess_formula(formula)
            
            if not processed_formula.strip():
                return self._create_node("EMPTY", "", position=(0, 0))
            
            # Tokenize
            tokens = self._tokenize(processed_formula)
            
            if not tokens:
                return self._create_node("EMPTY", "", position=(0, 0))
            
            # Parse using recursive descent
            root, _ = self._parse_expression(tokens, 0)
            
            return root
            
        except Exception as e:
            logger.warning(f"Error parsing formula '{formula}': {e}")
            # Return a simple error node
            return self._create_node("ERROR", formula[:100], position=(0, len(formula)))
    
    def _preprocess_formula(self, formula: str) -> str:
        """
        Preprocess the formula to normalize syntax
        
        Args:
            formula: Raw formula string
            
        Returns:
            Preprocessed formula string
        """
        # Remove extra whitespace and normalize
        formula = re.sub(r'\s+', ' ', formula.strip())
        
        # Handle multi-character operators first
        formula = re.sub(r'∃!', ' EXISTS_UNIQUE ', formula)
        
        # Replace unicode symbols with tokens
        for pattern, (replacement, _) in self.lean_patterns.items():
            formula = re.sub(pattern, f' {replacement} ', formula)
        
        # Handle lambda expressions
        formula = re.sub(r'λ', ' LAMBDA ', formula)
        formula = re.sub(r'fun ', ' FUN ', formula)
        
        # Normalize arrows and implications
        formula = re.sub(r'->', ' IMPLIES ', formula)
        formula = re.sub(r'<->', ' IFF ', formula)
        
        # Clean up multiple spaces
        formula = re.sub(r'\s+', ' ', formula)
        
        return formula.strip()
    
    def _tokenize(self, formula: str) -> List[Tuple[str, int, int]]:
        """
        Tokenize the preprocessed formula with position information
        
        Args:
            formula: Preprocessed formula string
            
        Returns:
            List of (token, start_pos, end_pos) tuples
        """
        tokens = []
        current_token = ""
        start_pos = 0
        
        i = 0
        while i < len(formula):
            char = formula[i]
            
            if char.isspace():
                if current_token:
                    tokens.append((current_token, start_pos, i))
                    current_token = ""
                # Skip whitespace
                while i < len(formula) and formula[i].isspace():
                    i += 1
                start_pos = i
                continue
            elif char in "()[]{},:":
                if current_token:
                    tokens.append((current_token, start_pos, i))
                    current_token = ""
                tokens.append((char, i, i + 1))
                i += 1
                start_pos = i
            else:
                if not current_token:
                    start_pos = i
                current_token += char
                i += 1
        
        if current_token:
            tokens.append((current_token, start_pos, len(formula)))
        
        return [(token, start, end) for token, start, end in tokens if token.strip()]
    
    def _parse_expression(self, tokens: List[Tuple[str, int, int]], 
                         start_idx: int) -> Tuple[FormulaNode, int]:
        """
        Parse a list of tokens into an expression tree
        
        Args:
            tokens: List of (token, start_pos, end_pos) tuples
            start_idx: Starting index in the token list
            
        Returns:
            Tuple of (root_node, next_index)
        """
        if start_idx >= len(tokens):
            return self._create_node("EMPTY", ""), start_idx
        
        # Check for node limit
        if self.next_node_id >= self.max_nodes:
            logger.warning(f"Reached maximum node limit ({self.max_nodes})")
            return self._create_node("TRUNCATED", "..."), start_idx
        
        token, start_pos, end_pos = tokens[start_idx]
        
        # Handle parentheses
        if token == '(':
            return self._parse_parenthesized(tokens, start_idx)
        
        # Handle quantifiers
        if token in ['FORALL', 'EXISTS', 'EXISTS_UNIQUE']:
            return self._parse_quantifier(tokens, start_idx)
        
        # Handle lambda expressions
        if token in ['LAMBDA', 'FUN']:
            return self._parse_lambda(tokens, start_idx)
        
        # Handle let expressions
        if token == 'let':
            return self._parse_let(tokens, start_idx)
        
        # Handle if-then-else
        if token == 'if':
            return self._parse_if_then_else(tokens, start_idx)
        
        # Parse as binary operation or application
        return self._parse_binary_or_application(tokens, start_idx)
    
    def _parse_parenthesized(self, tokens: List[Tuple[str, int, int]], 
                           start_idx: int) -> Tuple[FormulaNode, int]:
        """Parse parenthesized expressions"""
        level = 0
        end_idx = start_idx
        
        for i in range(start_idx, len(tokens)):
            token, _, _ = tokens[i]
            if token == '(':
                level += 1
            elif token == ')':
                level -= 1
                if level == 0:
                    end_idx = i
                    break
        
        if level != 0:
            # Unmatched parentheses, treat as error
            token, start_pos, end_pos = tokens[start_idx]
            return self._create_node("ERROR", token, position=(start_pos, end_pos)), start_idx + 1
        
        # Parse the inner expression
        inner_tokens = tokens[start_idx + 1:end_idx]
        if inner_tokens:
            inner_expr, _ = self._parse_expression(inner_tokens, 0)
        else:
            inner_expr = self._create_node("EMPTY", "")
        
        return inner_expr, end_idx + 1
    
    def _parse_quantifier(self, tokens: List[Tuple[str, int, int]], 
                         start_idx: int) -> Tuple[FormulaNode, int]:
        """Parse quantifier expressions"""
        token, start_pos, end_pos = tokens[start_idx]
        quant_node = self._create_node("QUANTIFIER", token, position=(start_pos, end_pos))
        
        idx = start_idx + 1
        
        # Parse bound variables
        while idx < len(tokens) and tokens[idx][0] not in [',', ':', '→', 'IMPLIES']:
            var_token, var_start, var_end = tokens[idx]
            var_node = self._create_node("VARIABLE", var_token, position=(var_start, var_end))
            quant_node.add_child(var_node)
            idx += 1
        
        # Skip colon or comma
        if idx < len(tokens) and tokens[idx][0] in [':', ',']:
            idx += 1
        
        # Parse the body
        if idx < len(tokens):
            body, next_idx = self._parse_expression(tokens, idx)
            quant_node.add_child(body)
            return quant_node, next_idx
        
        return quant_node, idx
    
    def _parse_lambda(self, tokens: List[Tuple[str, int, int]], 
                     start_idx: int) -> Tuple[FormulaNode, int]:
        """Parse lambda expressions"""
        token, start_pos, end_pos = tokens[start_idx]
        lambda_node = self._create_node("LAMBDA", token, position=(start_pos, end_pos))
        
        idx = start_idx + 1
        
        # Parse parameters
        while idx < len(tokens) and tokens[idx][0] not in ['=>', '↦', ',']:
            param_token, param_start, param_end = tokens[idx]
            param_node = self._create_node("PARAMETER", param_token, position=(param_start, param_end))
            lambda_node.add_child(param_node)
            idx += 1
        
        # Skip arrow
        if idx < len(tokens) and tokens[idx][0] in ['=>', '↦']:
            idx += 1
        
        # Parse body
        if idx < len(tokens):
            body, next_idx = self._parse_expression(tokens, idx)
            lambda_node.add_child(body)
            return lambda_node, next_idx
        
        return lambda_node, idx
    
    def _parse_let(self, tokens: List[Tuple[str, int, int]], 
                  start_idx: int) -> Tuple[FormulaNode, int]:
        """Parse let expressions"""
        token, start_pos, end_pos = tokens[start_idx]
        let_node = self._create_node("LET", token, position=(start_pos, end_pos))
        
        # Simplified let parsing - just collect tokens until 'in'
        idx = start_idx + 1
        while idx < len(tokens) and tokens[idx][0] != 'in':
            token_val, token_start, token_end = tokens[idx]
            child_node = self._create_node("LET_BINDING", token_val, position=(token_start, token_end))
            let_node.add_child(child_node)
            idx += 1
        
        # Skip 'in'
        if idx < len(tokens):
            idx += 1
        
        # Parse body
        if idx < len(tokens):
            body, next_idx = self._parse_expression(tokens, idx)
            let_node.add_child(body)
            return let_node, next_idx
        
        return let_node, idx
    
    def _parse_if_then_else(self, tokens: List[Tuple[str, int, int]], 
                           start_idx: int) -> Tuple[FormulaNode, int]:
        """Parse if-then-else expressions"""
        token, start_pos, end_pos = tokens[start_idx]
        if_node = self._create_node("IF_THEN_ELSE", token, position=(start_pos, end_pos))
        
        # This is a simplified implementation
        # In practice, you'd need more sophisticated parsing
        idx = start_idx + 1
        while idx < len(tokens) and idx < start_idx + 10:  # Limit to prevent runaway
            token_val, token_start, token_end = tokens[idx]
            child_node = self._create_node("IF_COMPONENT", token_val, position=(token_start, token_end))
            if_node.add_child(child_node)
            idx += 1
        
        return if_node, idx
    
    def _parse_binary_or_application(self, tokens: List[Tuple[str, int, int]], 
                                   start_idx: int) -> Tuple[FormulaNode, int]:
        """Parse binary operations or function applications"""
        if start_idx >= len(tokens):
            return self._create_node("EMPTY", ""), start_idx
        
        # Find operator with lowest precedence
        min_prec = float('inf')
        op_idx = -1
        
        for i in range(start_idx, min(start_idx + 20, len(tokens))):  # Limit lookahead
            token_val, _, _ = tokens[i]
            
            # Check if it's an operator
            for pattern, (op_name, precedence) in self.lean_patterns.items():
                if token_val == op_name and precedence <= min_prec:
                    min_prec = precedence
                    op_idx = i
                    break
        
        if op_idx == -1 or op_idx == start_idx:
            # No binary operator found, parse as application or single term
            return self._parse_application_or_term(tokens, start_idx)
        
        # Parse binary operation
        left_tokens = tokens[start_idx:op_idx]
        op_token, op_start, op_end = tokens[op_idx]
        right_tokens = tokens[op_idx + 1:]
        
        op_node = self._create_node("OPERATOR", op_token, position=(op_start, op_end))
        
        # Parse left side
        if left_tokens:
            left_child, _ = self._parse_expression(left_tokens, 0)
            op_node.add_child(left_child)
        
        # Parse right side
        if right_tokens:
            right_child, next_idx = self._parse_expression(right_tokens, 0)
            op_node.add_child(right_child)
            return op_node, op_idx + 1 + next_idx
        
        return op_node, op_idx + 1
    
    def _parse_application_or_term(self, tokens: List[Tuple[str, int, int]], 
                                  start_idx: int) -> Tuple[FormulaNode, int]:
        """Parse function application or single term"""
        if start_idx >= len(tokens):
            return self._create_node("EMPTY", ""), start_idx
        
        # Take only the first few tokens to avoid infinite applications
        end_idx = min(start_idx + 5, len(tokens))
        current_tokens = tokens[start_idx:end_idx]
        
        if len(current_tokens) == 1:
            # Single term
            token_val, token_start, token_end = current_tokens[0]
            node_type = self._classify_token(token_val)
            return self._create_node(node_type, token_val, position=(token_start, token_end)), start_idx + 1
        
        # Multiple terms - treat as application
        app_node = self._create_node("APPLICATION", "app")
        
        for token_val, token_start, token_end in current_tokens:
            node_type = self._classify_token(token_val)
            child_node = self._create_node(node_type, token_val, position=(token_start, token_end))
            app_node.add_child(child_node)
        
        return app_node, end_idx
    
    def _classify_token(self, token: str) -> str:
        """Classify a token into a node type"""
        if token in self.lean_keywords:
            return "KEYWORD"
        elif token.isdigit():
            return "CONSTANT"
        elif token in [op_name for _, (op_name, _) in self.lean_patterns.items()]:
            return "OPERATOR"
        elif token.isupper() and len(token) > 1:
            return "TYPE"
        else:
            return "VARIABLE"
    
    def _create_node(self, node_type: str, value: str, 
                    position: Optional[Tuple[int, int]] = None) -> FormulaNode:
        """
        Create a new formula node with unique ID
        
        Args:
            node_type: Type of the node
            value: Value of the node
            position: Position in original text
            
        Returns:
            New FormulaNode instance
        """
        if self.next_node_id >= self.max_nodes:
            logger.warning(f"Reached maximum node limit ({self.max_nodes})")
            node_type = "TRUNCATED"
            value = "..."
        
        node = FormulaNode(
            node_type=node_type,
            value=value,
            children=[],
            node_id=self.next_node_id,
            position=position
        )
        self.next_node_id += 1
        
        # Update vocabularies
        if node_type not in self.node_type_vocab:
            self.node_type_vocab[node_type] = len(self.node_type_vocab)
        if value not in self.value_vocab:
            self.value_vocab[value] = len(self.value_vocab)
        
        return node
    
    def formula_to_graph(self, formula: str) -> Data:
        """
        Convert a formula string to a PyTorch Geometric graph
        
        Args:
            formula: The formula string
            
        Returns:
            PyTorch Geometric Data object
        """
        try:
            # Parse the formula
            root = self.parse_formula(formula)
            
            # Extract nodes and edges
            nodes = []
            edges = []
            
            def traverse(node, parent_id=None):
                nodes.append({
                    'id': node.node_id,
                    'type': node.node_type,
                    'value': node.value,
                    'position': node.position
                })
                
                if parent_id is not None:
                    edges.append([parent_id, node.node_id])
                    edges.append([node.node_id, parent_id])  # Undirected
                
                for child in node.children:
                    traverse(child, node.node_id)
            
            traverse(root)
            
            # Handle empty graph
            if not nodes:
                nodes = [{'id': 0, 'type': 'EMPTY', 'value': '', 'position': None}]
            
            # Create node features
            node_features = []
            for node in nodes:
                type_id = self.node_type_vocab.get(node['type'], 0)
                value_id = self.value_vocab.get(node['value'], 0)
                
                # Enhanced feature encoding
                features = [
                    type_id,
                    value_id,
                    len(node['value']),  # Value length
                    1.0 if node['value'].isdigit() else 0.0,  # Is numeric
                    1.0 if node['value'] in self.lean_keywords else 0.0,  # Is keyword
                ]
                node_features.append(features)
            
            # Convert to tensors
            x = torch.tensor(node_features, dtype=torch.float)
            
            if edges:
                edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
            else:
                edge_index = torch.empty((2, 0), dtype=torch.long)
            
            # Add graph-level features
            num_nodes = len(nodes)
            num_edges = len(edges) // 2 if edges else 0
            
            return Data(
                x=x, 
                edge_index=edge_index,
                num_nodes=num_nodes,
                num_edges=num_edges,
                formula=formula
            )
            
        except Exception as e:
            logger.error(f"Error converting formula to graph: {e}")
            # Return minimal valid graph
            x = torch.tensor([[0, 0, 0, 0, 0]], dtype=torch.float)
            edge_index = torch.empty((2, 0), dtype=torch.long)
            return Data(x=x, edge_index=edge_index, num_nodes=1, num_edges=0, formula=formula)
    
    def get_vocab_sizes(self) -> Tuple[int, int]:
        """
        Get the sizes of the vocabularies
        
        Returns:
            Tuple of (node_type_vocab_size, value_vocab_size)
        """
        return len(self.node_type_vocab), len(self.value_vocab)
    
    def get_vocabularies(self) -> Tuple[Dict[str, int], Dict[str, int]]:
        """
        Get the vocabularies
        
        Returns:
            Tuple of (node_type_vocab, value_vocab)
        """
        return self.node_type_vocab.copy(), self.value_vocab.copy()
    
    def reset_vocabularies(self):
        """Reset vocabularies to initial state"""
        self.node_type_vocab = {}
        self.value_vocab = {}
        self._init_vocabularies()