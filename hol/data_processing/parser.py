# data_processing/parser.py
import re
import torch
from collections import defaultdict
import numpy as np

class HOLLightParser:
    """
    Parser for HOL Light formulas.
    Converts HOL Light syntax into a graph representation for GNN processing.
    """
    
    def __init__(self):
        # HOL Light term types
        self.term_types = {
            "VAR": 0,      # Variables like x, y, z
            "CONST": 1,    # Constants like 0, 1, +
            "COMB": 2,     # Function application
            "ABS": 3       # Lambda abstraction
        }
        
        # HOL Light type types
        self.type_types = {
            "TYVAR": 4,    # Type variables like 'a, 'b
            "TYCONST": 5,  # Type constants like num, real
            "TYAPP": 6     # Type application
        }
        
        # Commonly used operators in HOL Light
        self.operators = {
            "=": 0,
            "+": 1,
            "-": 2,
            "*": 3,
            "/": 4,
            "∧": 5,  # and
            "∨": 6,  # or
            "¬": 7,  # not
            "∀": 8,  # for all
            "∃": 9,  # exists
            "⇒": 10, # implies
            "⇔": 11  # if and only if
        }
        
        # Vocabulary for tokenization
        self.vocab = {}  # Will be built during parsing
        self.vocab_size = 0
        
    def _tokenize(self, formula_str):
        """Tokenize a HOL Light formula string."""
        # Replace special operators with spaces around them for easier tokenization
        for op in self.operators:
            formula_str = formula_str.replace(op, f" {op} ")
        
        # Basic tokenization by whitespace
        tokens = formula_str.split()
        
        # Update vocabulary
        for token in tokens:
            if token not in self.vocab:
                self.vocab[token] = self.vocab_size
                self.vocab_size += 1
        
        return tokens
    
    def _build_ast(self, tokens):
        """
        Build an abstract syntax tree from tokens.
        This is a simplified AST builder that handles basic HOL Light syntax.
        """
        # Stack-based parsing (simplified)
        stack = []
        ast = {"type": "ROOT", "children": []}
        
        for token in tokens:
            if token in self.operators:
                # Operator
                op_node = {"type": "CONST", "value": token, "children": []}
                
                # For binary operators, pop two operands
                if token in ["=", "+", "-", "*", "/", "∧", "∨", "⇒", "⇔"]:
                    if len(stack) >= 2:
                        right = stack.pop()
                        left = stack.pop()
                        op_node["children"] = [left, right]
                # For unary operators, pop one operand
                elif token in ["¬", "∀", "∃"]:
                    if stack:
                        operand = stack.pop()
                        op_node["children"] = [operand]
                
                stack.append(op_node)
            else:
                # Variable or constant
                if token[0].isalpha() and token[0].islower():
                    # Likely a variable
                    node = {"type": "VAR", "value": token, "children": []}
                else:
                    # Likely a constant
                    node = {"type": "CONST", "value": token, "children": []}
                
                stack.append(node)
        
        # The final AST is what's left on the stack
        if stack:
            ast["children"] = stack
        
        return ast
    
    def _ast_to_graph(self, ast):
        """Convert an AST to a graph representation for GNN processing."""
        nodes = []
        edges = []
        node_types = []
        node_map = {}
        
        def process_node(node, parent_idx=None):
            node_idx = len(nodes)
            node_map[id(node)] = node_idx
            
            # Add node features
            node_feature = self._get_node_features(node)
            nodes.append(node_feature)
            
            # Add node type
            if node["type"] in self.term_types:
                node_types.append(self.term_types[node["type"]])
            else:
                # Default to VAR type if unknown
                node_types.append(self.term_types["VAR"])
            
            # Add edge from parent to this node
            if parent_idx is not None:
                edges.append((parent_idx, node_idx))
                edges.append((node_idx, parent_idx))  # Add bidirectional edge
            
            # Process children recursively
            for child in node.get("children", []):
                child_idx = process_node(child, node_idx)
            
            return node_idx
        
        # Process the AST starting from the root
        for root_child in ast["children"]:
            process_node(root_child)
        
        # Convert to appropriate format for GNN
        edges = torch.tensor(edges, dtype=torch.long).t() if edges else torch.zeros((2, 0), dtype=torch.long)
        nodes = torch.tensor(nodes, dtype=torch.float32) if nodes else torch.zeros((0, 32), dtype=torch.float32)
        node_types = torch.tensor(node_types, dtype=torch.long) if node_types else torch.zeros((0,), dtype=torch.long)
        
        return nodes, edges, node_types
    
    def _get_node_features(self, node):
        """
        Extract features for a node based on its type and value.
        Returns a fixed-size feature vector.
        """
        # One-hot encoding for node type
        features = np.zeros(32, dtype=np.float32)
        
        if node["type"] == "VAR":
            features[0] = 1.0
        elif node["type"] == "CONST":
            features[1] = 1.0
        elif node["type"] == "COMB":
            features[2] = 1.0
        elif node["type"] == "ABS":
            features[3] = 1.0
        
        # Additional features based on the token value
        if "value" in node:
            value = node["value"]
            
            # Check if it's an operator
            if value in self.operators:
                features[4 + self.operators[value]] = 1.0
            
            # Add character-level features
            if value:
                # Check first character
                if value[0].isalpha():
                    if value[0].isupper():
                        features[16] = 1.0  # Uppercase letter
                    else:
                        features[17] = 1.0  # Lowercase letter
                elif value[0].isdigit():
                    features[18] = 1.0  # Digit
                
                # Add length feature (normalized)
                features[19] = min(len(value), 10) / 10.0
                
                # Vocabulary index (normalized)
                if value in self.vocab:
                    features[20] = self.vocab[value] / max(1, self.vocab_size)
        
        return features
    
    def parse_formula(self, formula_str):
        """
        Parse a HOL Light formula into a graph representation.
        
        Args:
            formula_str: String representation of a HOL Light formula
            
        Returns:
            nodes: Tensor of node features [num_nodes, features]
            edges: Tensor of edge indices [2, num_edges]
            node_types: Tensor of node type indices [num_nodes]
        """
        tokens = self._tokenize(formula_str)
        ast = self._build_ast(tokens)
        nodes, edges, node_types = self._ast_to_graph(ast)
        
        return nodes, edges, node_types