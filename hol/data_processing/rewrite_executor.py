import subprocess
import tempfile
import os
import time
import re
from typing import Dict, List, Tuple, Optional, Any

class ExpressionParser:
    """Utility class for parsing and processing HOL Light expressions"""
    
    @staticmethod
    def tokenize(expr: str) -> List[str]:
        """Break expression into tokens"""
        if not expr or expr == "None":
            return []
            
        # Remove extra spaces
        expr = re.sub(r'\s+', ' ', expr.strip())
        
        # Tokenize: operators, parentheses, variable names, etc.
        tokens = []
        i = 0
        while i < len(expr):
            if expr[i].isspace():
                i += 1
            elif expr[i] in '().':  # Include dot for quantifiers
                tokens.append(expr[i])
                i += 1
            elif expr[i:i+3] in ['<=>', '==>', '/\\', '\\/']:
                tokens.append(expr[i:i+3])
                i += 3
            elif expr[i:i+2] in ['~~']:
                tokens.append(expr[i:i+2])
                i += 2
            elif expr[i] in '+-*=~!':
                tokens.append(expr[i])
                i += 1
            else:
                # Read variable name or number
                j = i
                while j < len(expr) and (expr[j].isalnum() or expr[j] in '_'):
                    j += 1
                if j > i:
                    tokens.append(expr[i:j])
                    i = j
                else:
                    i += 1
        
        return [token for token in tokens if token.strip()]
    
    @staticmethod
    def parse_to_tree(tokens: List[str]) -> Dict[str, Any]:
        """Parse token list into syntax tree"""
        if not tokens:
            return {'type': 'empty'}
        
        # Handle quantifiers
        if tokens[0] == '!':
            try:
                vars_end = tokens.index('.')
                variables = [t for t in tokens[1:vars_end] if t != '.']
                body_tokens = tokens[vars_end + 1:]
                return {
                    'type': 'forall',
                    'variables': variables,
                    'body': ExpressionParser.parse_to_tree(body_tokens)
                }
            except ValueError:
                # No '.' found, treat as malformed quantifier
                print(f"Warning: Malformed quantifier in tokens: {tokens}")
                return {'type': 'atom', 'value': ' '.join(tokens)}
        
        # Handle equations
        if '=' in tokens:
            eq_pos = tokens.index('=')
            left = ExpressionParser.parse_to_tree(tokens[:eq_pos])
            right = ExpressionParser.parse_to_tree(tokens[eq_pos + 1:])
            return {
                'type': 'equation',
                'left': left,
                'right': right
            }
        
        # Handle parentheses
        if tokens[0] == '(':
            paren_count = 0
            for i, token in enumerate(tokens):
                if token == '(':
                    paren_count += 1
                elif token == ')':
                    paren_count -= 1
                    if paren_count == 0:
                        if i == len(tokens) - 1:
                            # Entire expression is surrounded by parentheses
                            return ExpressionParser.parse_to_tree(tokens[1:-1])
                        else:
                            # Content after parentheses expression
                            break
        
        # Handle binary operators (by precedence)
        for ops in [['<=>'], ['==>'], ['/\\', '\\/'], ['+', '-'], ['*', '/']]:
            for op in ops:
                # Search for operators from right to left (right associative)
                paren_count = 0
                for i in range(len(tokens) - 1, -1, -1):
                    if tokens[i] == ')':
                        paren_count += 1
                    elif tokens[i] == '(':
                        paren_count -= 1
                    elif tokens[i] == op and paren_count == 0:
                        left = ExpressionParser.parse_to_tree(tokens[:i])
                        right = ExpressionParser.parse_to_tree(tokens[i + 1:])
                        return {
                            'type': 'binary_op',
                            'operator': op,
                            'left': left,
                            'right': right
                        }
        
        # Handle unary operators
        if tokens[0] in ['~', '~~']:
            return {
                'type': 'unary_op',
                'operator': tokens[0],
                'operand': ExpressionParser.parse_to_tree(tokens[1:])
            }
        
        # Single token (variable or constant)
        if len(tokens) == 1:
            return {
                'type': 'atom',
                'value': tokens[0]
            }
        
        # Default case: return first token
        return {
            'type': 'atom',
            'value': tokens[0] if tokens else ''
        }

class PatternMatcher:
    """Pattern matcher for structural matching"""
    
    def __init__(self):
        self.bindings = {}
    
    def match(self, pattern: Dict[str, Any], expression: Dict[str, Any], 
              variables: List[str] = None) -> Optional[Dict[str, Dict[str, Any]]]:
        """
        Attempt to match pattern with expression
        
        Args:
            pattern: Pattern syntax tree
            expression: Target expression syntax tree
            variables: List of bindable variables
            
        Returns:
            Binding dictionary, or None if match fails
        """
        if variables is None:
            variables = []
        
        self.bindings = {}
        if self._match_recursive(pattern, expression, variables):
            return self.bindings.copy()
        return None
    
    def _match_recursive(self, pattern: Dict[str, Any], expression: Dict[str, Any], 
                        variables: List[str]) -> bool:
        """Recursive matching function"""
        
        # If pattern is a variable
        if (pattern.get('type') == 'atom' and 
            pattern.get('value') in variables):
            var_name = pattern['value']
            if var_name in self.bindings:
                # Variable already bound, check consistency
                return self._trees_equal(self.bindings[var_name], expression)
            else:
                # Bind variable
                self.bindings[var_name] = expression
                return True
        
        # Structure must match
        if pattern.get('type') != expression.get('type'):
            return False
        
        if pattern['type'] == 'atom':
            return pattern.get('value') == expression.get('value')
        
        elif pattern['type'] == 'binary_op':
            return (pattern.get('operator') == expression.get('operator') and
                    self._match_recursive(pattern.get('left', {}), expression.get('left', {}), variables) and
                    self._match_recursive(pattern.get('right', {}), expression.get('right', {}), variables))
        
        elif pattern['type'] == 'unary_op':
            return (pattern.get('operator') == expression.get('operator') and
                    self._match_recursive(pattern.get('operand', {}), expression.get('operand', {}), variables))
        
        elif pattern['type'] == 'equation':
            return (self._match_recursive(pattern.get('left', {}), expression.get('left', {}), variables) and
                    self._match_recursive(pattern.get('right', {}), expression.get('right', {}), variables))
        
        return False
    
    def _trees_equal(self, tree1: Dict[str, Any], tree2: Dict[str, Any]) -> bool:
        """Check if two syntax trees are equal"""
        if tree1.get('type') != tree2.get('type'):
            return False
        
        if tree1['type'] == 'atom':
            return tree1.get('value') == tree2.get('value')
        elif tree1['type'] == 'binary_op':
            return (tree1.get('operator') == tree2.get('operator') and
                    self._trees_equal(tree1.get('left', {}), tree2.get('left', {})) and
                    self._trees_equal(tree1.get('right', {}), tree2.get('right', {})))
        elif tree1['type'] == 'unary_op':
            return (tree1.get('operator') == tree2.get('operator') and
                    self._trees_equal(tree1.get('operand', {}), tree2.get('operand', {})))
        
        return False

class ExpressionRewriter:
    """Expression rewriter using structural pattern matching"""
    
    def __init__(self):
        self.parser = ExpressionParser()
        self.matcher = PatternMatcher()
    
    def rewrite_with_rule(self, expression_str: str, rule_str: str) -> Optional[str]:
        """
        Rewrite expression using given rule
        
        Args:
            expression_str: Expression to rewrite
            rule_str: Rewrite rule (e.g., "!x. (x + 0) = x")
            
        Returns:
            Rewritten expression, or None if rewrite not possible
        """
        try:
            # Check for invalid inputs
            if not expression_str or expression_str == "None" or not rule_str:
                return None
                
            # Parse rule
            rule_tokens = self.parser.tokenize(rule_str)
            if not rule_tokens:
                return None
                
            rule_tree = self.parser.parse_to_tree(rule_tokens)
            
            if rule_tree.get('type') != 'forall':
                return None
            
            variables = rule_tree.get('variables', [])
            equation = rule_tree.get('body')
            
            if equation.get('type') != 'equation':
                return None
            
            pattern = equation.get('left')
            replacement = equation.get('right')
            
            # Parse target expression
            expr_tokens = self.parser.tokenize(expression_str)
            if not expr_tokens:
                return None
                
            expr_tree = self.parser.parse_to_tree(expr_tokens)
            
            # Attempt rewrite
            new_tree = self._rewrite_tree(expr_tree, pattern, replacement, variables)
            
            # Check if trees are actually different
            if not self._trees_equal(new_tree, expr_tree):
                result = self._tree_to_string(new_tree)
                # Additional check: ensure result is meaningful
                if result and result != expression_str and result != "":
                    return result
            
            return None
            
        except Exception as e:
            # Uncomment for debugging: print(f"Error in rewrite_with_rule: {e}")
            return None
    
    def _trees_equal(self, tree1: Dict[str, Any], tree2: Dict[str, Any]) -> bool:
        """Check if two syntax trees are equal"""
        if tree1.get('type') != tree2.get('type'):
            return False
        
        if tree1['type'] == 'atom':
            return tree1.get('value') == tree2.get('value')
        elif tree1['type'] == 'binary_op':
            return (tree1.get('operator') == tree2.get('operator') and
                    self._trees_equal(tree1.get('left', {}), tree2.get('left', {})) and
                    self._trees_equal(tree1.get('right', {}), tree2.get('right', {})))
        elif tree1['type'] == 'unary_op':
            return (tree1.get('operator') == tree2.get('operator') and
                    self._trees_equal(tree1.get('operand', {}), tree2.get('operand', {})))
        elif tree1['type'] == 'equation':
            return (self._trees_equal(tree1.get('left', {}), tree2.get('left', {})) and
                    self._trees_equal(tree1.get('right', {}), tree2.get('right', {})))
        
        return True  # For empty or unknown types
    
    def _rewrite_tree(self, tree: Dict[str, Any], pattern: Dict[str, Any], 
                     replacement: Dict[str, Any], variables: List[str]) -> Dict[str, Any]:
        """Recursively rewrite syntax tree"""
        
        # Try to match current node
        bindings = self.matcher.match(pattern, tree, variables)
        if bindings:
            # Match successful, apply replacement
            return self._apply_bindings(replacement, bindings)
        
        # Recursively process child nodes
        if tree.get('type') == 'binary_op':
            new_left = self._rewrite_tree(tree.get('left', {}), pattern, replacement, variables)
            new_right = self._rewrite_tree(tree.get('right', {}), pattern, replacement, variables)
            return {
                'type': 'binary_op',
                'operator': tree.get('operator'),
                'left': new_left,
                'right': new_right
            }
        
        elif tree.get('type') == 'unary_op':
            new_operand = self._rewrite_tree(tree.get('operand', {}), pattern, replacement, variables)
            return {
                'type': 'unary_op',
                'operator': tree.get('operator'),
                'operand': new_operand
            }
        
        # Other cases return original tree
        return tree
    
    def _apply_bindings(self, template: Dict[str, Any], bindings: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """Apply variable bindings to template"""
        
        if template.get('type') == 'atom':
            var_name = template.get('value')
            if var_name in bindings:
                return bindings[var_name]
            return template
        
        elif template.get('type') == 'binary_op':
            return {
                'type': 'binary_op',
                'operator': template.get('operator'),
                'left': self._apply_bindings(template.get('left', {}), bindings),
                'right': self._apply_bindings(template.get('right', {}), bindings)
            }
        
        elif template.get('type') == 'unary_op':
            return {
                'type': 'unary_op',
                'operator': template.get('operator'),
                'operand': self._apply_bindings(template.get('operand', {}), bindings)
            }
        
        return template
    
    def _tree_to_string(self, tree: Dict[str, Any]) -> str:
        """Convert syntax tree back to string"""
        
        if tree.get('type') == 'atom':
            return tree.get('value', '')
        
        elif tree.get('type') == 'binary_op':
            left_str = self._tree_to_string(tree.get('left', {}))
            right_str = self._tree_to_string(tree.get('right', {}))
            op = tree.get('operator', '')
            
            # Only add parentheses when necessary
            return f"{left_str} {op} {right_str}"
        
        elif tree.get('type') == 'unary_op':
            operand_str = self._tree_to_string(tree.get('operand', {}))
            op = tree.get('operator', '')
            return f"{op}{operand_str}"
        
        return ""

class HOLLightRewriter:
    """
    Interface with HOL Light for executing rewrites.
    Executes REWRITE_TAC on HOL Light formulas.
    """
    
    def __init__(self, hol_light_path, timeout=10):
        """
        Initialize the HOL Light rewriter.
        
        Args:
            hol_light_path: Path to HOL Light installation
            timeout: Timeout in seconds for rewrite operations
        """
        self.hol_light_path = hol_light_path
        self.timeout = timeout
    
    def execute_rewrite(self, theorem, parameter):
        """
        Execute a rewrite operation in HOL Light.
        
        Args:
            theorem: The theorem to rewrite
            parameter: The parameter theorem to use for rewriting
            
        Returns:
            success: Whether the rewrite was successful (True/False)
            result: The result of the rewrite (if successful)
        """
        # Create a temporary OCaml script to execute the rewrite
        with tempfile.NamedTemporaryFile(suffix=".ml", delete=False) as f:
            script_path = f.name
            script = f"""
            #use "{self.hol_light_path}/hol.ml";;
            
            let theorem = parse `{theorem}`;;
            let parameter = parse `{parameter}`;;
            
            (* Set up exception handling *)
            let result = 
              try 
                let goal = mk_goal [] theorem in
                let (goals, _) = REWRITE_TAC [parameter] goal in
                if goals = [] then "SUCCESS:" ^ (string_of_thm (concl (hd (justification []))))
                else if term_eq (concl goal) (concl (hd goals)) then "NO_CHANGE"
                else "SUCCESS:" ^ (string_of_term (concl (hd goals)))
              with 
                | _ -> "FAILURE"
            ;;
            
            print_string result;;
            """
            f.write(script.encode())
        
        try:
            # Execute the script with OCaml
            start_time = time.time()
            proc = subprocess.run(
                ["ocaml", script_path], 
                capture_output=True, 
                timeout=self.timeout
            )
            end_time = time.time()
            
            # Parse the output
            output = proc.stdout.decode()
            print("HOL Light output:", output)
            
            if "SUCCESS:" in output:
                # Extract the result theorem
                result = output.split("SUCCESS:")[1].strip()
                
                # Check if the result is different from the input
                if result != theorem:
                    return True, result
                else:
                    return False, None  # No change
            else:
                return False, None
                
        except subprocess.TimeoutExpired:
            print(f"Rewrite timed out after {self.timeout}s: {theorem} with {parameter}")
            return False, None
        except Exception as e:
            print(f"Error executing rewrite: {e}")
            return False, None
        finally:
            # Clean up the temporary file
            try:
                os.unlink(script_path)
            except:
                pass

class SimulatedHOLLightRewriter:
    """
    Simulated HOL Light rewriter using pattern matching and structural rewriting.
    Provides the same interface as HOLLightRewriter but works without actual HOL Light.
    """
    
    def __init__(self, timeout=10):
        """
        Initialize the simulated HOL Light rewriter.
        
        Args:
            timeout: Timeout in seconds for rewrite operations
        """
        self.timeout = timeout
        self.rewriter = ExpressionRewriter()
        
        # Load predefined rewrite rules
        self.predefined_rules = [
            "!x. (x + 0) = x",
            "!x. (0 + x) = x", 
            "!x. (x * 1) = x",
            "!x. (1 * x) = x",
            "!x. (x * 0) = 0",
            "!x. (0 * x) = 0",
            "!x y. (x + y) = (y + x)",
            "!x y. (x * y) = (y * x)",
            "!x y z. (x + (y + z)) = ((x + y) + z)",
            "!x y z. (x * (y * z)) = ((x * y) * z)",
            "!x y z. (x * (y + z)) = ((x * y) + (x * z))",
            "!x y z. ((x + y) * z) = ((x * z) + (y * z))",
            "!x. ~~x = x",
            "!x y. ~(x /\\ y) = (~x \\/ ~y)",
            "!x y. ~(x \\/ y) = (~x /\\ ~y)",
            "!x y. (x ==> y) = (~x \\/ y)",
            "!x y. (x <=> y) = ((x ==> y) /\\ (y ==> x))"
        ]
    
    def execute_rewrite(self, theorem, parameter):
        """
        Execute a rewrite operation using pattern matching.
        
        Args:
            theorem: The theorem to rewrite (string)
            parameter: The parameter theorem to use for rewriting (string)
            
        Returns:
            success: Whether the rewrite was successful (True/False)
            result: The result of the rewrite (if successful)
        """
        try:
            # Check for invalid inputs
            if not theorem or theorem == "None" or not parameter:
                print(f"Invalid input: theorem='{theorem}', parameter='{parameter}'")
                return False, None
                
            start_time = time.time()
            
            print(f"Applying rewrite rule: {parameter}")
            print(f"To theorem: {theorem}")
            
            # First try using the given parameter rule
            result = self.rewriter.rewrite_with_rule(theorem, parameter)
            
            if result and result != theorem:
                elapsed = time.time() - start_time
                print(f"Rewrite successful in {elapsed:.3f}s: {theorem} -> {result}")
                return True, result
            
            # If given rule doesn't match, try predefined rules
            for rule in self.predefined_rules:
                result = self.rewriter.rewrite_with_rule(theorem, rule)
                if result and result != theorem:
                    elapsed = time.time() - start_time
                    print(f"Rewrite successful with predefined rule '{rule}' in {elapsed:.3f}s: {theorem} -> {result}")
                    return True, result
            
            print("No applicable rewrite found")
            return False, None
            
        except Exception as e:
            print(f"Error during rewrite: {e}")
            return False, None

# Test both rewriters
if __name__ == "__main__":
    print("=== Testing Simulated HOL Light Rewriter ===")
    
    # Test simulated HOL Light rewriter  
    rewriter = SimulatedHOLLightRewriter()
    success1, result1 = rewriter.execute_rewrite(
        "(a + 0) * (b * 1)", 
        "!x. (x + 0) = x"
    )
    print(f"Step 1: success={success1}, result='{result1}'\n")
    
    # Test case 2 - only proceed if step 1 was successful
    if success1 and result1:
        success2, result2 = rewriter.execute_rewrite(
            result1, 
            "!x. (x * 1) = x"
        )
        print(f"Step 2: success={success2}, result='{result2}'\n")
    else:
        print("Step 1 failed, skipping step 2\n")
    
    # Additional simple tests
    print("=== Additional Test Cases ===")
    
    test_cases = [
        ("a + 0", "!x. (x + 0) = x"),
        ("1 * x", "!x. (1 * x) = x"),
        ("0 * y", "!x. (0 * x) = 0"),
    ]
    
    for i, (expr, rule) in enumerate(test_cases, 3):
        print(f"Test {i}: Applying '{rule}' to '{expr}'")
        success, result = rewriter.execute_rewrite(expr, rule)
        print(f"  Result: success={success}, result='{result}'\n")