import ast
import matplotlib

matplotlib.use('Agg')  # Use a non-interactive backend (no plots will show)

import vip_ivp as vip  # Ensure vip is imported


class Variable:
    """Class representing a variable in the code with its dependencies, expression, and value."""

    def __init__(self, name, dependencies, expression, value):
        self.name = name
        self.dependencies = dependencies
        self.expression = expression
        self.value = value  # Store the value of the variable

    def __repr__(self):
        return f"Variable(name={self.name}, dependencies={self.dependencies}, expression={self.expression}, value={self.value})"


class DependencyVisitor(ast.NodeVisitor):
    def __init__(self, env):
        self.env = env
        self.variables = []  # Store a list of Variable objects
        self.imported_modules = set()  # Track imported module names

    def visit_Import(self, node):
        """Handle import statements."""
        for alias in node.names:
            self.imported_modules.add(alias.asname or alias.name)

    def visit_ImportFrom(self, node):
        """Handle from-import statements."""
        self.imported_modules.add(node.module)

    def visit_Assign(self, node):
        """Handle variable assignments."""
        targets = [t.id for t in node.targets if isinstance(t, ast.Name)]
        used_vars = set()

        # Extract variables used on the right-hand side of the assignment
        for subnode in ast.walk(node.value):
            if isinstance(subnode, ast.Name) and subnode.id in self.env:
                used_vars.add(subnode.id)

        for target in targets:
            variable_value = self.env.get(target, None)

            # Check if the variable is an instance of vip.TemporalVariable
            if isinstance(variable_value, vip.TemporalVar):
                # Create a Variable object with the name, dependencies, expression, and value
                expression = ast.dump(node.value)  # Get a string representation of the expression
                variable = Variable(target, list(used_vars - self.imported_modules), expression, variable_value)
                self.variables.append(variable)

    def visit_Call(self, node):
        """Handle function calls like loop_into()."""
        if isinstance(node.func, ast.Attribute):  # e.g., d_n.loop_into()
            if node.func.attr == "loop_into" and node.args:
                used_vars = self.extract_used_variables(node.args[0])  # Get used variables in loop_into argument
                if isinstance(node.func.value, ast.Name):  # e.g., d_n
                    caller_var = node.func.value.id
                    # Find the variable object for caller_var
                    caller_variable = next((var for var in self.variables if var.name == caller_var), None)
                    if caller_variable:
                        caller_variable.dependencies.extend(used_vars)

    def extract_used_variables(self, node):
        """Recursively extracts all variables used in an expression."""
        used_vars = set()

        # If the node is a variable (ast.Name), add it to used_vars
        if isinstance(node, ast.Name):
            used_vars.add(node.id)

        # If it's a binary operation, extract from both sides
        elif isinstance(node, ast.BinOp):
            used_vars.update(self.extract_used_variables(node.left))
            used_vars.update(self.extract_used_variables(node.right))

        # If it's a unary operation, extract from the operand
        elif isinstance(node, ast.UnaryOp):
            used_vars.update(self.extract_used_variables(node.operand))

        # If it's a function call, extract from arguments
        elif isinstance(node, ast.Call):
            for arg in node.args:
                used_vars.update(self.extract_used_variables(arg))

        # If it's an attribute (e.g., d_n.loop_into), extract from the value
        elif isinstance(node, ast.Attribute):
            used_vars.update(self.extract_used_variables(node.value))

        return used_vars


def extract_dependencies(file_path):
    """Extracts declared variables and their dependencies from a Python file."""
    with open(file_path, "r", encoding="utf-8") as f:
        code = f.read()

    # Execute the script to capture variable values
    env = {}
    exec(code, env)

    # Parse the AST
    tree = ast.parse(code)

    # Create the visitor and traverse the AST
    visitor = DependencyVisitor(env)

    # First, extract imports (for the imported modules handling)
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            visitor.visit(node)

    # Now, visit all nodes to extract dependencies
    visitor.visit(tree)

    return visitor.variables  # Return the list of Variable objects


# Example Usage:
file_path = "../demos/exponential_decay.py"
variables = extract_dependencies(file_path)

for var in variables:
    print(var)
