"""Python AST extractor using tree-sitter.

Extracts code relations from Python source files:
- imports: from X import Y, import X
- calls: function/method calls
- inherits: class inheritance
- defines: function/class definitions
- returns: return statements
- raises: raise statements
- decorates: decorator applications
- mutates: attribute assignments (self.x = y)
"""

from pathlib import Path

import tree_sitter_python as tspython
from tree_sitter import Language, Node, Parser

from infon.ast.base import BaseASTExtractor
from infon.infon import Infon
from infon.schema import AnchorSchema


class PythonASTExtractor(BaseASTExtractor):
    """Extracts code relations from Python source files using tree-sitter."""
    
    def __init__(self, schema: AnchorSchema):
        """Initialize Python extractor.
        
        Args:
            schema: AnchorSchema containing relation anchors
        """
        super().__init__(schema)
        self.language = Language(tspython.language())
        self.parser = Parser(self.language)
    
    def extract(self, file_path: Path) -> list[Infon]:
        """Extract infons from a Python source file.
        
        Args:
            file_path: Path to Python source file
            
        Returns:
            List of extracted Infons
        """
        infons: list[Infon] = []
        
        try:
            source_code = file_path.read_bytes()
            tree = self.parser.parse(source_code)
            root_node = tree.root_node
            
            # Walk the tree and extract relations
            self._walk_tree(root_node, file_path, source_code, infons)
            
        except Exception as e:
            # Log and skip files that can't be parsed
            print(f"Warning: Failed to extract from {file_path}: {e}")
        
        return infons
    
    def _walk_tree(self, node: Node, file_path: Path, source_code: bytes, infons: list[Infon]) -> None:
        """Walk the AST and extract infons."""
        # Process this node
        if node.type == "import_statement":
            self._extract_import(node, file_path, source_code, infons)
        elif node.type == "import_from_statement":
            self._extract_import_from(node, file_path, source_code, infons)
        elif node.type == "call":
            self._extract_call(node, file_path, source_code, infons)
        elif node.type == "class_definition":
            self._extract_class_def(node, file_path, source_code, infons)
        elif node.type == "function_definition":
            self._extract_function_def(node, file_path, source_code, infons)
        elif node.type == "return_statement":
            self._extract_return(node, file_path, source_code, infons)
        elif node.type == "raise_statement":
            self._extract_raise(node, file_path, source_code, infons)
        elif node.type == "decorated_definition":
            self._extract_decorator(node, file_path, source_code, infons)
        elif node.type == "assignment":
            self._extract_assignment(node, file_path, source_code, infons)
        
        # Recurse to children
        for child in node.children:
            self._walk_tree(child, file_path, source_code, infons)
    
    def _extract_import(self, node: Node, file_path: Path, source_code: bytes, infons: list[Infon]) -> None:
        """Extract import statement (import X)."""
        line_number = node.start_point[0] + 1
        
        # Find dotted_name child
        for child in node.children:
            if child.type == "dotted_name":
                module_name = source_code[child.start_byte:child.end_byte].decode('utf-8')
                infon = self._create_infon(
                    subject=file_path.stem,
                    predicate="imports",
                    obj=module_name,
                    file_path=file_path,
                    line_number=line_number,
                    node_type="import_statement",
                )
                infons.append(infon)
                break
    
    def _extract_import_from(self, node: Node, file_path: Path, source_code: bytes, infons: list[Infon]) -> None:
        """Extract from X import Y statement."""
        line_number = node.start_point[0] + 1
        
        # Find module name (dotted_name child)
        for child in node.children:
            if child.type == "dotted_name":
                module_name = source_code[child.start_byte:child.end_byte].decode('utf-8')
                infon = self._create_infon(
                    subject=file_path.stem,
                    predicate="imports",
                    obj=module_name,
                    file_path=file_path,
                    line_number=line_number,
                    node_type="import_from_statement",
                )
                infons.append(infon)
                break
    
    def _extract_call(self, node: Node, file_path: Path, source_code: bytes, infons: list[Infon]) -> None:
        """Extract function/method call."""
        line_number = node.start_point[0] + 1
        
        # Get function name from first child
        func_name = None
        for child in node.children:
            if child.type == "identifier":
                func_name = source_code[child.start_byte:child.end_byte].decode('utf-8')
                break
            elif child.type == "attribute":
                # For method calls like obj.method()
                for attr_child in child.children:
                    if attr_child.type == "identifier" and attr_child.parent.child_by_field_name("attribute") == attr_child:
                        func_name = source_code[attr_child.start_byte:attr_child.end_byte].decode('utf-8')
                        break
                if func_name:
                    break
        
        if func_name:
            # Find containing function for subject
            subject = self._find_containing_function(node, source_code) or file_path.stem
            
            infon = self._create_infon(
                subject=subject,
                predicate="calls",
                obj=func_name,
                file_path=file_path,
                line_number=line_number,
                node_type="call",
            )
            infons.append(infon)
    
    def _extract_class_def(self, node: Node, file_path: Path, source_code: bytes, infons: list[Infon]) -> None:
        """Extract class definition and inheritance."""
        line_number = node.start_point[0] + 1
        
        class_name = None
        base_classes = []
        
        # Find class name and superclasses
        for child in node.children:
            if child.type == "identifier":
                class_name = source_code[child.start_byte:child.end_byte].decode('utf-8')
            elif child.type == "argument_list":
                # Superclasses are in argument_list
                for arg in child.children:
                    if arg.type == "identifier":
                        base_classes.append(source_code[arg.start_byte:arg.end_byte].decode('utf-8'))
        
        # Create defines infon for the class itself
        if class_name:
            infon = self._create_infon(
                subject=file_path.stem,
                predicate="defines",
                obj=class_name,
                file_path=file_path,
                line_number=line_number,
                node_type="class_definition",
            )
            infons.append(infon)
        
        # Create inherits infons for each base class
        if class_name and base_classes:
            for base_class in base_classes:
                infon = self._create_infon(
                    subject=class_name,
                    predicate="inherits",
                    obj=base_class,
                    file_path=file_path,
                    line_number=line_number,
                    node_type="class_definition",
                )
                infons.append(infon)
    
    def _extract_function_def(self, node: Node, file_path: Path, source_code: bytes, infons: list[Infon]) -> None:
        """Extract function definition."""
        line_number = node.start_point[0] + 1
        
        # Find function name
        func_name = None
        for child in node.children:
            if child.type == "identifier":
                func_name = source_code[child.start_byte:child.end_byte].decode('utf-8')
                break
        
        if func_name:
            infon = self._create_infon(
                subject=file_path.stem,
                predicate="defines",
                obj=func_name,
                file_path=file_path,
                line_number=line_number,
                node_type="function_definition",
            )
            infons.append(infon)
    
    def _extract_return(self, node: Node, file_path: Path, source_code: bytes, infons: list[Infon]) -> None:
        """Extract return statement."""
        line_number = node.start_point[0] + 1
        
        # Find containing function
        func_name = self._find_containing_function(node, source_code)
        if func_name:
            # Get return value if simple identifier
            return_value = "value"
            for child in node.children:
                if child.type == "identifier":
                    return_value = source_code[child.start_byte:child.end_byte].decode('utf-8')
                    break
                elif child.type in ("true", "false", "none"):
                    return_value = source_code[child.start_byte:child.end_byte].decode('utf-8')
                    break
            
            infon = self._create_infon(
                subject=func_name,
                predicate="returns",
                obj=return_value,
                file_path=file_path,
                line_number=line_number,
                node_type="return_statement",
            )
            infons.append(infon)
    
    def _extract_raise(self, node: Node, file_path: Path, source_code: bytes, infons: list[Infon]) -> None:
        """Extract raise statement."""
        line_number = node.start_point[0] + 1
        
        exception_name = None
        for child in node.children:
            if child.type == "call":
                for call_child in child.children:
                    if call_child.type == "identifier":
                        exception_name = source_code[call_child.start_byte:call_child.end_byte].decode('utf-8')
                        break
            elif child.type == "identifier":
                # Direct raise (e.g., raise ValueError)
                exception_name = source_code[child.start_byte:child.end_byte].decode('utf-8')
                break
        
        if exception_name:
            func_name = self._find_containing_function(node, source_code) or file_path.stem
            
            infon = self._create_infon(
                subject=func_name,
                predicate="raises",
                obj=exception_name,
                file_path=file_path,
                line_number=line_number,
                node_type="raise_statement",
            )
            infons.append(infon)
    
    def _extract_decorator(self, node: Node, file_path: Path, source_code: bytes, infons: list[Infon]) -> None:
        """Extract decorated definition."""
        line_number = node.start_point[0] + 1
        
        decorators = []
        func_name = None
        
        # Find decorators and function name
        for child in node.children:
            if child.type == "decorator":
                for dec_child in child.children:
                    if dec_child.type == "identifier":
                        decorator_name = source_code[dec_child.start_byte:dec_child.end_byte].decode('utf-8')
                        decorators.append(decorator_name)
                        break
            elif child.type == "function_definition":
                for func_child in child.children:
                    if func_child.type == "identifier":
                        func_name = source_code[func_child.start_byte:func_child.end_byte].decode('utf-8')
                        break
        
        # Create infons for each decorator
        if func_name:
            for decorator_name in decorators:
                infon = self._create_infon(
                    subject=decorator_name,
                    predicate="decorates",
                    obj=func_name,
                    file_path=file_path,
                    line_number=line_number,
                    node_type="decorator",
                )
                infons.append(infon)
    
    def _extract_assignment(self, node: Node, file_path: Path, source_code: bytes, infons: list[Infon]) -> None:
        """Extract assignment (check for self.x = y mutations)."""
        line_number = node.start_point[0] + 1
        
        # Check if it's a self.x mutation
        obj_name = None
        attr_name = None
        
        for child in node.children:
            if child.type == "attribute":
                # Get object and attribute
                obj_child = child.child_by_field_name("object")
                attr_child = child.child_by_field_name("attribute")
                
                if obj_child and obj_child.type == "identifier":
                    obj_name = source_code[obj_child.start_byte:obj_child.end_byte].decode('utf-8')
                if attr_child and attr_child.type == "identifier":
                    attr_name = source_code[attr_child.start_byte:attr_child.end_byte].decode('utf-8')
                break
        
        if obj_name == "self" and attr_name:
            func_name = self._find_containing_function(node, source_code) or file_path.stem
            
            infon = self._create_infon(
                subject=func_name,
                predicate="mutates",
                obj=attr_name,
                file_path=file_path,
                line_number=line_number,
                node_type="assignment",
            )
            infons.append(infon)
    
    def _find_containing_function(self, node, source_code: bytes) -> str | None:
        """Find the name of the function containing this node."""
        current = node.parent
        while current:
            if current.type == "function_definition":
                for child in current.children:
                    if child.type == "identifier":
                        return source_code[child.start_byte:child.end_byte].decode('utf-8')
            current = current.parent
        return None
