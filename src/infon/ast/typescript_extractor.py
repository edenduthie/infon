"""TypeScript/JavaScript AST extractor using tree-sitter.

Extracts code relations from TypeScript and JavaScript source files:
- imports: import X from Y, require()
- calls: function/method calls
- inherits: class extends
- defines: function/class definitions
- returns: return statements
- raises: throw statements
- mutates: property assignments (this.x = y)
"""

from pathlib import Path

import tree_sitter_javascript as tsjs
from tree_sitter import Language, Node, Parser

from infon.ast.base import BaseASTExtractor
from infon.infon import Infon
from infon.schema import AnchorSchema


class TypeScriptASTExtractor(BaseASTExtractor):
    """Extracts code relations from TypeScript/JavaScript source files using tree-sitter."""
    
    def __init__(self, schema: AnchorSchema):
        """Initialize TypeScript/JavaScript extractor.
        
        Args:
            schema: AnchorSchema containing relation anchors
        """
        super().__init__(schema)
        self.language = Language(tsjs.language())
        self.parser = Parser(self.language)
    
    def extract(self, file_path: Path) -> list[Infon]:
        """Extract infons from a TypeScript/JavaScript source file.
        
        Args:
            file_path: Path to TypeScript/JavaScript source file
            
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
        elif node.type == "call_expression":
            self._extract_call(node, file_path, source_code, infons)
        elif node.type == "class_declaration":
            self._extract_class_decl(node, file_path, source_code, infons)
        elif node.type == "function_declaration":
            self._extract_function_decl(node, file_path, source_code, infons)
        elif node.type == "return_statement":
            self._extract_return(node, file_path, source_code, infons)
        elif node.type == "throw_statement":
            self._extract_throw(node, file_path, source_code, infons)
        elif node.type == "assignment_expression":
            self._extract_assignment(node, file_path, source_code, infons)
        
        # Recurse to children
        for child in node.children:
            self._walk_tree(child, file_path, source_code, infons)
    
    def _extract_import(self, node: Node, file_path: Path, source_code: bytes, infons: list[Infon]) -> None:
        """Extract import statement."""
        line_number = node.start_point[0] + 1
        
        # Find source string
        for child in node.children:
            if child.type == "string":
                # Remove quotes
                raw = source_code[child.start_byte:child.end_byte].decode('utf-8')
                module_name = raw.strip('"\'')
                # Extract just the module name (e.g., "./db.js" -> "db")
                if module_name.startswith('./') or module_name.startswith('../'):
                    module_name = Path(module_name).stem
                
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
    
    def _extract_call(self, node: Node, file_path: Path, source_code: bytes, infons: list[Infon]) -> None:
        """Extract function/method call."""
        line_number = node.start_point[0] + 1
        
        # Get function name
        func_name = None
        for child in node.children:
            if child.type == "identifier":
                func_name = source_code[child.start_byte:child.end_byte].decode('utf-8')
                break
            elif child.type == "member_expression":
                # For method calls like obj.method()
                prop_child = child.child_by_field_name("property")
                if prop_child and prop_child.type == "property_identifier":
                    func_name = source_code[prop_child.start_byte:prop_child.end_byte].decode('utf-8')
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
                node_type="call_expression",
            )
            infons.append(infon)
    
    def _extract_class_decl(self, node: Node, file_path: Path, source_code: bytes, infons: list[Infon]) -> None:
        """Extract class declaration and inheritance."""
        line_number = node.start_point[0] + 1
        
        class_name = None
        base_class = None
        
        # Find class name and extends clause
        for child in node.children:
            if child.type == "type_identifier" and not class_name:
                class_name = source_code[child.start_byte:child.end_byte].decode('utf-8')
            elif child.type == "class_heritage":
                # Look for extends_clause
                for heritage_child in child.children:
                    if heritage_child.type == "extends_clause":
                        # Find the identifier
                        for extends_child in heritage_child.children:
                            if extends_child.type == "identifier":
                                base_class = source_code[extends_child.start_byte:extends_child.end_byte].decode('utf-8')
                                break
        
        # Create defines infon for the class
        if class_name:
            infon = self._create_infon(
                subject=file_path.stem,
                predicate="defines",
                obj=class_name,
                file_path=file_path,
                line_number=line_number,
                node_type="class_declaration",
            )
            infons.append(infon)
        
        # Create inherits infon if there's a base class
        if class_name and base_class:
            infon = self._create_infon(
                subject=class_name,
                predicate="inherits",
                obj=base_class,
                file_path=file_path,
                line_number=line_number,
                node_type="class_declaration",
            )
            infons.append(infon)
    
    def _extract_function_decl(self, node: Node, file_path: Path, source_code: bytes, infons: list[Infon]) -> None:
        """Extract function declaration."""
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
                node_type="function_declaration",
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
                elif child.type in ("true", "false", "null"):
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
    
    def _extract_throw(self, node: Node, file_path: Path, source_code: bytes, infons: list[Infon]) -> None:
        """Extract throw statement."""
        line_number = node.start_point[0] + 1
        
        exception_name = None
        for child in node.children:
            if child.type == "new_expression":
                # Find constructor identifier
                for new_child in child.children:
                    if new_child.type == "identifier":
                        exception_name = source_code[new_child.start_byte:new_child.end_byte].decode('utf-8')
                        break
            elif child.type == "identifier":
                # Direct throw
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
                node_type="throw_statement",
            )
            infons.append(infon)
    
    def _extract_assignment(self, node: Node, file_path: Path, source_code: bytes, infons: list[Infon]) -> None:
        """Extract assignment (check for this.x = y mutations)."""
        line_number = node.start_point[0] + 1
        
        # Check if it's a this.x mutation
        is_this = False
        prop_name = None
        
        # Get left side
        left_child = node.child_by_field_name("left")
        if left_child and left_child.type == "member_expression":
            obj_child = left_child.child_by_field_name("object")
            prop_child = left_child.child_by_field_name("property")
            
            if obj_child and obj_child.type == "this":
                is_this = True
            if prop_child and prop_child.type == "property_identifier":
                prop_name = source_code[prop_child.start_byte:prop_child.end_byte].decode('utf-8')
        
        if is_this and prop_name:
            func_name = self._find_containing_function(node, source_code) or file_path.stem
            
            infon = self._create_infon(
                subject=func_name,
                predicate="mutates",
                obj=prop_name,
                file_path=file_path,
                line_number=line_number,
                node_type="assignment_expression",
            )
            infons.append(infon)
    
    def _find_containing_function(self, node: Node, source_code: bytes) -> str | None:
        """Find the name of the function containing this node."""
        current = node.parent
        while current:
            if current.type in ("function_declaration", "method_definition"):
                for child in current.children:
                    if child.type == "identifier":
                        return source_code[child.start_byte:child.end_byte].decode('utf-8')
                    elif child.type == "property_identifier":
                        return source_code[child.start_byte:child.end_byte].decode('utf-8')
            current = current.parent
        return None
