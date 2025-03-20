"""
Comprehensive Documentation Management System.

This module provides tools for managing, generating, validating, and maintaining
documentation across the entire trading platform, including API documentation.
"""

import inspect
import logging
import os
import re
import sys
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Type, Union
import importlib
import json
import yaml
from concurrent.futures import ThreadPoolExecutor

logger = logging.getLogger(__name__)


class DocumentationType(Enum):
    """Types of documentation supported by the system."""
    API = "api"
    USER = "user"
    DEVELOPER = "developer"
    SYSTEM = "system"
    ARCHITECTURE = "architecture"


@dataclass
class DocumentationMetadata:
    """Metadata for a documentation item."""
    author: str
    created_at: str
    updated_at: str
    version: str
    status: str  # draft, review, approved
    related_components: List[str]
    tags: List[str]


@dataclass
class DocumentationItem:
    """A single documentation item."""
    id: str
    title: str
    content: str
    doc_type: DocumentationType
    metadata: DocumentationMetadata
    path: Optional[Path] = None


class DocumentationValidator:
    """Validates documentation for completeness and correctness."""
    
    def __init__(self):
        self.validation_rules = []
        self._register_default_rules()
        
    def _register_default_rules(self):
        """Register default validation rules."""
        self.register_rule(self._validate_metadata)
        self.register_rule(self._validate_content_length)
        self.register_rule(self._validate_links)
        self.register_rule(self._validate_code_examples)
        
    def register_rule(self, rule_func: Callable[[DocumentationItem], Tuple[bool, List[str]]]):
        """Register a new validation rule.
        
        Args:
            rule_func: Function that takes a DocumentationItem and returns
                       (is_valid, list_of_issues)
        """
        self.validation_rules.append(rule_func)
        
    def validate(self, doc_item: DocumentationItem) -> Tuple[bool, List[str]]:
        """Validate a documentation item against all rules.
        
        Args:
            doc_item: The documentation item to validate
            
        Returns:
            Tuple of (is_valid, list_of_issues)
        """
        all_issues = []
        is_valid = True
        
        for rule in self.validation_rules:
            valid, issues = rule(doc_item)
            if not valid:
                is_valid = False
                all_issues.extend(issues)
                
        return is_valid, all_issues
    
    def _validate_metadata(self, doc_item: DocumentationItem) -> Tuple[bool, List[str]]:
        """Validate documentation metadata."""
        issues = []
        
        # Check required metadata fields
        if not doc_item.metadata.author:
            issues.append("Missing author information")
        if not doc_item.metadata.version:
            issues.append("Missing version information")
        
        # Validate date formats
        date_pattern = r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z$"
        if not re.match(date_pattern, doc_item.metadata.created_at):
            issues.append("Invalid created_at date format")
        if not re.match(date_pattern, doc_item.metadata.updated_at):
            issues.append("Invalid updated_at date format")
            
        return len(issues) == 0, issues
    
    def _validate_content_length(self, doc_item: DocumentationItem) -> Tuple[bool, List[str]]:
        """Validate documentation content length."""
        issues = []
        
        if len(doc_item.content) < 100:
            issues.append("Content is too short (< 100 characters)")
        
        return len(issues) == 0, issues
    
    def _validate_links(self, doc_item: DocumentationItem) -> Tuple[bool, List[str]]:
        """Validate links in documentation."""
        issues = []
        
        # Find all links in markdown format
        links = re.findall(r'\[([^\]]+)\]\(([^)]+)\)', doc_item.content)
        
        for text, url in links:
            if not url.startswith(('http://', 'https://', '#', '/')):
                issues.append(f"Invalid link format: {url}")
        
        return len(issues) == 0, issues
    
    def _validate_code_examples(self, doc_item: DocumentationItem) -> Tuple[bool, List[str]]:
        """Validate code examples in documentation."""
        issues = []
        
        # Find all code blocks
        code_blocks = re.findall(r'```(\w+)?\n(.*?)\n```', doc_item.content, re.DOTALL)
        
        for lang, code in code_blocks:
            if lang.lower() == 'python':
                try:
                    compile(code, '<string>', 'exec')
                except SyntaxError as e:
                    issues.append(f"Python code example has syntax error: {str(e)}")
        
        return len(issues) == 0, issues


class APIDocumentationGenerator:
    """Generates API documentation from code."""
    
    def __init__(self, modules: List[str]):
        """Initialize with a list of module paths to document.
        
        Args:
            modules: List of module import paths
        """
        self.modules = modules
        self.documented_items = {}
        
    async def generate(self) -> Dict[str, DocumentationItem]:
        """Generate API documentation for all registered modules.
        
        Returns:
            Dictionary of DocumentationItems keyed by ID
        """
        for module_path in self.modules:
            try:
                module = importlib.import_module(module_path)
                self._document_module(module)
            except ImportError as e:
                logger.error(f"Failed to import {module_path}: {str(e)}")
        
        return self.documented_items
    
    def _document_module(self, module: Any) -> None:
        """Generate documentation for a module."""
        module_doc = inspect.getdoc(module) or "No module documentation"
        
        module_name = module.__name__
        module_id = f"api.{module_name}"
        
        metadata = DocumentationMetadata(
            author="System Generated",
            created_at=self._get_iso_now(),
            updated_at=self._get_iso_now(),
            version="1.0.0",
            status="approved",
            related_components=[],
            tags=["api", "generated"]
        )
        
        doc_item = DocumentationItem(
            id=module_id,
            title=f"API: {module_name}",
            content=self._format_module_doc(module, module_doc),
            doc_type=DocumentationType.API,
            metadata=metadata
        )
        
        self.documented_items[module_id] = doc_item
        
        # Document classes in the module
        for name, obj in inspect.getmembers(module, inspect.isclass):
            if obj.__module__ == module.__name__:
                self._document_class(obj, module_id)
    
    def _document_class(self, cls: Type, parent_id: str) -> None:
        """Generate documentation for a class."""
        class_doc = inspect.getdoc(cls) or "No class documentation"
        
        class_id = f"{parent_id}.{cls.__name__}"
        
        metadata = DocumentationMetadata(
            author="System Generated",
            created_at=self._get_iso_now(),
            updated_at=self._get_iso_now(),
            version="1.0.0",
            status="approved",
            related_components=[parent_id],
            tags=["api", "class", "generated"]
        )
        
        doc_item = DocumentationItem(
            id=class_id,
            title=f"Class: {cls.__name__}",
            content=self._format_class_doc(cls, class_doc),
            doc_type=DocumentationType.API,
            metadata=metadata
        )
        
        self.documented_items[class_id] = doc_item
        
        # Document methods in the class
        for name, method in inspect.getmembers(cls, inspect.isfunction):
            if not name.startswith('_') or name == '__init__':
                self._document_method(method, class_id, cls.__name__)
    
    def _document_method(self, method: Callable, parent_id: str, class_name: str) -> None:
        """Generate documentation for a method."""
        method_doc = inspect.getdoc(method) or "No method documentation"
        
        method_id = f"{parent_id}.{method.__name__}"
        
        metadata = DocumentationMetadata(
            author="System Generated",
            created_at=self._get_iso_now(),
            updated_at=self._get_iso_now(),
            version="1.0.0",
            status="approved",
            related_components=[parent_id],
            tags=["api", "method", "generated"]
        )
        
        signature = inspect.signature(method)
        
        doc_item = DocumentationItem(
            id=method_id,
            title=f"Method: {class_name}.{method.__name__}",
            content=self._format_method_doc(method, method_doc, signature),
            doc_type=DocumentationType.API,
            metadata=metadata
        )
        
        self.documented_items[method_id] = doc_item
    
    def _format_module_doc(self, module: Any, doc_string: str) -> str:
        """Format module documentation."""
        return f"""# Module: {module.__name__}

{doc_string}

## Module Attributes

{self._get_module_attributes(module)}

## Module Classes

{self._get_module_classes(module)}

## Module Functions

{self._get_module_functions(module)}
"""
    
    def _format_class_doc(self, cls: Type, doc_string: str) -> str:
        """Format class documentation."""
        return f"""# Class: {cls.__name__}

{doc_string}

## Inheritance

{self._get_class_inheritance(cls)}

## Class Attributes

{self._get_class_attributes(cls)}

## Methods

{self._get_class_methods(cls)}
"""
    
    def _format_method_doc(self, method: Callable, doc_string: str, signature: inspect.Signature) -> str:
        """Format method documentation."""
        return f"""# Method: {method.__name__}{signature}

{doc_string}

## Parameters

{self._get_method_parameters(signature, doc_string)}

## Returns

{self._get_method_returns(doc_string)}

## Raises

{self._get_method_raises(doc_string)}
"""
    
    def _get_module_attributes(self, module: Any) -> str:
        """Extract module attributes for documentation."""
        attributes = []
        for name, value in inspect.getmembers(module):
            if not name.startswith('_') and not inspect.isfunction(value) and not inspect.isclass(value) and not inspect.ismodule(value):
                attributes.append(f"- **{name}**: `{type(value).__name__}`")
        
        return "\n".join(attributes) if attributes else "No attributes."
    
    def _get_module_classes(self, module: Any) -> str:
        """Extract module classes for documentation."""
        classes = []
        for name, value in inspect.getmembers(module, inspect.isclass):
            if value.__module__ == module.__name__:
                classes.append(f"- **{name}**: {inspect.getdoc(value) or 'No description.'}")
        
        return "\n".join(classes) if classes else "No classes."
    
    def _get_module_functions(self, module: Any) -> str:
        """Extract module functions for documentation."""
        functions = []
        for name, value in inspect.getmembers(module, inspect.isfunction):
            if value.__module__ == module.__name__ and not name.startswith('_'):
                functions.append(f"- **{name}**: {inspect.getdoc(value) or 'No description.'}")
        
        return "\n".join(functions) if functions else "No functions."
    
    def _get_class_inheritance(self, cls: Type) -> str:
        """Extract class inheritance for documentation."""
        if cls.__bases__ == (object,):
            return "No inheritance (inherits from `object`)."
        
        return "\n".join([f"- {base.__module__}.{base.__name__}" for base in cls.__bases__])
    
    def _get_class_attributes(self, cls: Type) -> str:
        """Extract class attributes for documentation."""
        attributes = []
        for name, value in inspect.getmembers(cls):
            if not name.startswith('__') and not inspect.isfunction(value) and not inspect.ismethod(value) and not inspect.isclass(value):
                attributes.append(f"- **{name}**: `{type(value).__name__}`")
        
        return "\n".join(attributes) if attributes else "No attributes."
    
    def _get_class_methods(self, cls: Type) -> str:
        """Extract class methods for documentation."""
        methods = []
        for name, method in inspect.getmembers(cls, inspect.isfunction):
            if not name.startswith('__') or name == '__init__':
                signature = inspect.signature(method)
                methods.append(f"- **{name}{signature}**: {inspect.getdoc(method) or 'No description.'}")
        
        return "\n".join(methods) if methods else "No methods."
    
    def _get_method_parameters(self, signature: inspect.Signature, doc_string: str) -> str:
        """Extract method parameters from signature and docstring."""
        params = []
        for name, param in signature.parameters.items():
            if name == 'self':
                continue
                
            # Try to extract param description from docstring
            param_desc = "No description."
            param_match = re.search(rf'{name}:\s*(.+?)(?:\n\s+\w+:|$)', doc_string, re.MULTILINE)
            if param_match:
                param_desc = param_match.group(1).strip()
            
            param_type = param.annotation if param.annotation != inspect.Parameter.empty else "Any"
            param_type_str = getattr(param_type, "__name__", str(param_type))
            
            default = ""
            if param.default != inspect.Parameter.empty:
                default = f" (default: `{param.default}`)"
            
            params.append(f"- **{name}** (`{param_type_str}`){default}: {param_desc}")
        
        return "\n".join(params) if params else "No parameters."
    
    def _get_method_returns(self, doc_string: str) -> str:
        """Extract return information from docstring."""
        returns_match = re.search(r'Returns?:\s*(.+?)(?:\n\s+\w+:|$)', doc_string, re.MULTILINE | re.DOTALL)
        if returns_match:
            return returns_match.group(1).strip()
        return "Not specified."
    
    def _get_method_raises(self, doc_string: str) -> str:
        """Extract exceptions from docstring."""

