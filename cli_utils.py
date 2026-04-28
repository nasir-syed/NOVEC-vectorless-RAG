"""
Utility functions and validators for the CLI application.
"""

import os
import logging
from pathlib import Path
from typing import List, Optional

logger = logging.getLogger(__name__)

# FILE VALIDATION

def validate_file_path(file_path: str) -> Optional[str]:
    if not file_path or not isinstance(file_path, str):
        return None

    # Expand user paths (support ~)
    expanded_path = os.path.expanduser(file_path.strip())

    # Check existence
    if not os.path.exists(expanded_path):
        return None

    # Check if it's a file (not directory)
    if not os.path.isfile(expanded_path):
        return None

    # Check readability
    if not os.access(expanded_path, os.R_OK):
        return None

    return expanded_path

def get_file_info(file_path: str) -> dict:
    try:
        stat_info = os.stat(file_path)
        return {
            "name": os.path.basename(file_path),
            "size": stat_info.st_size,
            "size_kb": stat_info.st_size / 1024,
            "size_mb": stat_info.st_size / (1024 * 1024),
            "exists": True
        }
    except Exception as e:
        logger.error(f"Error getting file info: {str(e)}")
        return {"exists": False}

def is_empty_file(file_path: str) -> bool:
    try:
        return os.path.getsize(file_path) == 0
    except Exception:
        return True

def is_supported_format(file_path: str, supported_extensions: List[str] = None) -> bool:
    if supported_extensions is None:
        supported_extensions = [".pdf", ".docx", ".doc", ".txt", ".md"]

    _, ext = os.path.splitext(file_path)
    return ext.lower() in supported_extensions

# INPUT VALIDATION

def validate_numeric_input(user_input: str, min_val: int = None, max_val: int = None) -> Optional[int]:
    try:
        value = int(user_input.strip())
        
        if min_val is not None and value < min_val:
            return None
        if max_val is not None and value > max_val:
            return None
            
        return value
    except ValueError:
        return None

def validate_comma_separated_numbers(input_str: str, max_count: int = None, max_value: int = None) -> Optional[List[int]]:
    try:
        numbers = [int(x.strip()) for x in input_str.split(",")]
        
        if max_count and len(numbers) > max_count:
            return None
        
        if max_value and any(n > max_value for n in numbers):
            return None
        
        if len(numbers) != len(set(numbers)):
            return None
        
        if any(n < 1 for n in numbers):
            return None
        
        return numbers
    except ValueError:
        return None

def validate_non_empty_string(user_input: str, min_length: int = 1, max_length: int = None) -> Optional[str]:
    
    cleaned = user_input.strip()
    
    if len(cleaned) < min_length:
        return None
    
    if max_length and len(cleaned) > max_length:
        return None
    
    return cleaned

# API VALIDATION

def validate_api_key(api_key: str, min_length: int = 10) -> bool:
    
    if not api_key or not isinstance(api_key, str):
        return False
    
    api_key = api_key.strip()
    
    if len(api_key) < min_length:
        return False
    
    # Should not have common placeholders
    placeholders = ["your_", "example", "placeholder", "xxx", "yyy"]
    if any(p in api_key.lower() for p in placeholders):
        return False
    
    return True

# DOCUMENT VALIDATION

def validate_document_id(doc_id: str) -> bool:
   
    if not doc_id or not isinstance(doc_id, str):
        return False
    
    # Should be non-empty and reasonable length
    return 1 <= len(doc_id) <= 256

def validate_document_data(doc_data: dict) -> bool:
    
    required_fields = ["doc_id", "name"]
    
    if not isinstance(doc_data, dict):
        return False
    
    return all(field in doc_data for field in required_fields)

# QUERY VALIDATION

def validate_query(query: str, min_length: int = 3, max_length: int = 1000) -> Optional[str]:
    
    cleaned = query.strip()
    
    if len(cleaned) < min_length:
        return None
    
    if len(cleaned) > max_length:
        return None
    
    return cleaned

# TREE VALIDATION

def is_valid_tree(tree: any) -> bool:
   
    if not isinstance(tree, list):
        return False
    
    if not tree:  # empty list is invalid
        return False
    
    # check if the first node has required fields
    first_node = tree[0]
    required = ["node_id", "title"]
    
    return isinstance(first_node, dict) and all(k in first_node for k in required)

def count_tree_nodes(nodes: list) -> int:

    total = len(nodes)
    for node in nodes:
        if node.get("nodes"):
            total += count_tree_nodes(node["nodes"])
    return total

# RESPONSE PARSING

def safe_get_nested(data: dict, keys: List[str], default: any = None) -> any:
    
    current = data
    for key in keys:
        if isinstance(current, dict):
            current = current.get(key)
        else:
            return default
    return current if current is not None else default

def extract_doc_id(response: dict) -> Optional[str]:
    
    return safe_get_nested(response, ["doc_id"])

def extract_documents_list(response: dict) -> list:
    
    docs = safe_get_nested(response, ["docs"], [])
    if docs:
        return docs
    
    docs = safe_get_nested(response, ["result"], [])
    if docs:
        return docs
    
    if isinstance(response, list):
        return response
    
    return []

# FORMATTING HELPERS

def format_file_size(size_bytes: int) -> str:
    
    for unit in ["B", "KB", "MB", "GB"]:
        if size_bytes < 1024:
            return f"{size_bytes:.2f} {unit}"
        size_bytes /= 1024
    return f"{size_bytes:.2f} TB"

def truncate_text(text: str, max_length: int = 100, suffix: str = "...") -> str:
    
    if not isinstance(text, str):
        return str(text)
    
    if len(text) <= max_length:
        return text
    
    return text[:max_length - len(suffix)] + suffix

def pluralize(count: int, singular: str, plural: str = None) -> str:
    
    if plural is None:
        plural = singular + "s"
    
    return singular if count == 1 else plural
