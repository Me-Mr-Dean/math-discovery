"""
Data processing utilities for mathematical discovery
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Dict, Any


def load_dataset(file_path: str, **kwargs) -> pd.DataFrame:
    """Load a dataset with proper error handling"""
    file_path = Path(file_path)
    
    if not file_path.exists():
        raise FileNotFoundError(f"Dataset not found: {file_path}")
    
    if file_path.suffix == '.csv':
        return pd.read_csv(file_path, **kwargs)
    elif file_path.suffix in ['.xlsx', '.xls']:
        return pd.read_excel(file_path, **kwargs)
    else:
        raise ValueError(f"Unsupported file format: {file_path.suffix}")


def save_results(data: Any, file_path: str, format: str = 'auto') -> None:
    """Save results with automatic format detection"""
    file_path = Path(file_path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    
    if format == 'auto':
        format = file_path.suffix[1:]  # Remove the dot
    
    if isinstance(data, pd.DataFrame):
        if format == 'csv':
            data.to_csv(file_path, index=False)
        elif format in ['xlsx', 'xls']:
            data.to_excel(file_path, index=False)
        else:
            raise ValueError(f"Unsupported format for DataFrame: {format}")
    else:
        # Handle other data types as needed
        raise ValueError(f"Unsupported data type: {type(data)}")


def clean_matrix_data(df: pd.DataFrame) -> pd.DataFrame:
    """Clean matrix data by removing metadata columns"""
    metadata_columns = ["range_start", "range_end", "prime_count", "prime_density"]
    data_columns = [col for col in df.columns if col not in metadata_columns]
    return df[data_columns]


def validate_sequence(sequence: List[int]) -> bool:
    """Validate that a sequence is suitable for analysis"""
    if not sequence:
        return False
    
    if len(sequence) < 3:
        return False
    
    if not all(isinstance(x, (int, float)) for x in sequence):
        return False
    
    return True
