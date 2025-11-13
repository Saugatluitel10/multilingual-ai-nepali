"""
Data validation utilities based on windsurf.json configuration
"""

import json
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataValidator:
    """
    Data validation class based on Windsurf configuration
    """
    
    def __init__(self, config_path: str = "windsurf.json"):
        """
        Initialize validator with configuration
        
        Args:
            config_path: Path to windsurf.json configuration file
        """
        self.config_path = Path(config_path)
        self.config = self._load_config()
        self.validation_rules = self.config.get('data', {}).get('validation', {}).get('rules', [])
        self.on_failure = self.config.get('data', {}).get('validation', {}).get('on_failure', 'warn')
        
    def _load_config(self) -> Dict:
        """Load configuration from windsurf.json"""
        try:
            with open(self.config_path, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            logger.warning(f"⚠️  Config file not found: {self.config_path}")
            return {}
        except json.JSONDecodeError as e:
            logger.error(f"❌ Error parsing config file: {e}")
            return {}
    
    def validate_dataframe(self, df: pd.DataFrame, dataset_name: str = "dataset") -> Tuple[bool, List[str]]:
        """
        Validate a DataFrame against configured rules
        
        Args:
            df: DataFrame to validate
            dataset_name: Name of the dataset for logging
            
        Returns:
            Tuple of (is_valid, list of error messages)
        """
        logger.info(f"🔍 Validating {dataset_name}...")
        
        errors = []
        warnings = []
        
        for rule in self.validation_rules:
            rule_name = rule.get('name', 'unnamed_rule')
            rule_type = rule.get('type')
            column = rule.get('column')
            required = rule.get('required', False)
            
            try:
                if rule_type == 'column_exists':
                    if column not in df.columns:
                        msg = f"Column '{column}' does not exist"
                        if required:
                            errors.append(msg)
                        else:
                            warnings.append(msg)
                
                elif rule_type == 'not_null':
                    if column in df.columns:
                        null_count = df[column].isnull().sum()
                        if null_count > 0:
                            msg = f"Column '{column}' has {null_count} null values"
                            errors.append(msg)
                
                elif rule_type == 'min_length':
                    if column in df.columns:
                        min_value = rule.get('min_value', 0)
                        invalid_count = (df[column].astype(str).str.len() < min_value).sum()
                        if invalid_count > 0:
                            msg = f"Column '{column}' has {invalid_count} values below min length {min_value}"
                            errors.append(msg)
                
                elif rule_type == 'value_range':
                    if column in df.columns:
                        min_value = rule.get('min_value')
                        max_value = rule.get('max_value')
                        
                        if min_value is not None:
                            below_min = (df[column] < min_value).sum()
                            if below_min > 0:
                                msg = f"Column '{column}' has {below_min} values below minimum {min_value}"
                                errors.append(msg)
                        
                        if max_value is not None:
                            above_max = (df[column] > max_value).sum()
                            if above_max > 0:
                                msg = f"Column '{column}' has {above_max} values above maximum {max_value}"
                                errors.append(msg)
                
            except Exception as e:
                logger.error(f"❌ Error validating rule '{rule_name}': {e}")
                errors.append(f"Rule '{rule_name}' failed: {str(e)}")
        
        # Report results
        if errors:
            logger.error(f"❌ Validation failed for {dataset_name}:")
            for error in errors:
                logger.error(f"   - {error}")
        
        if warnings:
            logger.warning(f"⚠️  Validation warnings for {dataset_name}:")
            for warning in warnings:
                logger.warning(f"   - {warning}")
        
        if not errors and not warnings:
            logger.info(f"✅ Validation passed for {dataset_name}")
        
        is_valid = len(errors) == 0
        
        # Handle failure based on configuration
        if not is_valid and self.on_failure == 'error':
            raise ValueError(f"Data validation failed for {dataset_name}")
        
        return is_valid, errors
    
    def validate_file(self, file_path: str) -> Tuple[bool, List[str]]:
        """
        Validate a data file
        
        Args:
            file_path: Path to the data file
            
        Returns:
            Tuple of (is_valid, list of error messages)
        """
        try:
            file_path = Path(file_path)
            
            if file_path.suffix == '.csv':
                df = pd.read_csv(file_path)
            elif file_path.suffix == '.json':
                df = pd.read_json(file_path)
            else:
                return False, [f"Unsupported file format: {file_path.suffix}"]
            
            return self.validate_dataframe(df, file_path.name)
            
        except Exception as e:
            logger.error(f"❌ Error validating file {file_path}: {e}")
            return False, [str(e)]
    
    def get_validation_summary(self, df: pd.DataFrame) -> Dict:
        """
        Get a summary of data quality metrics
        
        Args:
            df: DataFrame to analyze
            
        Returns:
            Dictionary with validation summary
        """
        summary = {
            'num_rows': len(df),
            'num_columns': len(df.columns),
            'columns': list(df.columns),
            'missing_values': df.isnull().sum().to_dict(),
            'duplicate_rows': df.duplicated().sum(),
            'memory_usage_mb': df.memory_usage(deep=True).sum() / 1024**2
        }
        
        # Add column-specific statistics
        for col in df.columns:
            if df[col].dtype in ['int64', 'float64']:
                summary[f'{col}_stats'] = {
                    'min': float(df[col].min()),
                    'max': float(df[col].max()),
                    'mean': float(df[col].mean()),
                    'std': float(df[col].std())
                }
            elif df[col].dtype == 'object':
                summary[f'{col}_stats'] = {
                    'unique_values': int(df[col].nunique()),
                    'most_common': df[col].value_counts().head(3).to_dict()
                }
        
        return summary


def validate_dataset_directory(data_dir: str, validator: Optional[DataValidator] = None) -> Dict:
    """
    Validate all datasets in a directory
    
    Args:
        data_dir: Directory containing data files
        validator: DataValidator instance (creates new one if None)
        
    Returns:
        Dictionary with validation results for each file
    """
    if validator is None:
        validator = DataValidator()
    
    data_path = Path(data_dir)
    results = {}
    
    # Find all CSV and JSON files
    files = list(data_path.glob('*.csv')) + list(data_path.glob('*.json'))
    
    logger.info(f"📂 Validating {len(files)} files in {data_dir}")
    
    for file_path in files:
        is_valid, errors = validator.validate_file(str(file_path))
        results[file_path.name] = {
            'valid': is_valid,
            'errors': errors
        }
    
    # Summary
    total_files = len(results)
    valid_files = sum(1 for r in results.values() if r['valid'])
    
    logger.info(f"\n📊 Validation Summary:")
    logger.info(f"   Total files: {total_files}")
    logger.info(f"   Valid files: {valid_files}")
    logger.info(f"   Invalid files: {total_files - valid_files}")
    
    return results


if __name__ == "__main__":
    # Example usage
    print("🔍 Data Validation Demo\n")
    
    # Create sample data
    sample_data = pd.DataFrame({
        'text': [
            'यो movie राम्रो थियो',
            'I love this',
            'Bad experience',
            None,  # This will trigger validation error
        ],
        'label': [2, 2, 0, 1]
    })
    
    # Validate
    validator = DataValidator()
    is_valid, errors = validator.validate_dataframe(sample_data, "sample_data")
    
    # Get summary
    summary = validator.get_validation_summary(sample_data)
    print("\n📊 Data Summary:")
    print(json.dumps(summary, indent=2))
    
    print("\n✅ Data validation utilities ready!")
