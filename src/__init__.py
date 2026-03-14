"""
src/
====
Package chứa core modules của VN Corporate Credit Rating System.

Modules:
    column_mapper   → Chuẩn hóa cột từ mọi nguồn dữ liệu
    script          → DataProcessor, FeatureEngineer, Model, Pipeline
    vn_crawler      → Thu thập BCTC qua vnstock
    vn_data_adapter → Transform raw BCTC → feature vector
    data_merger     → Gộp và cân bằng các datasets
"""

from src.column_mapper import ColumnMapper, CANONICAL_FEATURES, CANONICAL_META
from src.data_merger import DataMerger

# script.py import xgboost — lazy import để không fail khi chưa cài
def _import_pipeline():
    from src.script import Config, CreditRatingPipeline, DataProcessor, CreditRatingModel
    return Config, CreditRatingPipeline, DataProcessor, CreditRatingModel

__all__ = [
    'ColumnMapper',
    'CANONICAL_FEATURES',
    'CANONICAL_META',
    'DataMerger',
    '_import_pipeline',
]
