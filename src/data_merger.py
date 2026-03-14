"""
src/data_merger.py
==================
Gộp và cân bằng các datasets từ nhiều nguồn khác nhau thành một
training set chuẩn hóa duy nhất, sẵn sàng đưa vào pipeline.

Trách nhiệm:
  - Merge Kaggle (US rated) + VN rated + VN unrated
  - Upweight VN rated firms để tăng signal VN-specific
  - Impute missing values theo sector median VN
  - Thêm sector-normalized features
  - Tách VN unrated ra file riêng (predict targets)
  - Validate output schema trước khi lưu

Tại sao tách ra module riêng (không để inline trong main.py):
  - Dễ unit test từng phần
  - main.py chỉ điều phối, không chứa business logic
  - DataMerger có thể được import và dùng lại trong notebook/script khác
"""

import pandas as pd
import numpy as np
import logging
from pathlib import Path
from typing import List, Optional, Tuple, Dict

from src.column_mapper import CANONICAL_FEATURES, CANONICAL_META

logger = logging.getLogger(__name__)

# ================================================================
# SECTOR MEDIANS VN (tham chiếu chéo với script.py — single source)
# ================================================================
# Đây là nguồn duy nhất. script.py import từ đây nếu cần,
# không định nghĩa lại ở nhiều nơi.
SECTOR_MEDIANS_VN: Dict[str, Dict[str, float]] = {
    'Banking': {
        'ROA': 0.015, 'ROE': 0.14,  'ROCE': 0.10,
        'Debt/Assets': 0.88,  'Debt/Equity': 12.0,
        'Current_Ratio': 1.10, 'EBIT_Margin': 0.28,
        'Asset_Turnover': 0.10, 'Net_Debt/EBITDA': 0.03,
        'WCTA': 0.05, 'RETA': 0.30, 'Market_to_Book': 1.30,
        'Revenue_Growth_YoY': 0.09, 'Log_Revenue': 9.0, 'Market_Cap': 5e10,
    },
    'RealEstate': {
        'ROA': 0.040, 'ROE': 0.09,  'ROCE': 0.07,
        'Debt/Assets': 0.65,  'Debt/Equity': 2.50,
        'Current_Ratio': 1.20, 'EBIT_Margin': 0.20,
        'Asset_Turnover': 0.28, 'Net_Debt/EBITDA': 5.50,
        'WCTA': 0.10, 'RETA': 0.15, 'Market_to_Book': 1.50,
        'Revenue_Growth_YoY': 0.11, 'Log_Revenue': 8.5, 'Market_Cap': 1e10,
    },
    'Manufacturing': {
        'ROA': 0.060, 'ROE': 0.12,  'ROCE': 0.09,
        'Debt/Assets': 0.45,  'Debt/Equity': 1.00,
        'Current_Ratio': 1.80, 'EBIT_Margin': 0.09,
        'Asset_Turnover': 1.00, 'Net_Debt/EBITDA': 2.50,
        'WCTA': 0.25, 'RETA': 0.25, 'Market_to_Book': 1.80,
        'Revenue_Growth_YoY': 0.08, 'Log_Revenue': 8.0, 'Market_Cap': 5e9,
    },
    'Technology': {
        'ROA': 0.090, 'ROE': 0.16,  'ROCE': 0.13,
        'Debt/Assets': 0.20,  'Debt/Equity': 0.30,
        'Current_Ratio': 2.50, 'EBIT_Margin': 0.18,
        'Asset_Turnover': 0.85, 'Net_Debt/EBITDA': 0.80,
        'WCTA': 0.35, 'RETA': 0.25, 'Market_to_Book': 4.00,
        'Revenue_Growth_YoY': 0.20, 'Log_Revenue': 8.0, 'Market_Cap': 8e9,
    },
    'Retail': {
        'ROA': 0.070, 'ROE': 0.15,  'ROCE': 0.11,
        'Debt/Assets': 0.40,  'Debt/Equity': 0.80,
        'Current_Ratio': 1.50, 'EBIT_Margin': 0.05,
        'Asset_Turnover': 1.80, 'Net_Debt/EBITDA': 2.00,
        'WCTA': 0.20, 'RETA': 0.22, 'Market_to_Book': 2.00,
        'Revenue_Growth_YoY': 0.10, 'Log_Revenue': 8.5, 'Market_Cap': 3e9,
    },
    'Energy': {
        'ROA': 0.050, 'ROE': 0.11,  'ROCE': 0.08,
        'Debt/Assets': 0.50,  'Debt/Equity': 1.20,
        'Current_Ratio': 1.50, 'EBIT_Margin': 0.15,
        'Asset_Turnover': 0.50, 'Net_Debt/EBITDA': 3.00,
        'WCTA': 0.18, 'RETA': 0.20, 'Market_to_Book': 1.60,
        'Revenue_Growth_YoY': 0.07, 'Log_Revenue': 9.0, 'Market_Cap': 2e10,
    },
    'FinancialServices': {
        'ROA': 0.030, 'ROE': 0.12,  'ROCE': 0.09,
        'Debt/Assets': 0.70,  'Debt/Equity': 4.00,
        'Current_Ratio': 1.20, 'EBIT_Margin': 0.20,
        'Asset_Turnover': 0.15, 'Net_Debt/EBITDA': 1.50,
        'WCTA': 0.10, 'RETA': 0.18, 'Market_to_Book': 1.50,
        'Revenue_Growth_YoY': 0.10, 'Log_Revenue': 8.0, 'Market_Cap': 5e9,
    },
    'Healthcare': {
        'ROA': 0.070, 'ROE': 0.14,  'ROCE': 0.11,
        'Debt/Assets': 0.35,  'Debt/Equity': 0.70,
        'Current_Ratio': 2.00, 'EBIT_Margin': 0.12,
        'Asset_Turnover': 0.70, 'Net_Debt/EBITDA': 1.50,
        'WCTA': 0.28, 'RETA': 0.22, 'Market_to_Book': 2.50,
        'Revenue_Growth_YoY': 0.12, 'Log_Revenue': 7.5, 'Market_Cap': 2e9,
    },
    'Construction': {
        'ROA': 0.035, 'ROE': 0.10,  'ROCE': 0.07,
        'Debt/Assets': 0.60,  'Debt/Equity': 1.80,
        'Current_Ratio': 1.30, 'EBIT_Margin': 0.07,
        'Asset_Turnover': 0.60, 'Net_Debt/EBITDA': 4.00,
        'WCTA': 0.12, 'RETA': 0.12, 'Market_to_Book': 1.30,
        'Revenue_Growth_YoY': 0.06, 'Log_Revenue': 7.8, 'Market_Cap': 1e9,
    },
    'Agriculture': {
        'ROA': 0.045, 'ROE': 0.10,  'ROCE': 0.08,
        'Debt/Assets': 0.40,  'Debt/Equity': 0.90,
        'Current_Ratio': 1.60, 'EBIT_Margin': 0.08,
        'Asset_Turnover': 0.90, 'Net_Debt/EBITDA': 2.50,
        'WCTA': 0.22, 'RETA': 0.18, 'Market_to_Book': 1.40,
        'Revenue_Growth_YoY': 0.07, 'Log_Revenue': 7.5, 'Market_Cap': 1e9,
    },
    'Transportation': {
        'ROA': 0.040, 'ROE': 0.10,  'ROCE': 0.07,
        'Debt/Assets': 0.55,  'Debt/Equity': 1.50,
        'Current_Ratio': 1.20, 'EBIT_Margin': 0.10,
        'Asset_Turnover': 0.60, 'Net_Debt/EBITDA': 3.50,
        'WCTA': 0.08, 'RETA': 0.15, 'Market_to_Book': 1.40,
        'Revenue_Growth_YoY': 0.08, 'Log_Revenue': 8.0, 'Market_Cap': 2e9,
    },
    'Telecom': {
        'ROA': 0.055, 'ROE': 0.13,  'ROCE': 0.10,
        'Debt/Assets': 0.45,  'Debt/Equity': 1.10,
        'Current_Ratio': 1.30, 'EBIT_Margin': 0.18,
        'Asset_Turnover': 0.55, 'Net_Debt/EBITDA': 2.00,
        'WCTA': 0.10, 'RETA': 0.20, 'Market_to_Book': 2.00,
        'Revenue_Growth_YoY': 0.05, 'Log_Revenue': 9.0, 'Market_Cap': 5e10,
    },
}
_DEFAULT_SECTOR_MEDIAN = SECTOR_MEDIANS_VN['Manufacturing']


# ================================================================
# DATA MERGER
# ================================================================

class DataMerger:
    """
    Gộp nhiều datasets thành một training set chuẩn hóa.

    Có thể dùng standalone hoặc được gọi từ main.py::cmd_merge().

    Ví dụ sử dụng trực tiếp:
        merger = DataMerger(vn_weight=3.0)
        df_train, df_predict = merger.merge(
            kaggle_path='data/processed/kaggle_mapped.csv',
            vn_path='data/processed/vn_firms_crawled.csv',
        )
    """

    def __init__(self, vn_weight: float = 3.0):
        """
        Args:
            vn_weight: Hệ số upweight cho VN rated firms.
                       3.0 = mỗi VN firm được đếm 3 lần trong training.
        """
        self.vn_weight = vn_weight

    # ------------------------------------------------------------------ #
    # PUBLIC API                                                           #
    # ------------------------------------------------------------------ #

    def merge(
        self,
        kaggle_path: Optional[str] = None,
        vn_path: Optional[str] = None,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Gộp datasets và trả về (df_train, df_predict).

        df_train  : Tất cả records có Credit_Rating → dùng để train
        df_predict: VN firms chưa có rating → dùng để predict

        Args:
            kaggle_path: Path đến Kaggle CSV đã map (có Credit_Rating)
            vn_path:     Path đến VN crawled CSV (mix rated + unrated)
        """
        logger.info("=" * 60)
        logger.info("DATA MERGER")
        logger.info("=" * 60)

        dfs_train  = []
        df_predict = pd.DataFrame()

        # ---- Load Kaggle ----
        if kaggle_path and Path(kaggle_path).exists():
            df_kg = self._load_kaggle(kaggle_path)
            dfs_train.append(df_kg)
        elif kaggle_path:
            logger.warning(f"Kaggle file không tồn tại: {kaggle_path}")

        # ---- Load VN crawled ----
        if vn_path and Path(vn_path).exists():
            df_rated, df_unrated = self._split_vn(vn_path)
            if not df_rated.empty:
                dfs_train.append(self._upweight_vn(df_rated))
            if not df_unrated.empty:
                df_predict = df_unrated
        elif vn_path:
            logger.warning(f"VN file không tồn tại: {vn_path}")

        # ---- Validate có data để merge ----
        if not dfs_train:
            raise ValueError(
                "Không có dữ liệu để merge. "
                "Cần ít nhất một trong: kaggle_path, vn_path (có rated firms)."
            )

        # ---- Merge ----
        df_train = pd.concat(dfs_train, ignore_index=True)
        logger.info(f"Tổng records trước post-processing: {len(df_train)}")

        # ---- Post-processing ----
        df_train   = self.impute_missing(df_train)
        df_train   = self.add_sector_features(df_train)
        df_predict = self.impute_missing(df_predict) if not df_predict.empty else df_predict

        # ---- Log summary ----
        self._log_summary(df_train, df_predict)

        return df_train, df_predict

    def impute_missing(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Impute missing values theo thứ tự ưu tiên:
          1. Sector median VN (nếu có trong SECTOR_MEDIANS_VN)
          2. Global median của cột trong df
          3. DEFAULT_SECTOR_MEDIAN (Manufacturing)
        """
        if df.empty:
            return df

        df = df.copy()
        for feat in CANONICAL_FEATURES:
            if feat not in df.columns:
                df[feat] = np.nan

            if not df[feat].isna().any():
                continue

            # Bước 1: sector median VN
            if 'Sector' in df.columns:
                for sector, medians in SECTOR_MEDIANS_VN.items():
                    if feat not in medians:
                        continue
                    mask = (df['Sector'] == sector) & df[feat].isna()
                    if mask.any():
                        df.loc[mask, feat] = medians[feat]

            # Bước 2: global median
            if df[feat].isna().any():
                global_med = df[feat].median()
                if pd.isna(global_med):
                    global_med = _DEFAULT_SECTOR_MEDIAN.get(feat, 0.0)
                df[feat] = df[feat].fillna(global_med)

        return df

    def add_sector_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Thêm features đã normalize theo sector median VN.
        Tên format: {feature}_SectorRatio = value / sector_median

        Ý nghĩa: ROA_SectorRatio = 1.5 → ROA tốt hơn median ngành 50%
        Giúp model nhận biết "tốt/xấu trong bối cảnh ngành" thay vì
        chỉ so sánh absolute value.
        """
        if df.empty:
            return df

        df = df.copy()
        ratio_features = [
            'ROA', 'ROE', 'EBIT_Margin', 'Current_Ratio',
            'Debt/Assets', 'Asset_Turnover', 'Net_Debt/EBITDA',
        ]

        for feat in ratio_features:
            if feat not in df.columns:
                continue

            col_name = f'{feat}_SectorRatio'
            df[col_name] = 1.0  # default = at median

            if 'Sector' not in df.columns:
                continue

            for sector, medians in SECTOR_MEDIANS_VN.items():
                if feat not in medians or abs(medians[feat]) < 1e-9:
                    continue
                mask = df['Sector'] == sector
                if mask.any():
                    df.loc[mask, col_name] = (
                        df.loc[mask, feat] / medians[feat]
                    ).clip(0.0, 10.0)

            df[col_name] = df[col_name].fillna(1.0)

        return df

    def validate_schema(self, df: pd.DataFrame, label: str = '') -> bool:
        """
        Kiểm tra DataFrame có đủ canonical features không.
        Trả về True nếu pass, False nếu có vấn đề nghiêm trọng.
        """
        prefix = f"[{label}] " if label else ""
        missing = [c for c in CANONICAL_FEATURES if c not in df.columns]
        null_pct = {
            c: round(df[c].isna().mean() * 100, 1)
            for c in CANONICAL_FEATURES
            if c in df.columns and df[c].isna().any()
        }

        if missing:
            logger.error(f"{prefix}Missing canonical features: {missing}")
        if null_pct:
            logger.warning(f"{prefix}Null percentages: {null_pct}")

        is_valid = len(missing) == 0
        if is_valid:
            logger.info(f"{prefix}Schema validation PASSED ({len(df)} records)")
        else:
            logger.error(f"{prefix}Schema validation FAILED")
        return is_valid

    # ------------------------------------------------------------------ #
    # PRIVATE HELPERS                                                      #
    # ------------------------------------------------------------------ #

    def _load_kaggle(self, path: str) -> pd.DataFrame:
        """Load Kaggle mapped CSV và gắn data_source tag."""
        df = pd.read_csv(path)
        df['data_source'] = 'kaggle'

        # Chỉ lấy rows có rating hợp lệ
        from src.column_mapper import RATING_NORMALIZE_MAP, VALID_RATINGS
        if 'Credit_Rating' in df.columns:
            df = df[df['Credit_Rating'].isin(VALID_RATINGS)]

        logger.info(f"Kaggle: {len(df)} rated records loaded từ {path}")
        return df

    def _split_vn(self, path: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Load VN crawled data và tách:
          - Rated: có Has_Rating=True và Credit_Rating hợp lệ
          - Unrated: còn lại (predict targets)
        """
        df = pd.read_csv(path)
        df['data_source'] = 'vietnam'

        from src.column_mapper import VALID_RATINGS

        if 'Has_Rating' in df.columns:
            rated_mask = (
                (df['Has_Rating'] == True) &
                df['Credit_Rating'].notna() &
                df['Credit_Rating'].isin(VALID_RATINGS)
            )
        else:
            rated_mask = (
                df['Credit_Rating'].notna() &
                df['Credit_Rating'].isin(VALID_RATINGS)
            ) if 'Credit_Rating' in df.columns else pd.Series(False, index=df.index)

        df_rated   = df[rated_mask].copy()
        df_unrated = df[~rated_mask].copy()

        logger.info(f"VN rated: {len(df_rated)} firms")
        logger.info(f"VN unrated (predict targets): {len(df_unrated)} firms")
        return df_rated, df_unrated

    def _upweight_vn(self, df_rated: pd.DataFrame) -> pd.DataFrame:
        """
        Upweight VN rated firms bằng cách duplicate.

        Lý do duplicate thay vì dùng sample_weight:
        - sample_weight ảnh hưởng model fit nhưng không ảnh hưởng CV split
        - Duplicate đảm bảo VN signal xuất hiện nhiều hơn ở cả train lẫn validation
        - Số lần duplicate = int(vn_weight), phần lẻ dùng random sampling
        """
        n_full  = int(self.vn_weight)        # số lần duplicate đầy đủ
        frac    = self.vn_weight - n_full    # phần lẻ

        parts = [df_rated] * n_full
        if frac > 0:
            n_extra = int(len(df_rated) * frac)
            parts.append(df_rated.sample(n=n_extra, random_state=42))

        df_up = pd.concat(parts, ignore_index=True)
        logger.info(f"VN rated upweighted {self.vn_weight}x: "
                    f"{len(df_rated)} → {len(df_up)} records")
        return df_up

    def _log_summary(self, df_train: pd.DataFrame, df_predict: pd.DataFrame):
        logger.info("\n" + "=" * 60)
        logger.info("MERGE SUMMARY")
        logger.info("=" * 60)
        logger.info(f"Training set:   {len(df_train)} records, {df_train.shape[1]} cols")
        logger.info(f"Predict set:    {len(df_predict)} records")

        if 'data_source' in df_train.columns:
            logger.info(f"\nData source breakdown:\n{df_train['data_source'].value_counts()}")
        if 'Credit_Rating' in df_train.columns:
            logger.info(f"\nRating distribution (train):\n{df_train['Credit_Rating'].value_counts()}")
        if 'Sector' in df_train.columns:
            logger.info(f"\nSector distribution (train):\n{df_train['Sector'].value_counts()}")


# ================================================================
# CONVENIENCE FUNCTION (dùng cho main.py)
# ================================================================

def run_merge(
    kaggle_path: Optional[str],
    vn_path: Optional[str],
    train_output: str,
    predict_output: str,
    vn_weight: float = 3.0,
) -> Tuple[bool, int, int]:
    """
    Wrapper cho main.py::cmd_merge().

    Returns:
        (success, n_train, n_predict)
    """
    try:
        merger = DataMerger(vn_weight=vn_weight)
        df_train, df_predict = merger.merge(kaggle_path, vn_path)

        # Validate
        merger.validate_schema(df_train, label='training')

        # Save
        Path(train_output).parent.mkdir(parents=True, exist_ok=True)
        df_train.to_csv(train_output, index=False, encoding='utf-8-sig')
        logger.info(f"✓ Training data saved → {train_output}")

        if not df_predict.empty:
            Path(predict_output).parent.mkdir(parents=True, exist_ok=True)
            df_predict.to_csv(predict_output, index=False, encoding='utf-8-sig')
            logger.info(f"✓ Predict data saved → {predict_output}")

        return True, len(df_train), len(df_predict)

    except Exception as e:
        logger.error(f"Merge failed: {e}", exc_info=True)
        return False, 0, 0
