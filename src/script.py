"""
src/script.py  v7
=================
Credit Rating Engine — Model Training & Inference.

v7 Changes:
  - OrdinalClassifier: K-1 binary boundaries (Frank & Hall 2001)
  - High-grade signal features for AAA/AA discrimination
  - Financial Strength Index + InvestmentGrade_Score
  - Dynamic sector medians from training data (fallback: reference values)
  - DomainAdaptationWeighter: upweight US firms similar to VN market
  - TwoStagePipeline: Stage1 (Kaggle) → Stage2 (VN fine-tune)
  - fine_tune(): RF warm_start, XGB continue_training, GB warm_start
"""

import pandas as pd
import numpy as np
import warnings
import logging
import pickle
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
from collections import Counter

from sklearn.model_selection import (train_test_split, StratifiedKFold,
                                     RepeatedStratifiedKFold, KFold,
                                     cross_val_score)
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (accuracy_score, cohen_kappa_score,
                             classification_report, confusion_matrix,
                             mean_absolute_error, f1_score)
from sklearn.ensemble import (RandomForestClassifier, GradientBoostingClassifier,
                              StackingClassifier, VotingClassifier)
from sklearn.linear_model import LogisticRegression
from sklearn.base import BaseEstimator, ClassifierMixin
import xgboost as xgb
try:
    import lightgbm as lgb
    HAS_LIGHTGBM = True
except ImportError:
    HAS_LIGHTGBM = False
    logger_tmp = logging.getLogger(__name__)
    logger_tmp.warning("LightGBM không khả dụng — cài: pip install lightgbm")
try:
    from imblearn.over_sampling import SMOTE, SMOTENC
    from imblearn.combine import SMOTETomek
    HAS_IMBALANCED = True
except ImportError:
    HAS_IMBALANCED = False
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings('ignore')
logger = logging.getLogger(__name__)

# ================================================================
# CONSTANTS
# ================================================================

MIN_CLASS_SIZE = 3          # class có ít hơn N samples → merge với adjacent class
MIN_CV_CLASS   = 5          # class cần ít nhất N samples để dùng StratifiedKFold

# ================================================================
# CONFIGURATION
# ================================================================

@dataclass
class Config:
    # Rating scale — thứ tự từ tốt nhất đến xấu nhất
    RATING_SCALE: List[str] = field(default_factory=lambda: [
        'AAA', 'AA', 'A', 'BBB', 'BB', 'B', 'CCC', 'CC', 'C', 'D'
    ])

    # [v8] Extended 20-notch scale — dùng cho fine-tuning VN data
    # Giữ granularity AA+/AA/AA- thay vì gộp phẳng thành AA
    # Set True nếu VN data có đủ mẫu mỗi notch (>= MIN_CLASS_SIZE)
    USE_EXTENDED_RATING_SCALE: bool = False
    RATING_SCALE_EXTENDED: List[str] = field(default_factory=lambda: [
        'AAA', 'AA+', 'AA', 'AA-', 'A+', 'A', 'A-',
        'BBB+', 'BBB', 'BBB-', 'BB+', 'BB', 'BB-',
        'B+', 'B', 'B-', 'CCC', 'CC', 'C', 'D'
    ])

    # Features
    UNIVERSAL_FEATURES: List[str] = field(default_factory=lambda: [
        'ROA', 'ROE', 'ROCE', 'EBIT_Margin',
        'Current_Ratio', 'Revenue_Growth_YoY', 'Log_Revenue', 'Market_Cap'
    ])
    SECTOR_DEPENDENT: List[str] = field(default_factory=lambda: [
        'Debt/Assets', 'Debt/Equity', 'Net_Debt/EBITDA',
        'Asset_Turnover', 'WCTA', 'RETA', 'Market_to_Book'
    ])

    # Training
    RANDOM_STATE: int = 42
    TEST_SIZE: float = 0.2
    CV_FOLDS: int = 5
    CV_REPEATS: int = 3          # [NEW] số lần repeat CV để giảm variance
    VN_SAMPLE_WEIGHT: float = 3.0

    # ── Improvement v8 ──────────────────────────────────────────────────────
    USE_SMOTE: bool = True               # SMOTE oversampling cho minority classes
    SMOTE_MIN_SAMPLES: int = 15          # Chỉ SMOTE khi training set có >= N samples
    USE_STACKING: bool = True            # Stacking ensemble
    MODEL_SELECT_METRIC: str = 'kappa'   # 'kappa' hoặc 'accuracy' — dùng kappa cho ordinal
    LOO_CV_THRESHOLD: int = 50           # Dùng LOO-CV khi VN data < N samples
    LGBM_N_ESTIMATORS: int = 300
    LGBM_MAX_DEPTH: int = 7
    LGBM_LEARNING_RATE: float = 0.05
    LGBM_NUM_LEAVES: int = 63
    FINETUNE_N_ESTIMATORS_RF: int = 100
    FINETUNE_N_ROUNDS_XGB: int = 50
    FINETUNE_N_ESTIMATORS_GB: int = 50
    FINETUNE_LR_FACTOR: float = 0.3
    MIN_VN_SAMPLES_FOR_FINETUNE: int = 10

    # Random Forest
    N_ESTIMATORS_RF: int = 300
    MAX_DEPTH_RF: int = 10
    MIN_SAMPLES_SPLIT_RF: int = 5

    # XGBoost
    N_ESTIMATORS_XGB: int = 300
    MAX_DEPTH_XGB: int = 8
    LEARNING_RATE_XGB: float = 0.03
    SUBSAMPLE_XGB: float = 0.8
    COLSAMPLE_XGB: float = 0.8

    # Gradient Boosting
    N_ESTIMATORS_GB: int = 200
    MAX_DEPTH_GB: int = 6
    LEARNING_RATE_GB: float = 0.05

    # Paths
    OUTPUT_DIR: Path = field(default_factory=lambda: Path('data/output'))
    MODEL_DIR:  Path = field(default_factory=lambda: Path('models'))
    PLOTS_DIR:  Path = field(default_factory=lambda: Path('plots'))

    def __post_init__(self):
        # [FIX #5] Chỉ set default khi field là None (tránh overwrite yaml values)
        if self.RATING_SCALE is None:
            self.RATING_SCALE = ['AAA', 'AA', 'A', 'BBB', 'BB', 'B', 'CCC', 'CC', 'C', 'D']
        if self.UNIVERSAL_FEATURES is None:
            self.UNIVERSAL_FEATURES = [
                'ROA', 'ROE', 'ROCE', 'EBIT_Margin',
                'Current_Ratio', 'Revenue_Growth_YoY', 'Log_Revenue', 'Market_Cap'
            ]
        if self.SECTOR_DEPENDENT is None:
            self.SECTOR_DEPENDENT = [
                'Debt/Assets', 'Debt/Equity', 'Net_Debt/EBITDA',
                'Asset_Turnover', 'WCTA', 'RETA', 'Market_to_Book'
            ]
        for d in [self.OUTPUT_DIR, self.MODEL_DIR, self.PLOTS_DIR]:
            Path(d).mkdir(parents=True, exist_ok=True)

    @property
    def all_features(self) -> List[str]:
        return self.UNIVERSAL_FEATURES + self.SECTOR_DEPENDENT


# ================================================================
# FEATURE ENGINEERING
# ================================================================

SECTOR_MEDIANS_REFERENCE = {
    # Nguồn: NHNN 2023, FiinGroup 2023, GSO 2022, public filings HOSE/HNX
    'Banking':          {'ROA': 0.014, 'ROE': 0.140, 'ROCE': 0.095,
                         'Debt/Assets': 0.880, 'Debt/Equity': 11.5,
                         'Current_Ratio': 1.05, 'EBIT_Margin': 0.275,
                         'Asset_Turnover': 0.09, 'Net_Debt/EBITDA': 0.05},
    'RealEstate':       {'ROA': 0.038, 'ROE': 0.088, 'ROCE': 0.068,
                         'Debt/Assets': 0.620, 'Debt/Equity': 2.4,
                         'Current_Ratio': 1.25, 'EBIT_Margin': 0.195,
                         'Asset_Turnover': 0.27, 'Net_Debt/EBITDA': 5.8},
    'Manufacturing':    {'ROA': 0.058, 'ROE': 0.115, 'ROCE': 0.088,
                         'Debt/Assets': 0.440, 'Debt/Equity': 0.95,
                         'Current_Ratio': 1.75, 'EBIT_Margin': 0.088,
                         'Asset_Turnover': 1.0,  'Net_Debt/EBITDA': 2.6},
    'Technology':       {'ROA': 0.088, 'ROE': 0.158, 'ROCE': 0.128,
                         'Debt/Assets': 0.210, 'Debt/Equity': 0.32,
                         'Current_Ratio': 2.4,  'EBIT_Margin': 0.175,
                         'Asset_Turnover': 0.88, 'Net_Debt/EBITDA': 0.9},
    'Retail':           {'ROA': 0.065, 'ROE': 0.148, 'ROCE': 0.108,
                         'Debt/Assets': 0.390, 'Debt/Equity': 0.78,
                         'Current_Ratio': 1.45, 'EBIT_Margin': 0.048,
                         'Asset_Turnover': 1.75, 'Net_Debt/EBITDA': 2.1},
    'Energy':           {'ROA': 0.052, 'ROE': 0.108, 'ROCE': 0.082,
                         'Debt/Assets': 0.495, 'Debt/Equity': 1.18,
                         'Current_Ratio': 1.48, 'EBIT_Margin': 0.148,
                         'Asset_Turnover': 0.52, 'Net_Debt/EBITDA': 3.2},
    'FinancialServices':{'ROA': 0.032, 'ROE': 0.118, 'ROCE': 0.088,
                         'Debt/Assets': 0.680, 'Debt/Equity': 3.8,
                         'Current_Ratio': 1.22, 'EBIT_Margin': 0.195,
                         'Asset_Turnover': 0.16, 'Net_Debt/EBITDA': 1.6},
    'Healthcare':       {'ROA': 0.072, 'ROE': 0.138, 'ROCE': 0.108,
                         'Debt/Assets': 0.340, 'Debt/Equity': 0.68,
                         'Current_Ratio': 2.05, 'EBIT_Margin': 0.118,
                         'Asset_Turnover': 0.72, 'Net_Debt/EBITDA': 1.4},
}
SECTOR_MEDIANS_VN = SECTOR_MEDIANS_REFERENCE  # backward compat
MIN_SAMPLES_FOR_DYNAMIC_MEDIAN = 5
_DEFAULT_MEDIAN = SECTOR_MEDIANS_REFERENCE['Manufacturing']



# ================================================================
# DOMAIN ADAPTATION WEIGHTER  [v7]
# ================================================================

class DomainAdaptationWeighter:
    """Upweight US firms similar to VN market to reduce domain gap in Stage 1."""
    VN_MARKET_PROFILE = {
        'Debt/Assets': 0.52, 'EBIT_Margin': 0.14,
        'Asset_Turnover': 0.65, 'ROA': 0.055, 'ROE': 0.125,
    }
    VN_MARKET_STD = {
        'Debt/Assets': 0.25, 'EBIT_Margin': 0.12,
        'Asset_Turnover': 0.55, 'ROA': 0.05, 'ROE': 0.08,
    }
    BASE_WEIGHT = 0.5
    MAX_WEIGHT  = 2.0

    def compute_weights(self, df: pd.DataFrame) -> np.ndarray:
        weights = np.ones(len(df))
        is_us = (df.get('data_source', pd.Series('kaggle', index=df.index)) != 'vietnam')
        if not is_us.any():
            return weights
        df_us = df[is_us]
        distances = np.zeros(len(df_us))
        n_feat = 0
        for feat, vn_val in self.VN_MARKET_PROFILE.items():
            if feat not in df_us.columns:
                continue
            std = self.VN_MARKET_STD.get(feat, 1.0)
            distances += ((df_us[feat].fillna(vn_val).values - vn_val) / std) ** 2
            n_feat += 1
        if n_feat > 0:
            distances = np.sqrt(distances / n_feat)
        sim_w = self.BASE_WEIGHT + (self.MAX_WEIGHT - self.BASE_WEIGHT) * np.exp(-distances)
        us_pos = np.where(is_us.values)[0]
        weights[us_pos] = sim_w
        logger.info(f"DomainWeighter: US weight range [{sim_w.min():.2f}, {sim_w.max():.2f}] mean={sim_w.mean():.2f}")
        return weights


class FeatureEngineer:
    """
    Tạo feature set đầy đủ từ raw features.

    Pipeline:
      1. Sector Z-score normalization
      2. Sector median ratio normalization
      3. Sector one-hot dummies
      4. Interaction terms: key_features × sector_dummies
      5. Altman Z-score components
      6. Binary signal features
    """

    def __init__(self, config: Config):
        self.config = config
        self.sector_stats: Dict = {}
        self.feature_names: List[str] = []
        self._fitted = False

    def fit(self, df: pd.DataFrame) -> 'FeatureEngineer':
        # Z-score stats
        for col in self.config.SECTOR_DEPENDENT:
            if col not in df.columns:
                continue
            grp = df.groupby('Sector')[col].agg(['mean', 'std']).fillna(0)
            self.sector_stats[col] = grp.to_dict()

        # Dynamic sector medians from training data
        norm_cols = ['ROA', 'ROE', 'Debt/Assets', 'Current_Ratio',
                     'EBIT_Margin', 'Asset_Turnover', 'ROCE', 'Debt/Equity', 'Net_Debt/EBITDA']
        self.sector_medians_dynamic: Dict = {}
        sector_counts = df.groupby('Sector').size()
        for sector, count in sector_counts.items():
            sec_data = df[df['Sector'] == sector]
            sec_med: Dict = {}
            for col in norm_cols:
                if col not in df.columns:
                    continue
                vals = sec_data[col].dropna()
                if len(vals) >= MIN_SAMPLES_FOR_DYNAMIC_MEDIAN:
                    sec_med[col] = float(vals.median())
                else:
                    ref = SECTOR_MEDIANS_REFERENCE.get(sector, _DEFAULT_MEDIAN)
                    if col in ref:
                        sec_med[col] = ref[col]
            if sec_med:
                self.sector_medians_dynamic[sector] = sec_med

        self._fitted = True
        logger.info(f"FeatureEngineer fitted on {len(df)} samples | "
                    f"Dynamic medians: {len(self.sector_medians_dynamic)} sectors "
                    f"(threshold: n>={MIN_SAMPLES_FOR_DYNAMIC_MEDIAN})")
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        if not self._fitted:
            raise RuntimeError("Call fit() or fit_transform() before transform()")
        df = df.copy()
        logger.info("Starting feature engineering...")

        df = self._sector_zscore(df)
        df = self._sector_median_ratio(df)
        df = self._sector_dummies(df)
        df = self._interaction_terms(df)
        df = self._altman_components(df)
        df = self._signal_features(df)

        _exclude = {'Ticker', 'Sector', 'Credit_Rating', 'Rating_Numeric',
                    'data_source', 'Company_Name', 'Exchange', 'Has_Rating',
                    'sample_weight', 'Rating_Group'}
        self.feature_names = [
            c for c in df.columns
            if c not in _exclude
            and c not in df.select_dtypes(exclude=[np.number]).columns
            and not c.startswith('_')
        ]
        logger.info(f"Total features: {len(self.feature_names)}")
        return df

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        return self.fit(df).transform(df)

    def _sector_zscore(self, df: pd.DataFrame) -> pd.DataFrame:
        for col in self.config.SECTOR_DEPENDENT:
            if col not in df.columns:
                continue
            new_col = f'{col}_SectorZ'
            df[new_col] = 0.0
            if col in self.sector_stats:
                means = self.sector_stats[col]['mean']
                stds  = self.sector_stats[col]['std']
                for sector in df['Sector'].unique():
                    mask = df['Sector'] == sector
                    m = means.get(sector, df[col].mean())
                    s = stds.get(sector, df[col].std()) or 1e-8
                    df.loc[mask, new_col] = (df.loc[mask, col] - m) / s
            else:
                df[new_col] = df.groupby('Sector')[col].transform(
                    lambda x: (x - x.mean()) / (x.std() + 1e-8)
                )
            df[new_col] = df[new_col].fillna(0).clip(-4, 4)
        return df

    def _sector_median_ratio(self, df: pd.DataFrame) -> pd.DataFrame:
        """Priority: 1) dynamic from training data  2) reference  3) global median"""
        norm_cols = ['ROA', 'ROE', 'Debt/Assets', 'Current_Ratio',
                     'EBIT_Margin', 'Asset_Turnover']
        for col in norm_cols:
            if col not in df.columns:
                continue
            new_col = f'{col}_SectorRatio'
            df[new_col] = 1.0
            for sector in df['Sector'].unique():
                mask = df['Sector'] == sector
                if not mask.any():
                    continue
                dyn = (self.sector_medians_dynamic.get(sector, {}).get(col)
                       if hasattr(self, 'sector_medians_dynamic') else None)
                ref = SECTOR_MEDIANS_REFERENCE.get(sector, {}).get(col)
                glb = df[col].median() if col in df.columns else None
                med = dyn or ref or glb
                if med is not None and abs(med) > 1e-9:
                    df.loc[mask, new_col] = df.loc[mask, col] / med
            df[new_col] = df[new_col].clip(0, 10).fillna(1.0)
        return df

    def _sector_dummies(self, df: pd.DataFrame) -> pd.DataFrame:
        dummies = pd.get_dummies(df['Sector'], prefix='Sector', dtype=float)
        return pd.concat([df, dummies], axis=1)

    def _interaction_terms(self, df: pd.DataFrame) -> pd.DataFrame:
        key = [f for f in ['ROA', 'ROE', 'Debt/Assets',
                            'EBIT_Margin', 'Current_Ratio',
                            'Revenue_Growth_YoY']
               if f in df.columns]
        sector_cols = [c for c in df.columns if c.startswith('Sector_')]
        count = 0
        for feat in key:
            for sc in sector_cols:
                df[f'{feat}_x_{sc.replace("Sector_", "")}'] = df[feat] * df[sc]
                count += 1
        logger.info(f"  Interaction terms: {count}")
        return df

    def _altman_components(self, df: pd.DataFrame) -> pd.DataFrame:
        if 'WCTA' in df.columns:
            df['Altman_WCTA']     = 1.2 * df['WCTA'].clip(-1, 1)
        if 'RETA' in df.columns:
            df['Altman_RETA']     = 1.4 * df['RETA'].clip(-1, 1)
        if 'EBIT_Margin' in df.columns:
            df['Altman_EBIT']     = 3.3 * df['EBIT_Margin'].clip(-1, 1)
        if 'Market_to_Book' in df.columns:
            df['Altman_MB']       = 0.6 * df['Market_to_Book'].clip(0, 20)
        if 'Asset_Turnover' in df.columns:
            df['Altman_Turnover'] = 1.0 * df['Asset_Turnover'].clip(0, 10)

        z_components = [c for c in ['Altman_WCTA', 'Altman_RETA', 'Altman_EBIT',
                                    'Altman_MB', 'Altman_Turnover']
                        if c in df.columns]
        if z_components:
            df['Altman_Z'] = df[z_components].sum(axis=1)

        if 'EBIT_Margin' in df.columns and 'Debt/Assets' in df.columns:
            d = df['Debt/Assets'].clip(0.01, 0.99)
            df['Coverage_Proxy'] = df['EBIT_Margin'] / d

        if 'ROA' in df.columns and 'Revenue_Growth_YoY' in df.columns:
            df['Quality_Growth'] = df['ROA'] * df['Revenue_Growth_YoY']

        if 'Debt/Assets' in df.columns:
            df['Equity_Ratio'] = (1 - df['Debt/Assets']).clip(0, 1)

        # ── High-Grade Composite Scores (v7) ─────────────────────────────────
        # Financial Strength Index (Piotroski-inspired)
        fs_parts = []
        if 'ROA' in df.columns:
            fs_parts.append(df['ROA'].clip(-0.3, 0.3) / 0.3)
        if 'RETA' in df.columns:
            fs_parts.append(df['RETA'].clip(-0.5, 0.5) / 0.5)
        if 'EBIT_Margin' in df.columns:
            fs_parts.append(df['EBIT_Margin'].clip(-0.5, 0.5) / 0.5)
        if 'Current_Ratio' in df.columns:
            fs_parts.append(((df['Current_Ratio'] - 1) / 2).clip(-1, 1))
        if 'Debt/Assets' in df.columns:
            fs_parts.append((0.5 - df['Debt/Assets']).clip(-0.5, 0.5) / 0.5)
        if fs_parts:
            df['Financial_Strength_Index'] = (sum(fs_parts) / len(fs_parts)).clip(-1, 1)

        # Debt Safety Margin
        if 'Debt/Assets' in df.columns:
            df['Debt_Safety_Margin'] = (0.75 - df['Debt/Assets']).clip(-0.5, 0.75)

        # InvestmentGrade_Score
        ig_parts = []
        if 'Altman_Z' in df.columns:
            ig_parts.append(df['Altman_Z'].clip(0, 6) / 6)
        if 'Financial_Strength_Index' in df.columns:
            ig_parts.append((df['Financial_Strength_Index'] + 1) / 2)
        if 'Debt_Safety_Margin' in df.columns:
            ig_parts.append(df['Debt_Safety_Margin'].clip(0, 0.75) / 0.75)
        if ig_parts:
            df['InvestmentGrade_Score'] = sum(ig_parts) / len(ig_parts)

        return df

    def _signal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        # Basic signals
        if 'ROA' in df.columns:
            df['Signal_Profitable']   = (df['ROA'] > 0).astype(float)
        if 'Revenue_Growth_YoY' in df.columns:
            df['Signal_GrowthPos']    = (df['Revenue_Growth_YoY'] > 0).astype(float)
            df['Signal_HighGrowth']   = (df['Revenue_Growth_YoY'] > 0.15).astype(float)
        if 'Debt/Assets' in df.columns:
            df['Signal_LowDebt']      = (df['Debt/Assets'] < 0.5).astype(float)
            df['Signal_HighDebt']     = (df['Debt/Assets'] > 0.75).astype(float)
        if 'Current_Ratio' in df.columns:
            df['Signal_GoodLiquidity']= (df['Current_Ratio'] > 1.5).astype(float)
        if 'ROE' in df.columns:
            df['Signal_HighROE']      = (df['ROE'] > 0.15).astype(float)

        # ── High-Grade Signals — AAA/AA discriminators (v7) ──────────────────
        if 'Debt/Assets' in df.columns:
            df['Signal_VeryLowDebt']        = (df['Debt/Assets'] < 0.25).astype(float)
            df['Signal_ConservativeDebt']   = (df['Debt/Assets'] < 0.35).astype(float)
        if 'ROA' in df.columns and 'ROE' in df.columns:
            df['Signal_StrongProfitability']= ((df['ROA'] > 0.08) & (df['ROE'] > 0.15)).astype(float)
            df['Signal_EliteProfitability'] = ((df['ROA'] > 0.12) & (df['ROE'] > 0.20)).astype(float)
        if 'EBIT_Margin' in df.columns and 'Debt/Assets' in df.columns:
            d = df['Debt/Assets'].clip(0.01, 0.99)
            coverage = df['EBIT_Margin'] / d
            df['Signal_StrongCoverage']     = (coverage > 0.3).astype(float)
            df['Signal_ExcellentCoverage']  = (coverage > 0.5).astype(float)
            df['Coverage_Ratio']            = coverage.clip(0, 10)
        if 'Debt/Equity' in df.columns:
            df['Signal_LowLeverage']        = (df['Debt/Equity'] < 0.5).astype(float)
            df['Signal_VeryLowLeverage']    = (df['Debt/Equity'] < 0.2).astype(float)
        if 'Net_Debt/EBITDA' in df.columns:
            df['Signal_LowNetDebt']         = (df['Net_Debt/EBITDA'] < 1.5).astype(float)
            df['Signal_NetCash']            = (df['Net_Debt/EBITDA'] < 0).astype(float)
        # Composite high-grade score
        hg = [c for c in ['Signal_VeryLowDebt','Signal_StrongProfitability',
                           'Signal_StrongCoverage','Signal_LowLeverage','Signal_LowNetDebt']
              if c in df.columns]
        if hg:
            df['HighGrade_Score']      = df[hg].sum(axis=1)
            df['HighGrade_Score_Norm'] = df['HighGrade_Score'] / len(hg)
        if 'Current_Ratio' in df.columns:
            df['Signal_ExcellentLiquidity'] = (df['Current_Ratio'] > 2.5).astype(float)
        if 'RETA' in df.columns:
            df['Signal_HighRETA']  = (df['RETA'] > 0.3).astype(float)
            df['Signal_EliteRETA'] = (df['RETA'] > 0.5).astype(float)
        if 'Altman_Z' in df.columns:
            df['Signal_AltmanSafe']    = (df['Altman_Z'] > 3.0).astype(float)
            df['Signal_AltmanGrey']    = ((df['Altman_Z'] >= 1.8) & (df['Altman_Z'] <= 3.0)).astype(float)
            df['Signal_AltmanDistress']= (df['Altman_Z'] < 1.8).astype(float)
        return df



# ================================================================
# ORDINAL CLASSIFIER  [v6]
# ================================================================

class OrdinalClassifier(BaseEstimator, ClassifierMixin):
    """
    Frank & Hall (2001) ordinal classifier.
    Trains K-1 binary classifiers for K ordinal classes.
    AAA cannot be predicted as B — it must cross all boundaries.
    """
    def __init__(self, base_estimator=None):
        if base_estimator is None:
            base_estimator = GradientBoostingClassifier(
                n_estimators=200, max_depth=4, learning_rate=0.05,
                subsample=0.8, random_state=42)
        self.base_estimator = base_estimator
        self.classifiers_: List = []
        self.classes_: np.ndarray = np.array([])
        self.n_classes_: int = 0

    def fit(self, X, y, sample_weight=None):
        from sklearn.base import clone
        self.classes_ = np.sort(np.unique(y))
        self.n_classes_ = len(self.classes_)
        self.classifiers_ = []
        for k in range(self.n_classes_ - 1):
            threshold = self.classes_[k]
            y_bin = (y > threshold).astype(int)
            clf = clone(self.base_estimator)
            try:
                clf.fit(X, y_bin, sample_weight=sample_weight) if sample_weight is not None else clf.fit(X, y_bin)
            except TypeError:
                clf.fit(X, y_bin)
            self.classifiers_.append(clf)
        return self

    def predict_proba(self, X):
        K = self.n_classes_
        n = X.shape[0]
        p_gt = np.zeros((n, K - 1))
        for k, clf in enumerate(self.classifiers_):
            proba = clf.predict_proba(X)
            p_gt[:, k] = proba[:, 1] if proba.shape[1] == 2 else proba[:, 0]
        # Enforce monotonicity
        for k in range(K - 2):
            p_gt[:, k] = np.maximum(p_gt[:, k], p_gt[:, k + 1])
        probs = np.zeros((n, K))
        probs[:, 0] = 1.0 - p_gt[:, 0]
        for k in range(1, K - 1):
            probs[:, k] = p_gt[:, k - 1] - p_gt[:, k]
        probs[:, K - 1] = p_gt[:, K - 2]
        probs = np.clip(probs, 0, 1)
        row_sums = probs.sum(axis=1, keepdims=True)
        return probs / np.where(row_sums == 0, 1, row_sums)

    def predict(self, X):
        return self.classes_[np.argmax(self.predict_proba(X), axis=1)]

    def get_params(self, deep=True):
        return {'base_estimator': self.base_estimator}

    def set_params(self, **params):
        if 'base_estimator' in params:
            self.base_estimator = params['base_estimator']
        return self


# ================================================================
# RARE CLASS HANDLER
# ================================================================

def _merge_rare_classes(
    y: np.ndarray,
    rating_scale: List[str],
    min_count: int = MIN_CLASS_SIZE,
) -> Tuple[np.ndarray, Dict[int, int], List[str]]:
    """
    [FIX #3] Gộp các class hiếm (< min_count samples) với class adjacent.

    Strategy:
      - Duyệt từ hai đầu rating scale (AAA và D) vào giữa
      - Class hiếm → gộp với class kế cận (better hoặc worse)
      - Trả về y_mapped, mapping dict, và active_labels mới

    Lý do:
      - AAA/CC/C/D thường chỉ có 1-3 samples VN → CV fail hoặc predict wrong
      - Gộp Adjacent giữ được thứ tự ordinal của ratings
      - Sau khi gộp, LabelEncoder và XGBoost label range đều hợp lệ

    Returns:
        y_merged: array với labels đã gộp (vẫn là indices vào rating_scale)
        merge_map: {old_label_idx → new_label_idx}
        active_labels: list tên rating còn lại sau khi gộp
    """
    counts = Counter(y)
    n_classes = len(rating_scale)

    # Xây dựng merged mapping: mặc định mỗi class map về chính nó
    merge_map = {i: i for i in range(n_classes)}

    # Xác định các class nào xuất hiện trong y
    present = set(y.tolist())

    changed = True
    while changed:
        changed = False
        for idx in range(n_classes):
            if idx not in present:
                continue
            mapped_idx = merge_map[idx]
            if counts.get(mapped_idx, 0) < min_count:
                # Tìm adjacent class tốt nhất để merge vào
                # Ưu tiên merge về phía class lớn hơn (BBB > BB > B)
                best_neighbor = None
                for delta in [+1, -1]:
                    neighbor = idx + delta
                    while 0 <= neighbor < n_classes:
                        resolved = merge_map[neighbor]
                        if counts.get(resolved, 0) >= min_count:
                            best_neighbor = resolved
                            break
                        neighbor += delta
                    if best_neighbor is not None:
                        break

                if best_neighbor is not None and best_neighbor != mapped_idx:
                    old_count = counts.get(mapped_idx, 0)
                    # Chuyển tất cả samples từ mapped_idx → best_neighbor
                    for k in list(merge_map.keys()):
                        if merge_map[k] == mapped_idx:
                            merge_map[k] = best_neighbor
                    counts[best_neighbor] = counts.get(best_neighbor, 0) + old_count
                    counts[mapped_idx] = 0
                    changed = True
                    logger.warning(
                        f"  Rare class merge: '{rating_scale[idx]}' (n={old_count}) "
                        f"→ '{rating_scale[best_neighbor]}'"
                    )

    # Apply mapping
    y_merged = np.array([merge_map[v] for v in y])

    # Re-encode để labels liên tục từ 0 (bắt buộc cho XGBoost)
    unique_merged = sorted(set(y_merged.tolist()))
    reindex = {old: new for new, old in enumerate(unique_merged)}
    y_reindexed = np.array([reindex[v] for v in y_merged])

    # Map reindex lại về rating_scale names
    active_labels = [rating_scale[idx] for idx in unique_merged]

    # Rebuild final merge_map: original idx → reindexed label
    final_map = {}
    for orig_idx, merged_idx in merge_map.items():
        if orig_idx in present:
            final_map[orig_idx] = reindex[merged_idx]

    logger.info(f"Active classes after merge: {active_labels} "
                f"({len(active_labels)}/{n_classes})")

    return y_reindexed, final_map, active_labels


def _build_cv(y: np.ndarray, n_folds: int, n_repeats: int,
              random_state: int) -> Any:
    """
    [FIX #4] Chọn CV strategy phù hợp với phân bố class.

    Priority:
      1. RepeatedStratifiedKFold — nếu tất cả class >= n_folds * n_repeats
      2. StratifiedKFold — nếu tất cả class >= n_folds
      3. KFold — fallback cuối cùng
    """
    counts = Counter(y.tolist())
    min_count = min(counts.values())

    if min_count >= n_folds * n_repeats:
        logger.info(f"CV: RepeatedStratifiedKFold({n_folds} folds × {n_repeats} repeats)")
        return RepeatedStratifiedKFold(
            n_splits=n_folds, n_repeats=n_repeats, random_state=random_state
        )
    elif min_count >= n_folds:
        logger.info(f"CV: StratifiedKFold({n_folds} folds)")
        return StratifiedKFold(
            n_splits=n_folds, shuffle=True, random_state=random_state
        )
    else:
        # Reduce folds nếu cần
        safe_folds = max(2, min_count)
        logger.warning(
            f"CV: KFold({safe_folds} folds) — min class size={min_count} < {n_folds}"
        )
        return KFold(n_splits=safe_folds, shuffle=True, random_state=random_state)


# ================================================================
# DATA PROCESSOR
# ================================================================

class DataProcessor:
    """
    Load → Map cột → Clean → Impute → FeatureEngineer → Scale.
    """

    def __init__(self, config: Config):
        self.config = config
        self.feature_engineer = FeatureEngineer(config)
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        # [FIX #1] Encoder riêng cho XGBoost — đảm bảo 0-indexed integers
        self._xgb_label_map: Dict[str, int] = {}
        self._xgb_label_inv: Dict[int, str] = {}

    def load_and_map(self, filepath: str) -> pd.DataFrame:
        from src.column_mapper import ColumnMapper
        logger.info(f"Loading: {filepath}")
        df = pd.read_csv(filepath, low_memory=False)
        logger.info(f"Loaded: {len(df)} records, {df.shape[1]} columns")
        mapper = ColumnMapper()
        df = mapper.map(df)
        report = mapper.validate(df, require_rating=False)
        logger.info(f"Mapping report: {report}")
        return df

    def clean_data(self, df: pd.DataFrame, dedup_col: str = 'Ticker') -> pd.DataFrame:
        logger.info(f"Cleaning {len(df)} records...")
        df = df.copy()

        if dedup_col in df.columns:
            before = len(df)
            # [FIX BUG-4] Ưu tiên giữ bản data_source='vietnam' khi dedup
            # Trước: keep='first' ngẫu nhiên → có thể xóa VN rows đã upweight
            if 'data_source' in df.columns:
                sk = df['data_source'].map(lambda x: 0 if str(x) == 'vietnam' else 1)
                df = (df.assign(_sk=sk)
                        .sort_values([dedup_col, '_sk'])
                        .drop_duplicates(subset=[dedup_col], keep='first')
                        .drop(columns=['_sk']))
            else:
                df = df.drop_duplicates(subset=[dedup_col], keep='first')
            logger.info(f"  Dedup by {dedup_col}: {before} → {len(df)}")

        all_feat = self.config.all_features

        for col in all_feat:
            if col not in df.columns:
                df[col] = np.nan
                continue
            if df[col].isna().any():
                df[col] = df.groupby('Sector')[col].transform(
                    lambda x: x.fillna(x.median())
                )
                global_med = df[col].median()
                if pd.isna(global_med):
                    global_med = _DEFAULT_MEDIAN.get(col, 0.0)
                df[col] = df[col].fillna(global_med)

        BANKING_LEVERAGE = {'Debt/Assets', 'Debt/Equity', 'Net_Debt/EBITDA'}
        for col in all_feat:
            if col not in df.columns:
                continue
            Q1, Q3 = df[col].quantile([0.25, 0.75])
            IQR = Q3 - Q1

            if col in BANKING_LEVERAGE and 'Sector' in df.columns:
                non_bank = df['Sector'] != 'Banking'
                lo, hi = Q1 - 1.5 * IQR, Q3 + 1.5 * IQR
                df.loc[non_bank, col] = df.loc[non_bank, col].clip(lo, hi)
            else:
                lo, hi = Q1 - 1.5 * IQR, Q3 + 1.5 * IQR
                df[col] = df[col].clip(lo, hi)

        logger.info(f"Cleaned: {len(df)} records")
        return df

    def prepare_features(
        self, df: pd.DataFrame, fit: bool = True
    ) -> Tuple[np.ndarray, Optional[np.ndarray], pd.DataFrame]:
        if 'Sector' not in df.columns:
            df['Sector'] = 'Other'

        if fit:
            df_eng = self.feature_engineer.fit_transform(df)
        else:
            df_eng = self.feature_engineer.transform(df)

        feat_names = self.feature_engineer.feature_names

        if fit:
            self._train_feature_names = feat_names
            logger.info(f"Feature matrix shape: ({len(df_eng)}, {len(feat_names)})")
        else:
            if hasattr(self, '_train_feature_names'):
                train_feats = self._train_feature_names
            else:
                train_feats = feat_names
            for col in train_feats:
                if col not in df_eng.columns:
                    df_eng[col] = 0.0
            feat_names = train_feats
            logger.info(f"Feature matrix shape: ({len(df_eng)}, {len(feat_names)}) [aligned]")

        X_raw = df_eng[feat_names].fillna(0).values
        X = self.scaler.fit_transform(X_raw) if fit else self.scaler.transform(X_raw)

        y = None
        if 'Credit_Rating' in df_eng.columns and df_eng['Credit_Rating'].notna().any():
            if fit:
                valid = [r for r in df_eng['Credit_Rating']
                         if r in self.config.RATING_SCALE]
                self.label_encoder.fit(
                    sorted(set(valid),
                           key=lambda r: self.config.RATING_SCALE.index(r)
                           if r in self.config.RATING_SCALE else 99)
                )
            if df_eng['Credit_Rating'].notna().all():
                y = self.label_encoder.transform(df_eng['Credit_Rating'])

        return X, y, df_eng

    def get_sample_weights(self, df: pd.DataFrame, vn_weight: float) -> np.ndarray:
        weights = np.ones(len(df))
        if 'data_source' in df.columns:
            weights[df['data_source'] == 'vietnam'] = vn_weight
        return weights


# ================================================================
# MODEL
# ================================================================

class CreditRatingModel:
    """
    Ensemble: Random Forest + XGBoost + Gradient Boosting.

    FIXES:
      - XGBoost dùng label encoding riêng (0-indexed integers liên tục)
      - Rare class merging trước khi train/CV
      - CV strategy adaptive theo phân bố class
      - sample_weight đúng keyword cho từng model
    """

    def __init__(self, config: Config):
        self.config = config
        self.models: Dict = {}
        self.best_model = None
        self.best_model_name: str = ''
        self.cv_results: Dict = {}
        self.feature_importance: Optional[pd.DataFrame] = None
        # [FIX #1] State cho XGBoost label mapping
        self._xgb_class_map: Dict[int, int] = {}     # original label → xgb label
        self._xgb_class_inv: Dict[int, int] = {}     # xgb label → original label
        self._active_labels: List[str] = []
        self._merge_map: Dict[int, int] = {}

    def _build_models(self) -> Dict:
        c = self.config
        models = {
            'RandomForest': RandomForestClassifier(
                n_estimators=c.N_ESTIMATORS_RF,
                max_depth=c.MAX_DEPTH_RF,
                min_samples_split=c.MIN_SAMPLES_SPLIT_RF,
                class_weight='balanced',
                random_state=c.RANDOM_STATE,
                n_jobs=-1,
            ),
            'XGBoost': xgb.XGBClassifier(
                n_estimators=c.N_ESTIMATORS_XGB,
                max_depth=c.MAX_DEPTH_XGB,
                learning_rate=c.LEARNING_RATE_XGB,
                subsample=c.SUBSAMPLE_XGB,
                colsample_bytree=c.COLSAMPLE_XGB,
                min_child_weight=3,
                gamma=0.1,
                eval_metric='mlogloss',
                use_label_encoder=False,
                random_state=c.RANDOM_STATE,
                n_jobs=-1,
                verbosity=0,
            ),
            'GradientBoosting': GradientBoostingClassifier(
                n_estimators=c.N_ESTIMATORS_GB,
                max_depth=c.MAX_DEPTH_GB,
                learning_rate=c.LEARNING_RATE_GB,
                subsample=0.8,
                random_state=c.RANDOM_STATE,
            ),
            'OrdinalClassifier': OrdinalClassifier(
                base_estimator=GradientBoostingClassifier(
                    n_estimators=c.N_ESTIMATORS_GB,
                    max_depth=min(c.MAX_DEPTH_GB, 5),
                    learning_rate=c.LEARNING_RATE_GB,
                    subsample=0.8,
                    random_state=c.RANDOM_STATE,
                )
            ),
        }
        # [v8] Thêm LightGBM nếu khả dụng
        if HAS_LIGHTGBM:
            models['LightGBM'] = lgb.LGBMClassifier(
                n_estimators=getattr(c, 'LGBM_N_ESTIMATORS', 300),
                max_depth=getattr(c, 'LGBM_MAX_DEPTH', 7),
                learning_rate=getattr(c, 'LGBM_LEARNING_RATE', 0.05),
                num_leaves=getattr(c, 'LGBM_NUM_LEAVES', 63),
                class_weight='balanced',
                subsample=0.8,
                colsample_bytree=0.8,
                min_child_samples=5,
                random_state=c.RANDOM_STATE,
                n_jobs=-1,
                verbose=-1,
            )
            logger.info("LightGBM model added ✓")
        return models

    # ---- [FIX #1] XGBoost-specific label encoding ----

    def _encode_for_xgb(self, y: np.ndarray) -> Tuple[np.ndarray, Dict]:
        """
        Encode labels thành integers 0..N-1 liên tục cho XGBoost.
        XGBoost yêu cầu: labels ∈ {0, 1, ..., num_class-1}, không được có gaps.

        Ví dụ: nếu y = [0, 2, 3, 5] (class 1, 4 bị thiếu trong train split)
        → XGBoost sẽ báo lỗi "label 5 không hợp lệ với num_class=4"
        → Fix: remap {0→0, 2→1, 3→2, 5→3}
        """
        unique_labels = sorted(set(y.tolist()))
        encode_map = {old: new for new, old in enumerate(unique_labels)}
        decode_map = {new: old for old, new in encode_map.items()}
        y_encoded = np.array([encode_map[v] for v in y])
        return y_encoded, encode_map, decode_map

    def _decode_from_xgb(self, y_xgb: np.ndarray, decode_map: Dict) -> np.ndarray:
        """Giải mã XGBoost predictions về label gốc."""
        return np.array([decode_map.get(v, v) for v in y_xgb])

    # ---- [v8] SMOTE resampling ----

    def _apply_smote(
        self,
        X: np.ndarray,
        y: np.ndarray,
        sample_weight: Optional[np.ndarray],
    ) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
        """
        [v8] SMOTE oversampling để xử lý class imbalance.

        Chỉ áp dụng khi:
          - HAS_IMBALANCED = True
          - config.USE_SMOTE = True
          - Training set >= SMOTE_MIN_SAMPLES
          - Có ít nhất 1 minority class (< 20% of mean class size)

        Strategy: SMOTETomek (SMOTE + Tomek links cleaning) —
          oversample minority + clean borderline majority → tốt hơn SMOTE thuần
        """
        if not HAS_IMBALANCED:
            logger.warning("imbalanced-learn không có — bỏ qua SMOTE. Cài: pip install imbalanced-learn")
            return X, y, sample_weight

        use_smote = getattr(self.config, 'USE_SMOTE', True)
        min_samples = getattr(self.config, 'SMOTE_MIN_SAMPLES', 15)

        if not use_smote or len(X) < min_samples:
            return X, y, sample_weight

        counts = Counter(y.tolist())
        mean_size = np.mean(list(counts.values()))
        minority_classes = {cls: cnt for cls, cnt in counts.items() if cnt < mean_size * 0.5}

        if not minority_classes:
            logger.info("  SMOTE: phân bố class cân bằng, bỏ qua")
            return X, y, sample_weight

        logger.info(f"  SMOTE: {len(minority_classes)} minority classes → {minority_classes}")

        # k_neighbors phải < min class size
        min_cls_size = min(counts.values())
        k_neighbors = max(1, min(5, min_cls_size - 1))

        try:
            smote = SMOTETomek(
                smote=SMOTE(k_neighbors=k_neighbors, random_state=self.config.RANDOM_STATE),
                random_state=self.config.RANDOM_STATE,
            )
            X_res, y_res = smote.fit_resample(X, y)
            logger.info(f"  SMOTE: {len(X)} → {len(X_res)} samples")
            # sample_weight không dùng được sau SMOTE → reset về None
            return X_res, y_res, None
        except Exception as e:
            logger.warning(f"  SMOTE failed: {e} — dùng data gốc")
            return X, y, sample_weight

    # ---- Training ----

    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        feature_names: List[str],
        sample_weight: Optional[np.ndarray] = None,
        label_encoder=None,
    ) -> Dict:
        logger.info("=" * 70)
        logger.info("TRAINING MODELS  [v8: LightGBM + SMOTE + Kappa selection]")
        logger.info("=" * 70)
        self._le = label_encoder

        # [FIX #3] Merge rare classes
        logger.info("\nChecking class distribution...")
        counts = Counter(y_train.tolist())
        for i, cnt in sorted(counts.items()):
            if hasattr(self, '_le') and self._le is not None:
                try:
                    lbl = self._le.classes_[i]
                except IndexError:
                    lbl = f'idx_{i}'
            else:
                lbl = (self.config.RATING_SCALE[i]
                       if i < len(self.config.RATING_SCALE) else str(i))
            logger.info(f"  Class {lbl}: {cnt} samples")

        y_merged, self._merge_map, self._active_labels = _merge_rare_classes(
            y_train,
            rating_scale=self.config.RATING_SCALE,
            min_count=MIN_CLASS_SIZE,
        )

        # [FIX DATA LEAKAGE] SMOTE KHÔNG được chạy trước CV.
        # Thay vào đó:
        #   - CV dùng data GỐC (y_merged, không SMOTE) → CV score trung thực
        #   - Final fit dùng data SMOTE → model tốt hơn khi deploy
        # Nếu SMOTE trước CV → synthetic samples leak vào validation fold
        # → CV inflated (0.76) nhưng test thực thấp (0.28)
        logger.info("\n[DATA LEAKAGE FIX] CV dùng data gốc, SMOTE chỉ cho final fit")

        # [FIX #4] Adaptive CV trên data GỐC
        cv = _build_cv(
            y_merged,
            n_folds=self.config.CV_FOLDS,
            n_repeats=getattr(self.config, 'CV_REPEATS', 3),
            random_state=self.config.RANDOM_STATE,
        )

        # [FIX #1] Prepare XGBoost-specific encoding từ data GỐC
        y_xgb, xgb_encode_map, xgb_decode_map = self._encode_for_xgb(y_merged)
        n_xgb_classes = len(set(y_xgb.tolist()))
        logger.info(f"\nXGBoost label range: 0..{n_xgb_classes - 1} "
                    f"(num_class={n_xgb_classes})")

        model_defs = self._build_models()
        select_metric = getattr(self.config, 'MODEL_SELECT_METRIC', 'kappa')
        logger.info(f"Model selection metric: {select_metric}")

        for i, (name, model) in enumerate(model_defs.items(), 1):
            logger.info(f"\n[{i}/{len(model_defs)}] Training {name}...")

            if name == 'XGBoost':
                y_for_cv = y_xgb   # CV dùng y_xgb gốc (không SMOTE)
                model.set_params(num_class=n_xgb_classes if n_xgb_classes > 2 else None)
            else:
                y_for_cv = y_merged

            # ── CV trên data GỐC (không SMOTE) → honest estimate ──────────
            try:
                from sklearn.metrics import make_scorer
                kappa_scorer = make_scorer(cohen_kappa_score, weights='quadratic')
                acc_scores = cross_val_score(
                    model, X_train, y_for_cv,
                    cv=cv, scoring='accuracy', n_jobs=-1
                )
                kappa_scores = cross_val_score(
                    model, X_train, y_for_cv,
                    cv=cv, scoring=kappa_scorer, n_jobs=-1
                )
                mean_acc   = acc_scores.mean()
                std_acc    = acc_scores.std()
                mean_kappa = kappa_scores.mean()
                std_kappa  = kappa_scores.std()
            except Exception as e:
                logger.error(f"  {name} CV failed: {e}")
                y_pred_train = model.predict(X_train)
                mean_acc   = accuracy_score(y_for_cv, y_pred_train)
                std_acc    = 0.0
                mean_kappa = cohen_kappa_score(y_for_cv, y_pred_train, weights='quadratic')
                std_kappa  = 0.0
                logger.warning(f"  Fallback train metrics: acc={mean_acc:.4f} kappa={mean_kappa:.4f}")

            self.cv_results[name] = {
                'scores': acc_scores if 'acc_scores' in dir() else np.array([mean_acc]),
                'mean_accuracy': mean_acc,
                'std_accuracy': std_acc,
                'mean_kappa': mean_kappa,
                'std_kappa': std_kappa,
                'xgb_decode_map': xgb_decode_map if name == 'XGBoost' else None,
            }
            logger.info(f"  CV Accuracy: {mean_acc:.4f} ± {std_acc:.4f}  |  Kappa: {mean_kappa:.4f} ± {std_kappa:.4f}")

            # ── Final fit trên SMOTE data → tốt hơn cho deployment ────────
            X_fit_sm, y_fit_sm, sw_fit = self._apply_smote(X_train, y_for_cv, sample_weight)
            fit_ok = False
            if sw_fit is not None:
                try:
                    model.fit(X_fit_sm, y_fit_sm, sample_weight=sw_fit)
                    fit_ok = True
                except TypeError as e:
                    logger.warning(f"  {name} không hỗ trợ sample_weight: {e}")
            if not fit_ok:
                model.fit(X_fit_sm, y_fit_sm)

            self.models[name] = model

            self.cv_results[name] = {
                'scores': acc_scores if 'acc_scores' in dir() else np.array([mean_acc]),
                'mean_accuracy': mean_acc,
                'std_accuracy': std_acc,
                'mean_kappa': mean_kappa,
                'std_kappa': std_kappa,
                'xgb_decode_map': xgb_decode_map if name == 'XGBoost' else None,
            }
            logger.info(f"  CV Accuracy: {mean_acc:.4f} ± {std_acc:.4f}  |  Kappa: {mean_kappa:.4f} ± {std_kappa:.4f}")

        # Lưu decode map cho predict — từ data GỐC
        self._xgb_decode_map = xgb_decode_map
        self._xgb_encode_map = xgb_encode_map
        self._y_merged_classes = sorted(set(y_merged.tolist()))

        # [v8] Stacking Ensemble — CV trên data gốc, fit trên SMOTE
        use_stacking = getattr(self.config, 'USE_STACKING', True)
        if use_stacking and len(self.models) >= 2:
            try:
                logger.info("\n[v8] Building Stacking Ensemble...")
                estimators_for_stack = [
                    (n, m) for n, m in self.models.items()
                    if n != 'OrdinalClassifier'
                ]
                if len(estimators_for_stack) >= 2:
                    stacker = StackingClassifier(
                        estimators=estimators_for_stack,
                        final_estimator=LogisticRegression(
                            max_iter=1000, C=1.0,
                            class_weight='balanced',
                            random_state=self.config.RANDOM_STATE,
                        ),
                        cv=3, n_jobs=-1,
                    )
                    # CV trên data GỐC (y_xgb, không SMOTE)
                    stack_cv = _build_cv(y_xgb, n_folds=3, n_repeats=1,
                                        random_state=self.config.RANDOM_STATE)
                    from sklearn.metrics import make_scorer as _ms
                    kappa_scorer_st = _ms(cohen_kappa_score, weights='quadratic')
                    try:
                        stack_acc = cross_val_score(stacker, X_train, y_xgb,
                                                    cv=stack_cv, scoring='accuracy', n_jobs=-1).mean()
                        stack_kap = cross_val_score(stacker, X_train, y_xgb,
                                                    cv=stack_cv, scoring=kappa_scorer_st, n_jobs=-1).mean()
                    except Exception:
                        stacker.fit(X_train, y_xgb)
                        y_stk_pred = stacker.predict(X_train)
                        stack_acc = accuracy_score(y_xgb, y_stk_pred)
                        stack_kap = cohen_kappa_score(y_xgb, y_stk_pred, weights='quadratic')

                    # Final fit trên SMOTE data
                    X_st_sm, y_st_sm, _ = self._apply_smote(X_train, y_xgb, None)
                    stacker.fit(X_st_sm, y_st_sm)
                    self.models['Stacking'] = stacker

                    self.cv_results['Stacking'] = {
                        'mean_accuracy': stack_acc,
                        'std_accuracy': 0.0,
                        'mean_kappa': stack_kap,
                        'std_kappa': 0.0,
                        'xgb_decode_map': xgb_decode_map,
                    }
                    logger.info(f"  Stacking CV (honest): Accuracy={stack_acc:.4f}  Kappa={stack_kap:.4f}")
            except Exception as e:
                logger.warning(f"  Stacking failed: {e}")

        # [v8] Chọn best model theo kappa (ordinal metric) thay vì accuracy
        select_metric = getattr(self.config, 'MODEL_SELECT_METRIC', 'kappa')
        if select_metric == 'kappa':
            self.best_model_name = max(
                self.cv_results,
                key=lambda k: self.cv_results[k].get('mean_kappa', -1)
            )
            best_metric_val = self.cv_results[self.best_model_name].get('mean_kappa', 0)
            logger.info(f"\n[v8] Model selection by KAPPA (quadratic)")
        else:
            self.best_model_name = max(
                self.cv_results,
                key=lambda k: self.cv_results[k]['mean_accuracy']
            )
            best_metric_val = self.cv_results[self.best_model_name]['mean_accuracy']

        self.best_model = self.models[self.best_model_name]
        best_acc = self.cv_results[self.best_model_name]['mean_accuracy']

        logger.info(f"\n{'='*70}")
        logger.info(f"BEST MODEL: {self.best_model_name}")
        logger.info(f"  CV Accuracy: {best_acc:.4f}  |  {select_metric.capitalize()}: {best_metric_val:.4f}")
        logger.info(f"{'='*70}")

        # Summary bảng so sánh tất cả models
        logger.info("\n── Model Comparison Summary ──")
        logger.info(f"  {'Model':<22} {'Accuracy':>10} {'Kappa':>10}")
        logger.info(f"  {'-'*44}")
        for mname, mres in self.cv_results.items():
            marker = " ←" if mname == self.best_model_name else ""
            logger.info(f"  {mname:<22} {mres['mean_accuracy']:>10.4f} {mres.get('mean_kappa', 0):>10.4f}{marker}")

        # Feature importance
        if hasattr(self.best_model, 'feature_importances_'):
            self.feature_importance = pd.DataFrame({
                'feature':    feature_names,
                'importance': self.best_model.feature_importances_,
            }).sort_values('importance', ascending=False)
            logger.info(f"\nTop 20 Features:\n"
                        f"{self.feature_importance.head(20).to_string(index=False)}")

        return self.cv_results


    # ── Stage 2 Fine-tune  [v8] ──────────────────────────────────────────────

    def fine_tune(self, X_vn, y_vn, feature_names, sample_weight=None, config=None):
        """
        Fine-tune all models on VN data after Stage 1 training.

        [v8] Improvements:
          - LOO-CV evaluation khi VN data nhỏ (<= LOO_CV_THRESHOLD)
          - LightGBM fine-tuning support
          - Đánh giá bằng kappa thay vì accuracy
        """
        cfg = config or self.config
        logger.info("\n" + "=" * 70)
        logger.info("STAGE 2: FINE-TUNING ON VN DATA  [v8]")
        logger.info("=" * 70)
        logger.info(f"VN samples: {len(X_vn)}")

        min_needed = getattr(cfg, 'MIN_VN_SAMPLES_FOR_FINETUNE', 10)
        if len(X_vn) < min_needed:
            logger.warning(f"Only {len(X_vn)} VN samples (need >={min_needed}). Skipping.")
            return {}

        y_vn_merged = np.array([self._merge_map.get(v, v) for v in y_vn])
        y_vn_xgb   = np.array([self._xgb_encode_map.get(v, v) for v in y_vn_merged])
        ft_lr_factor = getattr(cfg, 'FINETUNE_LR_FACTOR', 0.3)

        # [v8] LOO-CV thay vì train/test split khi data nhỏ
        loo_threshold = getattr(cfg, 'LOO_CV_THRESHOLD', 50)
        use_loo = len(X_vn) <= loo_threshold
        if use_loo:
            logger.info(f"  [v8] LOO-CV mode (n={len(X_vn)} <= threshold={loo_threshold})")
        else:
            logger.info(f"  Standard CV mode (n={len(X_vn)})")

        results = {}

        for name, model in self.models.items():
            logger.info(f"  Fine-tuning {name}...")
            y_fm = y_vn_xgb if name in ('XGBoost', 'Stacking') else y_vn_merged
            try:
                if name == 'RandomForest':
                    n_add = getattr(cfg, 'FINETUNE_N_ESTIMATORS_RF', 100)
                    old_n = model.n_estimators
                    model.set_params(warm_start=True, n_estimators=old_n + n_add)
                    model.fit(X_vn, y_fm, sample_weight=sample_weight) if sample_weight is not None else model.fit(X_vn, y_fm)
                    logger.info(f"    RF: {old_n}→{model.n_estimators} trees")

                elif name == 'XGBoost':
                    n_rounds = getattr(cfg, 'FINETUNE_N_ROUNDS_XGB', 50)
                    ft_lr = getattr(cfg, 'LEARNING_RATE_XGB', 0.03) * ft_lr_factor
                    # [BUG FIX] Kiểm tra n_classes khớp trước khi continue training
                    # Nếu VN data có ít class hơn US model → base_score mismatch → train fresh
                    vn_n_classes = len(set(y_fm.tolist()))
                    try:
                        existing_booster = model.get_booster()
                        us_n_classes = int(existing_booster.attr('num_class') or
                                          model.n_classes_ if hasattr(model, 'n_classes_') else vn_n_classes)
                    except Exception:
                        us_n_classes = vn_n_classes

                    if vn_n_classes != us_n_classes:
                        logger.warning(f"    XGB: VN classes={vn_n_classes} ≠ US classes={us_n_classes} → train fresh (no continue)")
                        ft_model = xgb.XGBClassifier(
                            n_estimators=n_rounds * 3,  # nhiều rounds hơn vì train từ đầu
                            max_depth=getattr(cfg, 'MAX_DEPTH_XGB', 8),
                            learning_rate=ft_lr,
                            subsample=getattr(cfg, 'SUBSAMPLE_XGB', 0.8),
                            colsample_bytree=getattr(cfg, 'COLSAMPLE_XGB', 0.8),
                            eval_metric='mlogloss', use_label_encoder=False,
                            random_state=cfg.RANDOM_STATE, verbosity=0,
                        )
                        if sample_weight is not None:
                            ft_model.fit(X_vn, y_fm, sample_weight=sample_weight)
                        else:
                            ft_model.fit(X_vn, y_fm)
                    else:
                        ft_model = xgb.XGBClassifier(
                            n_estimators=n_rounds,
                            max_depth=getattr(cfg, 'MAX_DEPTH_XGB', 8),
                            learning_rate=ft_lr,
                            subsample=getattr(cfg, 'SUBSAMPLE_XGB', 0.8),
                            colsample_bytree=getattr(cfg, 'COLSAMPLE_XGB', 0.8),
                            eval_metric='mlogloss', use_label_encoder=False,
                            random_state=cfg.RANDOM_STATE, verbosity=0,
                        )
                        try:
                            if sample_weight is not None:
                                ft_model.fit(X_vn, y_fm, sample_weight=sample_weight,
                                             xgb_model=existing_booster)
                            else:
                                ft_model.fit(X_vn, y_fm, xgb_model=existing_booster)
                        except Exception as e_xgb:
                            logger.warning(f"    XGB continue-train failed ({e_xgb}) → train fresh")
                            ft_model = xgb.XGBClassifier(
                                n_estimators=n_rounds * 3,
                                max_depth=getattr(cfg, 'MAX_DEPTH_XGB', 8),
                                learning_rate=ft_lr, eval_metric='mlogloss',
                                use_label_encoder=False, random_state=cfg.RANDOM_STATE, verbosity=0,
                            )
                            ft_model.fit(X_vn, y_fm, sample_weight=sample_weight) if sample_weight is not None else ft_model.fit(X_vn, y_fm)

                    self.models[name] = ft_model
                    model = ft_model
                    logger.info(f"    XGB fine-tune OK @ lr={ft_lr:.4f}")

                elif name == 'GradientBoosting':
                    n_add = getattr(cfg, 'FINETUNE_N_ESTIMATORS_GB', 50)
                    ft_lr = getattr(cfg, 'LEARNING_RATE_GB', 0.05) * ft_lr_factor
                    old_n = model.n_estimators
                    model.set_params(warm_start=True, n_estimators=old_n + n_add, learning_rate=ft_lr)
                    model.fit(X_vn, y_fm, sample_weight=sample_weight) if sample_weight is not None else model.fit(X_vn, y_fm)
                    logger.info(f"    GB: {old_n}→{model.n_estimators} est @ lr={ft_lr:.4f}")

                elif name == 'LightGBM' and HAS_LIGHTGBM:
                    # [v8] LightGBM fine-tune via re-fit với VN data + higher weight
                    ft_lr_lgbm = getattr(cfg, 'LGBM_LEARNING_RATE', 0.05) * ft_lr_factor
                    n_add = getattr(cfg, 'LGBM_N_ESTIMATORS', 300) // 3
                    try:
                        model.set_params(learning_rate=ft_lr_lgbm, n_estimators=n_add)
                        model.fit(X_vn, y_fm, sample_weight=sample_weight) if sample_weight is not None else model.fit(X_vn, y_fm)
                        logger.info(f"    LGBM: +{n_add} leaves @ lr={ft_lr_lgbm:.4f}")
                    except Exception as e:
                        logger.warning(f"    LGBM fine-tune fallback: {e}")
                        model.fit(X_vn, y_fm)

                elif name == 'OrdinalClassifier':
                    n_add = getattr(cfg, 'FINETUNE_N_ESTIMATORS_GB', 50)
                    ft_lr = getattr(cfg, 'LEARNING_RATE_GB', 0.05) * ft_lr_factor
                    for k, clf in enumerate(model.classifiers_):
                        threshold = model.classes_[k]
                        y_bin = (y_fm > threshold).astype(int)
                        if len(set(y_bin)) < 2:
                            continue
                        try:
                            old_n = clf.n_estimators
                            clf.set_params(warm_start=True, n_estimators=old_n + n_add, learning_rate=ft_lr)
                            clf.fit(X_vn, y_bin, sample_weight=sample_weight) if sample_weight is not None else clf.fit(X_vn, y_bin)
                        except Exception as e:
                            logger.warning(f"    OrdinalCLF[{k}] failed: {e}")

                elif name == 'Stacking':
                    # [BUG FIX] Stacking dùng y_xgb (0-indexed) — cần re-check label set
                    # Nếu VN data có classes khác với lúc train → skip, giữ model gốc
                    current_classes = set(y_fm.tolist())
                    trained_classes = set(range(len(model.classes_))) if hasattr(model, 'classes_') else current_classes
                    if not current_classes.issubset(trained_classes):
                        logger.warning(f"    Stacking skip: VN classes {current_classes} không subset của trained {trained_classes}")
                        results[name] = {'skipped': True}
                        continue
                    try:
                        if sample_weight is not None:
                            model.fit(X_vn, y_fm, sample_weight=sample_weight)
                        else:
                            model.fit(X_vn, y_fm)
                        logger.info(f"    Stacking re-fit on VN data OK")
                    except Exception as e:
                        logger.warning(f"    Stacking fine-tune skip: {e}")
                        results[name] = {'skipped': True}
                        continue

                # [v8] Đánh giá bằng LOO-CV hoặc train metrics
                # LOO-CV: dùng sklearn.base.clone thay vì type(model)(**get_params())
                # để tránh clone lỗi với StackingClassifier
                if use_loo and len(X_vn) >= 5 and name != 'Stacking':
                    from sklearn.model_selection import LeaveOneOut
                    from sklearn.base import clone as sk_clone
                    loo = LeaveOneOut()
                    y_loo_pred = []
                    n_loo_fail = 0
                    for train_idx, test_idx in loo.split(X_vn):
                        try:
                            m_clone = sk_clone(model)
                            m_clone.fit(X_vn[train_idx], y_fm[train_idx])
                            y_loo_pred.append(int(m_clone.predict(X_vn[test_idx])[0]))
                        except Exception:
                            # [BUG FIX] fallback về majority class, KHÔNG đoán đúng
                            majority = Counter(y_fm[train_idx].tolist()).most_common(1)[0][0]
                            y_loo_pred.append(majority)
                            n_loo_fail += 1
                    if n_loo_fail > 0:
                        logger.warning(f"    LOO-CV: {n_loo_fail}/{len(X_vn)} folds dùng majority fallback")
                    y_loo_arr = np.array(y_loo_pred)
                    acc = accuracy_score(y_fm, y_loo_arr)
                    kap = cohen_kappa_score(y_fm, y_loo_arr, weights='quadratic') if len(set(y_fm.tolist())) > 1 else 0.0
                    logger.info(f"    LOO-CV → Accuracy: {acc:.4f}  Kappa: {kap:.4f}")
                    results[name] = {'vn_loo_acc': acc, 'vn_loo_kappa': kap}
                else:
                    try:
                        y_pred_ft = model.predict(X_vn)
                        acc = accuracy_score(y_fm, y_pred_ft)
                        kap = cohen_kappa_score(y_fm, y_pred_ft, weights='quadratic') if len(set(y_fm.tolist())) > 1 else 0.0
                        logger.info(f"    Train acc: {acc:.4f}  kappa: {kap:.4f}")
                        results[name] = {'vn_train_acc': acc, 'vn_train_kappa': kap}
                    except Exception as e:
                        logger.warning(f"    Eval failed: {e}")
                        results[name] = {'error': str(e)}

            except Exception as e:
                logger.error(f"  Fine-tune {name} failed: {e}")
                results[name] = {'error': str(e)}

        # [BUG FIX] Sau fine-tune, nếu best_model_name là 'Stacking' nhưng Stacking
        # bị skip/lỗi → fallback về model tốt nhất còn lại (không bị lỗi)
        candidate = self.best_model_name
        stacking_ok = (
            'Stacking' not in results
            or ('skipped' not in results.get('Stacking', {}))
               and ('error' not in results.get('Stacking', {}))
        )
        if candidate == 'Stacking' and not stacking_ok:
            # Tìm model tốt nhất không bị lỗi
            fallback_candidates = {
                n: r for n, r in results.items()
                if 'error' not in r and 'skipped' not in r
            }
            if fallback_candidates:
                candidate = max(
                    fallback_candidates,
                    key=lambda k: fallback_candidates[k].get('vn_loo_kappa',
                                  fallback_candidates[k].get('vn_train_kappa', 0))
                )
                logger.warning(f"  best_model fallback: Stacking lỗi → dùng {candidate}")

        self.best_model_name = candidate
        self.best_model = self.models[self.best_model_name]
        if hasattr(self.best_model, 'feature_importances_'):
            self.feature_importance = pd.DataFrame({
                'feature': feature_names,
                'importance': self.best_model.feature_importances_,
            }).sort_values('importance', ascending=False)
        logger.info(f"\n✓ Fine-tuning complete.")
        return results

    def _predict_internal(self, X: np.ndarray, model_name: str) -> np.ndarray:
        """
        Predict và decode về label space thống nhất (merged labels).

        XGBoost: decode từ xgb_label → merged_label
        Others: predict trực tiếp (đã train trên merged labels)
        """
        model = self.models[model_name]
        y_raw = model.predict(X)

        if model_name == 'XGBoost':
            # Decode từ XGBoost 0-indexed → merged label indices
            y_decoded = self._decode_from_xgb(y_raw, self._xgb_decode_map)
            return y_decoded
        return y_raw

    def evaluate(
        self,
        X_test: np.ndarray,
        y_test: np.ndarray,
        label_encoder: LabelEncoder,
        df_test: Optional[pd.DataFrame] = None,
    ) -> Dict:
        logger.info("\n" + "=" * 70)
        logger.info("EVALUATION")
        logger.info("=" * 70)

        # Apply merge_map đến y_test (cùng mapping với training)
        y_test_merged = np.array([
            self._merge_map.get(v, v) for v in y_test
        ])

        # Predict từ best model
        y_pred_merged = self._predict_internal(X_test, self.best_model_name)
        y_pred_proba  = self.best_model.predict_proba(X_test)

        # Decode về rating names
        rating_from_merged = {
            merged_idx: self.config.RATING_SCALE[orig_idx]
            for orig_idx, merged_idx in self._merge_map.items()
        }

        y_test_labels = np.array([
            rating_from_merged.get(v, f'Unknown_{v}') for v in y_test_merged
        ])
        y_pred_labels = np.array([
            rating_from_merged.get(v, f'Unknown_{v}') for v in y_pred_merged
        ])

        # Metrics
        accuracy  = accuracy_score(y_test_merged, y_pred_merged)
        kappa     = cohen_kappa_score(y_test_merged, y_pred_merged,
                                      weights='quadratic')
        mae       = mean_absolute_error(y_test_merged, y_pred_merged)
        within_1  = np.mean(np.abs(y_test_merged - y_pred_merged) <= 1)
        within_2  = np.mean(np.abs(y_test_merged - y_pred_merged) <= 2)

        # [FIX #7] Thêm F1 scores
        f1_macro    = f1_score(y_test_merged, y_pred_merged, average='macro',
                               zero_division=0)
        f1_weighted = f1_score(y_test_merged, y_pred_merged, average='weighted',
                               zero_division=0)

        logger.info(f"Accuracy:         {accuracy:.4f}")
        logger.info(f"Kappa (quad):     {kappa:.4f}")
        logger.info(f"MAE (notches):    {mae:.4f}")
        logger.info(f"Within 1 notch:   {within_1:.4f}")
        logger.info(f"Within 2 notch:   {within_2:.4f}")
        logger.info(f"F1 (macro):       {f1_macro:.4f}")
        logger.info(f"F1 (weighted):    {f1_weighted:.4f}")

        # Classification report
        active_labels = self._active_labels
        unique_test = sorted(set(y_test_labels.tolist()))
        report = classification_report(
            y_test_labels, y_pred_labels,
            labels=[l for l in active_labels if l in unique_test],
            zero_division=0
        )
        logger.info(f"\n{report}")

        # Per-sector accuracy
        if df_test is not None and 'Sector' in df_test.columns:
            logger.info("\nPer-sector accuracy:")
            for sector in sorted(df_test['Sector'].unique()):
                mask = (df_test['Sector'] == sector).values
                if mask.sum() < 2:
                    continue
                sec_acc = accuracy_score(y_test_merged[mask], y_pred_merged[mask])
                logger.info(f"  {sector:20s}: {sec_acc:.3f}  (n={mask.sum()})")

        cm = confusion_matrix(y_test_labels, y_pred_labels,
                              labels=[l for l in active_labels if l in unique_test])

        return {
            'accuracy': accuracy,
            'kappa': kappa,
            'mae': mae,
            'within_1_notch': within_1,
            'within_2_notch': within_2,
            'f1_macro': f1_macro,
            'f1_weighted': f1_weighted,
            'y_pred': y_pred_labels,
            'y_pred_proba': y_pred_proba,
            'confusion_matrix': cm,
            'active_labels': [l for l in active_labels if l in unique_test],
            'report': report,
        }

    def plot_results(self, eval_results: Dict, save_dir: Path):
        """4-panel evaluation plot — tất cả panels luôn có nội dung."""
        fig, axes = plt.subplots(2, 2, figsize=(18, 13))
        fig.suptitle('Credit Rating Model Evaluation', fontsize=16, fontweight='bold')

        # ── 1. Confusion Matrix ────────────────────────────────────
        labels = eval_results.get('active_labels', [])
        cm     = eval_results.get('confusion_matrix', None)
        ax_cm  = axes[0, 0]
        if cm is not None and len(labels) > 0 and cm.size > 0:
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                        xticklabels=labels, yticklabels=labels, ax=ax_cm)
            ax_cm.set_title('Confusion Matrix', fontweight='bold')
            ax_cm.set_ylabel('True Label')
            ax_cm.set_xlabel('Predicted Label')
        else:
            ax_cm.text(0.5, 0.5, 'Không đủ dữ liệu\n(test set quá nhỏ)',
                       ha='center', va='center', fontsize=13,
                       transform=ax_cm.transAxes, color='gray')
            ax_cm.set_title('Confusion Matrix', fontweight='bold')
            ax_cm.axis('off')

        # ── 2. Feature Importance (với fallback cho OrdinalClassifier) ──
        ax_fi = axes[0, 1]
        fi_df = self._get_feature_importance_safe()
        if fi_df is not None and not fi_df.empty:
            top = fi_df.head(15)
            colors = ['#FF9800' if 'Sector' in f else '#2196F3'
                      for f in top['feature']]
            ax_fi.barh(range(len(top)), top['importance'], color=colors)
            ax_fi.set_yticks(range(len(top)))
            ax_fi.set_yticklabels(top['feature'], fontsize=9)
            ax_fi.set_title('Top 15 Feature Importance\n(🟦 Financial  🟧 Sector)',
                             fontweight='bold')
            ax_fi.invert_yaxis()
            ax_fi.set_xlabel('Importance')
        else:
            # Fallback: vẽ per-class precision/recall thay thế
            report_str = eval_results.get('report', '')
            y_pred = eval_results.get('y_pred', [])
            y_test = eval_results.get('y_true', [])
            if len(labels) > 0 and cm is not None and cm.size > 0:
                per_class_acc = cm.diagonal() / cm.sum(axis=1).clip(min=1)
                ax_fi.barh(labels, per_class_acc, color='#2196F3', alpha=0.8)
                ax_fi.set_xlim(0, 1)
                ax_fi.set_title('Per-Class Accuracy\n(Feature importance N/A cho model này)',
                                fontweight='bold')
                ax_fi.set_xlabel('Accuracy')
                for i, v in enumerate(per_class_acc):
                    ax_fi.text(v + 0.01, i, f'{v:.2f}', va='center', fontsize=10)
            else:
                ax_fi.text(0.5, 0.5, 'Feature importance\nkhông có cho model này\n(OrdinalClassifier)',
                           ha='center', va='center', fontsize=12,
                           transform=ax_fi.transAxes, color='gray')
                ax_fi.set_title('Feature Importance', fontweight='bold')
                ax_fi.axis('off')

        # ── 3. Model Comparison — tất cả models, dùng Kappa ──────
        ax_mc = axes[1, 0]
        # Lấy tất cả models có trong cv_results (không hardcode)
        all_model_names = list(self.cv_results.keys()) if self.cv_results else \
            ['RandomForest', 'XGBoost', 'GradientBoosting', 'OrdinalClassifier', 'LightGBM', 'Stacking']
        present = [m for m in all_model_names if m in self.cv_results and
                   self.cv_results[m].get('mean_kappa', 0) > 0]
        if not present:
            present = all_model_names[:4]

        kappas = [self.cv_results.get(m, {}).get('mean_kappa', 0.0) for m in present]
        accs   = [self.cv_results.get(m, {}).get('mean_accuracy', 0.0) for m in present]
        stds_k = [self.cv_results.get(m, {}).get('std_kappa', 0.0) for m in present]

        palette = ['#4CAF50','#2196F3','#FF5722','#9C27B0','#00BCD4','#FF9800']
        bar_colors = [palette[i % len(palette)] for i in range(len(present))]

        # Highlight best model
        best_idx = int(np.argmax(kappas)) if kappas else 0
        edge_colors = ['gold' if i == best_idx else 'none' for i in range(len(present))]
        edge_widths = [3 if i == best_idx else 0 for i in range(len(present))]

        short_names = [m.replace('GradientBoosting','GradBoost')
                        .replace('OrdinalClassifier','Ordinal')
                        .replace('RandomForest','RF')
                        .replace('LightGBM','LGBM') for m in present]

        bars = ax_mc.bar(short_names, kappas, yerr=stds_k, capsize=5,
                         color=bar_colors,
                         edgecolor=edge_colors, linewidth=edge_widths)
        ax_mc.set_ylim(0, 1)
        ax_mc.set_ylabel('Kappa (quadratic weighted)', fontsize=10)
        ax_mc.set_title('Model Comparison — CV Kappa\n(★ = best model selected)', fontweight='bold')
        for bar, k, a in zip(bars, kappas, accs):
            ax_mc.text(bar.get_x() + bar.get_width() / 2,
                       k + 0.02, f'κ={k:.3f}\nacc={a:.3f}',
                       ha='center', fontsize=8, fontweight='bold')
        # Mark best
        if kappas:
            ax_mc.text(best_idx, kappas[best_idx] + 0.08, '★',
                       ha='center', fontsize=16, color='gold')

        # ── 4. Confidence Distribution ────────────────────────────
        ax_cd = axes[1, 1]
        y_pred_proba = eval_results.get('y_pred_proba', None)
        if y_pred_proba is not None and len(y_pred_proba) > 0:
            max_probs = np.array(y_pred_proba).max(axis=1)
            n_bins = max(5, min(25, len(max_probs)))
            ax_cd.hist(max_probs, bins=n_bins, color='#9C27B0',
                       edgecolor='white', alpha=0.8)
            ax_cd.axvline(max_probs.mean(), color='red', linestyle='--',
                          linewidth=2, label=f'Mean: {max_probs.mean():.3f}')
            ax_cd.set_xlabel('Confidence Score')
            ax_cd.set_ylabel('Frequency')
            ax_cd.set_title('Prediction Confidence Distribution', fontweight='bold')
            ax_cd.legend()
        else:
            ax_cd.text(0.5, 0.5, 'Không có dữ liệu\nconfidence',
                       ha='center', va='center', fontsize=13,
                       transform=ax_cd.transAxes, color='gray')
            ax_cd.set_title('Prediction Confidence Distribution', fontweight='bold')
            ax_cd.axis('off')

        # ── Metrics box ───────────────────────────────────────────
        metrics_text = (
            f"Accuracy: {eval_results.get('accuracy', 0):.3f}\n"
            f"Kappa: {eval_results.get('kappa', 0):.3f}\n"
            f"F1 (macro): {eval_results.get('f1_macro', 0):.3f}\n"
            f"Within-1: {eval_results.get('within_1_notch', 0):.3f}\n"
            f"Within-2: {eval_results.get('within_2_notch', 0):.3f}\n"
            f"Eval: {eval_results.get('eval_method', 'holdout')}"
        )
        fig.text(0.01, 0.01, metrics_text, fontsize=9,
                 bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

        plt.tight_layout(rect=[0, 0.06, 1, 1])
        out_path = save_dir / 'evaluation_results.png'
        plt.savefig(out_path, dpi=150, bbox_inches='tight')
        logger.info(f"Plots saved: {out_path}")
        plt.close()

    def _get_feature_importance_safe(self) -> Optional[pd.DataFrame]:
        """Lấy feature importance từ best_model, hỗ trợ nhiều model types."""
        if self.feature_importance is not None and not self.feature_importance.empty:
            return self.feature_importance

        model = self.best_model
        if model is None:
            return None

        fi = None
        # Tree-based models
        if hasattr(model, 'feature_importances_'):
            fi = model.feature_importances_
        # Stacking: dùng meta-estimator coefficients
        elif hasattr(model, 'final_estimator_') and hasattr(model.final_estimator_, 'coef_'):
            # Không có feature names → trả None, caller sẽ dùng fallback
            return None
        # OrdinalClassifier: dùng base estimator nếu có
        elif hasattr(model, 'estimators_'):
            for est in (model.estimators_ if isinstance(model.estimators_, list) else []):
                if hasattr(est, 'feature_importances_'):
                    fi = est.feature_importances_
                    break

        if fi is None:
            return None

        feat_names = (self.feature_names if self.feature_names is not None
                      else [f'f{i}' for i in range(len(fi))])
        if len(fi) != len(feat_names):
            feat_names = [f'f{i}' for i in range(len(fi))]

        df_fi = pd.DataFrame({'feature': feat_names, 'importance': fi})
        return df_fi.sort_values('importance', ascending=False).reset_index(drop=True)

    def save(self, filepath: Path):
        with open(filepath, 'wb') as f:
            pickle.dump(self, f)
        logger.info(f"Model saved: {filepath}")

    @classmethod
    def load(cls, filepath: Path) -> 'CreditRatingModel':
        with open(filepath, 'rb') as f:
            model = pickle.load(f)
        logger.info(f"Model loaded: {filepath}")
        return model


# ================================================================
# PIPELINE
# ================================================================

class CreditRatingPipeline:
    """
    Orchestrate DataProcessor + CreditRatingModel.
    """

    def __init__(self, config: Config):
        self.config = config
        self.processor = DataProcessor(config)
        self.model = CreditRatingModel(config)

    def train(self, train_data_path: str) -> Dict:
        logger.info("\n" + "=" * 70)
        logger.info("STARTING TRAINING PIPELINE")
        logger.info("=" * 70)

        df = self.processor.load_and_map(train_data_path)

        if 'Credit_Rating' not in df.columns:
            raise ValueError("Training data phải có cột 'Credit_Rating'")
        df = df.dropna(subset=['Credit_Rating'])
        df = df[df['Credit_Rating'].isin(self.config.RATING_SCALE)]
        logger.info(f"Records với valid rating: {len(df)}")

        df = self.processor.clean_data(df)
        X, y, df_eng = self.processor.prepare_features(df, fit=True)
        weights = self.processor.get_sample_weights(df_eng, self.config.VN_SAMPLE_WEIGHT)

        # Train/test split — disable stratify khi có rare class
        rare = [c for c, n in Counter(y).items() if n < 2]
        use_stratify = None if rare else y
        if rare:
            logger.warning(f"Stratify disabled: {len(rare)} class(es) with 1 sample")

        X_tr, X_te, y_tr, y_te, w_tr, w_te, idx_tr, idx_te = train_test_split(
            X, y, weights, np.arange(len(df_eng)),
            test_size=self.config.TEST_SIZE,
            stratify=use_stratify,
            random_state=self.config.RANDOM_STATE,
        )
        logger.info(f"Train: {len(X_tr)}, Test: {len(X_te)}")

        self.model.train(
            X_tr, y_tr,
            self.processor.feature_engineer.feature_names,
            sample_weight=w_tr,
            label_encoder=self.processor.label_encoder,  # [FIX BUG-3]
        )

        df_test = df_eng.iloc[idx_te].reset_index(drop=True)
        eval_results = self.model.evaluate(
            X_te, y_te,
            self.processor.label_encoder,
            df_test=df_test,
        )

        self.model.plot_results(eval_results, self.config.PLOTS_DIR)
        self._save_artifacts()

        logger.info("\n✅ Training complete!")
        return eval_results

    def predict(self, data_path: str, output_path: str) -> pd.DataFrame:
        logger.info(f"\nPredicting: {data_path}")

        df = self.processor.load_and_map(data_path)
        df = self.processor.clean_data(df, dedup_col='Ticker')
        X, _, df_eng = self.processor.prepare_features(df, fit=False)

        # Predict từ best model (với decode XGBoost nếu cần)
        y_pred_merged = self.model._predict_internal(X, self.model.best_model_name)
        y_pred_proba  = self.model.best_model.predict_proba(X)

        # Decode merged labels → rating names
        rating_from_merged = {
            merged_idx: self.config.RATING_SCALE[orig_idx]
            for orig_idx, merged_idx in self.model._merge_map.items()
        }
        ratings    = np.array([rating_from_merged.get(v, 'Unknown') for v in y_pred_merged])
        confidence = y_pred_proba.max(axis=1)

        rating_order = {r: i for i, r in enumerate(self.config.RATING_SCALE)}

        df_out = df_eng.copy()
        df_out['Predicted_Rating'] = ratings
        df_out['Rating_Numeric']   = [rating_order.get(r, 99) for r in ratings]
        df_out['Confidence']       = confidence.round(4)

        # Probability columns — dùng active_labels từ model
        active = self.model._active_labels
        if y_pred_proba.shape[1] == len(active):
            for j, cls in enumerate(active):
                df_out[f'P_{cls}'] = y_pred_proba[:, j].round(4)

        df_out = df_out.sort_values('Rating_Numeric').reset_index(drop=True)

        priority_cols = ['Ticker', 'Sector', 'Predicted_Rating', 'Confidence']
        prob_cols     = [f'P_{c}' for c in active]
        feat_cols     = [c for c in self.config.all_features if c in df_out.columns]
        other_cols    = [c for c in df_out.columns
                         if c not in priority_cols + prob_cols + feat_cols
                         and not c.startswith('_')]
        out_cols = [c for c in (priority_cols + prob_cols + feat_cols + other_cols)
                    if c in df_out.columns]

        df_save = df_out[out_cols]
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        df_save.to_csv(output_path, index=False, encoding='utf-8-sig')

        logger.info(f"Saved predictions: {output_path}")
        logger.info(f"\nRating distribution:\n"
                    f"{pd.Series(ratings).value_counts()}")
        return df_out

    def _save_artifacts(self):
        self.model.save(self.config.MODEL_DIR / 'best_model.pkl')
        with open(self.config.MODEL_DIR / 'processor.pkl', 'wb') as f:
            pickle.dump(self.processor, f)
        logger.info(f"Artifacts saved → {self.config.MODEL_DIR}")

    @classmethod
    def load_trained(cls, config: Config) -> 'CreditRatingPipeline':
        pipeline = cls(config)
        pipeline.model = CreditRatingModel.load(config.MODEL_DIR / 'best_model.pkl')
        with open(config.MODEL_DIR / 'processor.pkl', 'rb') as f:
            pipeline.processor = pickle.load(f)
        return pipeline


# ================================================================
# TWO-STAGE PIPELINE  [v7]
# ================================================================

class TwoStagePipeline:
    """
    Stage 1: Train on Kaggle US data (with DomainAdaptationWeighter)
    Stage 2: Fine-tune on VN rated firms
    """
    def __init__(self, config: Config):
        self.config = config
        self.processor = DataProcessor(config)
        self.model = CreditRatingModel(config)
        self._domain_weighter = DomainAdaptationWeighter()
        self._stage1_eval: Dict = {}
        self._stage2_eval: Dict = {}

    def train(self, kaggle_path: str) -> Dict:
        logger.info("\n" + "=" * 70)
        logger.info("TWO-STAGE — STAGE 1: BASE TRAINING (Kaggle/US)")
        logger.info("=" * 70)
        df = self.processor.load_and_map(kaggle_path)
        df = df.dropna(subset=['Credit_Rating'])
        df = df[df['Credit_Rating'].isin(self.config.RATING_SCALE)]
        logger.info(f"Kaggle records: {len(df)}")
        df = self.processor.clean_data(df)
        X, y, df_eng = self.processor.prepare_features(df, fit=True)
        base_w  = self.processor.get_sample_weights(df_eng, self.config.VN_SAMPLE_WEIGHT)
        domain_w = self._domain_weighter.compute_weights(df_eng)
        combined = base_w * domain_w

        rare = [c for c, n in Counter(y).items() if n < 2]
        X_tr, X_te, y_tr, y_te, w_tr, w_te, idx_tr, idx_te = train_test_split(
            X, y, combined, np.arange(len(df_eng)),
            test_size=self.config.TEST_SIZE,
            stratify=None if rare else y,
            random_state=self.config.RANDOM_STATE,
        )
        logger.info(f"Train: {len(X_tr)}, Test: {len(X_te)}")
        self.model.train(X_tr, y_tr, self.processor.feature_engineer.feature_names,
                         sample_weight=w_tr, label_encoder=self.processor.label_encoder)
        df_test = df_eng.iloc[idx_te].reset_index(drop=True)
        self._stage1_eval = self.model.evaluate(X_te, y_te,
                                                 self.processor.label_encoder, df_test=df_test)
        self._stage1_eval['source'] = 'kaggle_us'
        logger.info(f"\n✓ Stage 1 — Kaggle accuracy: {self._stage1_eval['accuracy']:.4f}")
        self._save_artifacts()
        return self._stage1_eval

    def fine_tune(self, vn_path: str, vn_crawled_path: str = None) -> Dict:
        """
        Stage 2: Fine-tune trên VN rated firms.

        [v8 FIX] vn_rated.csv chỉ có ticker + credit_rating, KHÔNG có financial features.
        Cần join với vn_crawled.csv (có BCTC đầy đủ) để fine-tune có ý nghĩa.
        Nếu không có vn_crawled_path → warning + skip (không fine-tune với data rỗng).

        [v8 FIX] VN test set: dùng LOO-CV thay vì random split khi n < 50.
        """
        logger.info("\n" + "=" * 70)
        logger.info("TWO-STAGE — STAGE 2: FINE-TUNE (VN Data)")
        logger.info("=" * 70)

        # ── Tìm vn_crawled nếu không được truyền vào ──────────────────────
        if vn_crawled_path is None:
            # Thử tự tìm từ config paths
            from pathlib import Path as _P
            candidates = [
                getattr(self.config, 'VN_CRAWLED_PATH', None),
                'data/processed/vn_firms_crawled.csv',
                'data/processed/vn_crawled.csv',
            ]
            for c in candidates:
                if c and _P(c).exists():
                    vn_crawled_path = c
                    logger.info(f"VN crawled data tự tìm thấy: {vn_crawled_path}")
                    break

        # ── Load VN data ─────────────────────────────────────────────────
        df_vn = self.processor.load_and_map(vn_path)
        df_vn = df_vn.dropna(subset=['Credit_Rating'])
        df_vn = df_vn[df_vn['Credit_Rating'].isin(self.config.RATING_SCALE)]
        df_vn['data_source'] = 'vietnam'
        logger.info(f"VN rated firms (ratings only): {len(df_vn)}")

        # ── Join với BCTC crawled nếu có ─────────────────────────────────
        has_financial_features = any(
            c in df_vn.columns for c in ['ROA', 'ROE', 'EBIT_Margin', 'Debt/Assets']
        )

        if not has_financial_features:
            if vn_crawled_path:
                try:
                    df_crawled = self.processor.load_and_map(vn_crawled_path)
                    logger.info(f"VN crawled BCTC: {len(df_crawled)} rows")

                    # Join theo Ticker, lấy financial features từ crawled
                    df_vn = df_vn[['Ticker', 'Credit_Rating', 'data_source']].copy()
                    feat_cols = [c for c in df_crawled.columns
                                 if c not in ['Credit_Rating', 'data_source']]
                    df_merged = df_vn.merge(
                        df_crawled[feat_cols], on='Ticker', how='inner'
                    )
                    if len(df_merged) == 0:
                        logger.error("Join Ticker không match → không có VN data để fine-tune")
                        return {}
                    df_vn = df_merged
                    logger.info(f"Sau join: {len(df_vn)} VN firms có đầy đủ BCTC + rating")
                except Exception as e:
                    logger.error(f"Không load được vn_crawled: {e}")
                    logger.warning("Fine-tune với features toàn NaN → kết quả không đáng tin")
            else:
                logger.warning(
                    "\n" + "!"*70 +
                    "\n  VN rated file chỉ có ticker + rating, KHÔNG có financial features."
                    "\n  Fine-tune với data rỗng sẽ cho kết quả sai (model predict all AAA)."
                    "\n  → Chạy 'crawl' trước để lấy BCTC:"
                    "\n    python main.py crawl --rated data/raw/vn_rated.csv"
                    "\n  → Sau đó: python main.py finetune --rated data/raw/vn_rated.csv"
                    "\n" + "!"*70
                )
                logger.warning("Fine-tune bị bỏ qua (không có financial features). Giữ Stage 1 model.")
                return {'skipped': True, 'reason': 'no_financial_features'}

        df_vn = self.processor.clean_data(df_vn)
        X_vn, y_vn, df_vn_eng = self.processor.prepare_features(df_vn, fit=False)
        if y_vn is None or len(y_vn) == 0:
            logger.error("No valid VN labels.")
            return {}

        # ── VN train/test split → LOO-CV nếu nhỏ ────────────────────────
        n_vn = len(y_vn)
        loo_threshold = getattr(self.config, 'LOO_CV_THRESHOLD', 50)
        if n_vn <= loo_threshold:
            logger.info(f"VN n={n_vn} ≤ {loo_threshold}: dùng LOO-CV để evaluate (không split)")
            X_vn_tr, y_vn_tr = X_vn, y_vn
            X_vn_te, y_vn_te = X_vn, y_vn
            use_loo_eval = True
        else:
            vn_test_size = min(0.2, max(0.1, 5 / max(n_vn, 1)))
            rare_vn = [c for c, n in Counter(y_vn).items() if n < 2]
            X_vn_tr, X_vn_te, y_vn_tr, y_vn_te = train_test_split(
                X_vn, y_vn, test_size=vn_test_size,
                stratify=None if rare_vn else y_vn,
                random_state=self.config.RANDOM_STATE)
            use_loo_eval = False

        vn_ft_w = getattr(self.config, 'FINETUNE_VN_WEIGHT', 5.0)
        w_vn_tr = np.full(len(X_vn_tr), vn_ft_w)
        logger.info(f"VN fine-tune weight: {vn_ft_w}x | Train: {len(X_vn_tr)} | Eval: {'LOO-CV' if use_loo_eval else len(X_vn_te)}")

        self.model.fine_tune(X_vn_tr, y_vn_tr,
                             self.processor.feature_engineer.feature_names,
                             sample_weight=w_vn_tr, config=self.config)

        # ── Evaluate ────────────────────────────────────────────────────
        logger.info("\n── Evaluation on VN Data ──")
        if use_loo_eval:
            # LOO-CV evaluation cho honest estimate khi n nhỏ
            from sklearn.model_selection import LeaveOneOut
            from sklearn.base import clone as sk_clone
            loo = LeaveOneOut()
            y_loo_pred, y_loo_true = [], []
            for tr_idx, te_idx in loo.split(X_vn):
                try:
                    m_tmp = sk_clone(self.model.best_model)
                    y_tr_loo = np.array([self.model._merge_map.get(v, v) for v in y_vn[tr_idx]])
                    m_tmp.fit(X_vn[tr_idx], y_tr_loo)
                    pred = int(m_tmp.predict(X_vn[te_idx])[0])
                except Exception:
                    pred = Counter(y_vn[tr_idx].tolist()).most_common(1)[0][0]
                y_loo_pred.append(pred)
                y_loo_true.append(int(self.model._merge_map.get(y_vn[te_idx][0], y_vn[te_idx][0])))

            y_loo_pred = np.array(y_loo_pred)
            y_loo_true = np.array(y_loo_true)
            loo_acc = accuracy_score(y_loo_true, y_loo_pred)
            loo_kap = cohen_kappa_score(y_loo_true, y_loo_pred, weights='quadratic') if len(set(y_loo_true.tolist())) > 1 else 0.0
            w1 = float(np.mean(np.abs(y_loo_true - y_loo_pred) <= 1))
            w2 = float(np.mean(np.abs(y_loo_true - y_loo_pred) <= 2))
            logger.info(f"LOO-CV Accuracy: {loo_acc:.4f}  Kappa: {loo_kap:.4f}  Within-1: {w1:.4f}  Within-2: {w2:.4f}")
            self._stage2_eval = {
                'accuracy': loo_acc, 'kappa': loo_kap,
                'within_1_notch': w1, 'within_2_notch': w2,
                'source': 'vietnam', 'eval_method': 'LOO-CV',
                'n_vn': n_vn,
            }
        else:
            self._stage2_eval = self.model.evaluate(X_vn_te, y_vn_te,
                                                     self.processor.label_encoder)
            self._stage2_eval['source'] = 'vietnam'
            self._stage2_eval['eval_method'] = 'holdout'

        self._report_domain_gap()
        self.model.plot_results(self._stage2_eval, self.config.PLOTS_DIR)
        self._save_artifacts()
        return self._stage2_eval

    def _report_domain_gap(self):
        if not self._stage1_eval or not self._stage2_eval:
            return
        acc_us, acc_vn = self._stage1_eval.get('accuracy', 0), self._stage2_eval.get('accuracy', 0)
        kap_us, kap_vn = self._stage1_eval.get('kappa', 0), self._stage2_eval.get('kappa', 0)
        w1_us,  w1_vn  = self._stage1_eval.get('within_1_notch', 0), self._stage2_eval.get('within_1_notch', 0)
        logger.info("\n" + "=" * 55)
        logger.info("DOMAIN GAP REPORT")
        logger.info(f"{'Metric':<18} {'US':>10} {'VN':>10} {'Gap':>8}")
        logger.info("-" * 48)
        logger.info(f"{'Accuracy':<18} {acc_us:>10.4f} {acc_vn:>10.4f} {abs(acc_vn-acc_us):>8.4f}")
        logger.info(f"{'Kappa':<18} {kap_us:>10.4f} {kap_vn:>10.4f} {abs(kap_vn-kap_us):>8.4f}")
        logger.info(f"{'Within-1':<18} {w1_us:>10.4f} {w1_vn:>10.4f} {abs(w1_vn-w1_us):>8.4f}")
        gap = abs(acc_vn - acc_us)
        status = "✓ Small gap (<5%)" if gap < 0.05 else ("⚠ Medium gap (5-15%)" if gap < 0.15 else "✗ Large gap (>15%)")
        logger.info(f"Status: {status}")
        logger.info("=" * 55)

    def predict(self, data_path: str, output_path: str) -> pd.DataFrame:
        from pathlib import Path as _Path
        df = self.processor.load_and_map(data_path)
        df = self.processor.clean_data(df, dedup_col='Ticker')
        X, _, df_eng = self.processor.prepare_features(df, fit=False)
        y_pred_merged = self.model._predict_internal(X, self.model.best_model_name)
        y_pred_proba  = self.model.best_model.predict_proba(X)
        rating_from_merged = {mid: self.config.RATING_SCALE[oid]
                              for oid, mid in self.model._merge_map.items()}
        ratings    = np.array([rating_from_merged.get(v, 'Unknown') for v in y_pred_merged])
        confidence = y_pred_proba.max(axis=1)
        rating_order = {r: i for i, r in enumerate(self.config.RATING_SCALE)}
        df_out = df_eng.copy()
        df_out['Predicted_Rating'] = ratings
        df_out['Rating_Numeric']   = [rating_order.get(r, 99) for r in ratings]
        df_out['Confidence']       = confidence.round(4)
        active = self.model._active_labels
        if y_pred_proba.shape[1] == len(active):
            for j, cls in enumerate(active):
                df_out[f'P_{cls}'] = y_pred_proba[:, j].round(4)
        df_out = df_out.sort_values('Rating_Numeric').reset_index(drop=True)
        priority = ['Ticker', 'Sector', 'Predicted_Rating', 'Confidence']
        prob_cols = [f'P_{c}' for c in active]
        feat_cols = [c for c in self.config.all_features if c in df_out.columns]
        other = [c for c in df_out.columns if c not in priority + prob_cols + feat_cols and not c.startswith('_')]
        out_cols = [c for c in priority + prob_cols + feat_cols + other if c in df_out.columns]
        _Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        df_out[out_cols].to_csv(output_path, index=False, encoding='utf-8-sig')
        logger.info(f"Saved: {output_path} ({len(df_out)} firms)")
        return df_out

    def _save_artifacts(self):
        self.model.save(self.config.MODEL_DIR / 'best_model.pkl')
        with open(self.config.MODEL_DIR / 'processor.pkl', 'wb') as f:
            pickle.dump(self.processor, f)
        with open(self.config.MODEL_DIR / 'domain_gap_report.pkl', 'wb') as f:
            pickle.dump({'stage1': self._stage1_eval, 'stage2': self._stage2_eval}, f)
        logger.info(f"Artifacts saved → {self.config.MODEL_DIR}")

    @classmethod
    def load_trained(cls, config: Config) -> 'TwoStagePipeline':
        pipeline = cls(config)
        pipeline.model = CreditRatingModel.load(config.MODEL_DIR / 'best_model.pkl')
        with open(config.MODEL_DIR / 'processor.pkl', 'rb') as f:
            pipeline.processor = pickle.load(f)
        try:
            with open(config.MODEL_DIR / 'domain_gap_report.pkl', 'rb') as f:
                rpt = pickle.load(f)
                pipeline._stage1_eval = rpt.get('stage1', {})
                pipeline._stage2_eval = rpt.get('stage2', {})
        except FileNotFoundError:
            pass
        return pipeline
