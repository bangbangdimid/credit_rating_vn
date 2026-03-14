"""
src/column_mapper.py
====================
Chuẩn hóa tên cột từ nhiều nguồn dữ liệu khác nhau về một schema thống nhất.

Hỗ trợ:
  - Kaggle corporate_credit_rating.csv (2029 US firms, camelCase columns)
  - vnstock output (tiếng Anh hoặc tiếng Việt, tùy version)
  - FiinRatings / VIS Rating export CSV
  - File tự nhập tay của nhóm

Sau khi map, mọi DataFrame đều có đúng các cột sau:
  - Ticker, Sector, Credit_Rating (optional cho predict)
  - ROA, ROE, ROCE, EBIT_Margin
  - Current_Ratio, Debt/Assets, Debt/Equity, Net_Debt/EBITDA
  - Asset_Turnover, WCTA, RETA, Market_to_Book
  - Revenue_Growth_YoY, Log_Revenue, Market_Cap
"""

import pandas as pd
import numpy as np
import logging
from typing import Optional

logger = logging.getLogger(__name__)

# ============================================================
# SCHEMA CHUẨN (canonical) — không thay đổi tên này
# ============================================================
CANONICAL_FEATURES = [
    # Profitability
    'ROA', 'ROE', 'ROCE', 'EBIT_Margin',
    # Liquidity
    'Current_Ratio',
    # Leverage
    'Debt/Assets', 'Debt/Equity', 'Net_Debt/EBITDA',
    # Activity
    'Asset_Turnover',
    # Quality / Altman components
    'WCTA', 'RETA', 'Market_to_Book',
    # Growth & Size
    'Revenue_Growth_YoY', 'Log_Revenue', 'Market_Cap',
]

CANONICAL_META = ['Ticker', 'Sector', 'Credit_Rating']

# ============================================================
# RATING NORMALIZER — mọi scale đều về AAA..D
# ============================================================
RATING_NORMALIZE_MAP = {
    'AAA': 'AAA',
    'AA+': 'AA', 'AA': 'AA', 'AA-': 'AA',
    'A+':  'A',  'A':  'A',  'A-':  'A',
    'BBB+': 'BBB', 'BBB': 'BBB', 'BBB-': 'BBB',
    'BB+': 'BB', 'BB': 'BB', 'BB-': 'BB',
    'B+':  'B',  'B':  'B',  'B-':  'B',
    'CCC+': 'CCC', 'CCC': 'CCC', 'CCC-': 'CCC',
    'CC': 'CC', 'C': 'C', 'D': 'D', 'SD': 'D',
    # FiinRatings tiếng Việt
    'AAA.vn': 'AAA', 'AA+.vn': 'AA', 'AA.vn': 'AA', 'AA-.vn': 'AA',
}

VALID_RATINGS = {'AAA', 'AA', 'A', 'BBB', 'BB', 'B', 'CCC', 'CC', 'C', 'D'}

# [v8] Extended 20-notch scale — giữ granularity cho VN data
# VN market chủ yếu dùng thang BB- đến AAA với +/- modifiers
RATING_SCALE_EXTENDED = [
    'AAA',
    'AA+', 'AA', 'AA-',
    'A+',  'A',  'A-',
    'BBB+','BBB','BBB-',
    'BB+', 'BB', 'BB-',
    'B+',  'B',  'B-',
    'CCC', 'CC', 'C', 'D'
]

# Map extended → 10-notch (dùng khi cần standard scale)
EXTENDED_TO_STANDARD = {
    'AAA': 'AAA',
    'AA+': 'AA', 'AA': 'AA', 'AA-': 'AA',
    'A+':  'A',  'A':  'A',  'A-':  'A',
    'BBB+':'BBB','BBB':'BBB','BBB-':'BBB',
    'BB+': 'BB', 'BB': 'BB', 'BB-': 'BB',
    'B+':  'B',  'B':  'B',  'B-':  'B',
    'CCC': 'CCC','CC': 'CC', 'C':   'C',  'D': 'D',
}

# Numeric ordinal value cho extended scale (nhỏ hơn = tốt hơn)
RATING_NUMERIC = {r: i for i, r in enumerate(RATING_SCALE_EXTENDED)}

# ============================================================
# KAGGLE COLUMN MAP
# ============================================================
# Kaggle: camelCase, đơn vị thường là ratio (0–1) hoặc raw
KAGGLE_COL_MAP = {
    # Meta
    'Rating':               'Credit_Rating',
    'Symbol':               'Ticker',
    'Sector':               'Sector',
    # Profitability — Kaggle lưu dạng ratio (0.05 = 5%), NHƯNG có outliers cực đoan
    'returnOnAssets':           'ROA',
    'returnOnEquity':           'ROE',
    'returnOnCapitalEmployed':  'ROCE',
    'operatingProfitMargin':    'EBIT_Margin',   # primary
    'ebitPerRevenue':           '_ebitPerRevenue',  # fallback nếu thiếu
    'netProfitMargin':          '_netProfitMargin',
    'grossProfitMargin':        '_grossProfitMargin',
    # Liquidity
    'currentRatio':             'Current_Ratio',
    'quickRatio':               '_quickRatio',
    'cashRatio':                '_cashRatio',
    # Leverage
    'debtRatio':                'Debt/Assets',
    'debtEquityRatio':          'Debt/Equity',
    'companyEquityMultiplier':  '_equityMultiplier',
    # Activity
    'assetTurnover':            'Asset_Turnover',
    'fixedAssetTurnover':       '_fixedAssetTurnover',
    'payablesTurnover':         '_payablesTurnover',
    'daysOfSalesOutstanding':   '_dso',
    # Cash flow
    'freeCashFlowOperatingCashFlowRatio': '_fcf_ocf_ratio',
    'operatingCashFlowSalesRatio':        '_ocf_sales_ratio',
    'freeCashFlowPerShare':               '_fcf_per_share',
    'operatingCashFlowPerShare':          '_ocf_per_share',
    'cashPerShare':                       '_cash_per_share',
    # Other
    'effectiveTaxRate':         '_tax_rate',
    'enterpriseValueMultiple':  '_ev_multiple',
    'pretaxProfitMargin':       '_pretax_margin',
}

# Sector mapping Kaggle → chuẩn
KAGGLE_SECTOR_MAP = {
    'Finance':               'FinancialServices',
    'Capital Goods':         'Manufacturing',
    'Basic Industries':      'Manufacturing',
    'Consumer Durables':     'Manufacturing',
    'Consumer Non-Durables': 'Manufacturing',
    'Consumer Services':     'Retail',
    'Energy':                'Energy',
    'Health Care':           'Healthcare',
    'Technology':            'Technology',
    'Transportation':        'Transportation',
    'Public Utilities':      'Energy',
    'Miscellaneous':         'Other',
}

# ============================================================
# VNSTOCK COLUMN MAP (lang='en')
# ============================================================
VNSTOCK_COL_MAP = {
    # Từ vn_data_adapter.py đã map → đây chỉ là fallback nếu dùng trực tiếp
    'roa': 'ROA', 'ROA': 'ROA', 'returnOnAssets': 'ROA',
    'roe': 'ROE', 'ROE': 'ROE', 'returnOnEquity': 'ROE',
    'roce': 'ROCE', 'ROCE': 'ROCE',
    'ebit_margin': 'EBIT_Margin', 'EBIT_Margin': 'EBIT_Margin',
    'current_ratio': 'Current_Ratio', 'Current_Ratio': 'Current_Ratio',
    'debt_assets': 'Debt/Assets', 'Debt/Assets': 'Debt/Assets',
    'debt_equity': 'Debt/Equity', 'Debt/Equity': 'Debt/Equity',
    'net_debt_ebitda': 'Net_Debt/EBITDA', 'Net_Debt/EBITDA': 'Net_Debt/EBITDA',
    'asset_turnover': 'Asset_Turnover', 'Asset_Turnover': 'Asset_Turnover',
    'wcta': 'WCTA', 'WCTA': 'WCTA',
    'reta': 'RETA', 'RETA': 'RETA',
    'market_to_book': 'Market_to_Book', 'Market_to_Book': 'Market_to_Book',
    'revenue_growth_yoy': 'Revenue_Growth_YoY', 'Revenue_Growth_YoY': 'Revenue_Growth_YoY',
    'log_revenue': 'Log_Revenue', 'Log_Revenue': 'Log_Revenue',
    'market_cap': 'Market_Cap', 'Market_Cap': 'Market_Cap',
    'ticker': 'Ticker', 'Ticker': 'Ticker', 'symbol': 'Ticker',
    'sector': 'Sector', 'Sector': 'Sector',
    'credit_rating': 'Credit_Rating', 'Credit_Rating': 'Credit_Rating',
}

# ============================================================
# OUTLIER THRESHOLDS cho Kaggle (dựa vào phân tích data)
# Giá trị ngoài range này cực kỳ bất thường → winsorize về clip bounds
# ============================================================
KAGGLE_CLIP_BOUNDS = {
    'ROA':                (-0.5,    0.5),
    'ROE':                (-2.0,    2.0),
    'ROCE':               (-1.0,    1.0),
    'EBIT_Margin':        (-1.0,    1.0),
    'Current_Ratio':      (0.0,     20.0),
    'Debt/Assets':        (0.0,     1.0),   # ratio, bình thường 0-1
    'Debt/Equity':        (-10.0,   30.0),
    'Asset_Turnover':     (0.0,     10.0),
    '_quickRatio':        (0.0,     20.0),
    '_fixedAssetTurnover':(0.0,     50.0),
    '_payablesTurnover':  (0.0,     100.0),
    '_dso':               (0.0,     500.0),
    '_fcf_ocf_ratio':     (-5.0,    5.0),
    '_ocf_sales_ratio':   (-2.0,    5.0),
    '_ev_multiple':       (-100.0,  200.0),
    '_tax_rate':          (-1.0,    2.0),
}

# ============================================================
# MAPPER CLASS
# ============================================================

class ColumnMapper:
    """
    Phát hiện nguồn dữ liệu và map về schema chuẩn.

    Usage:
        mapper = ColumnMapper()
        df_std = mapper.map(df)          # auto-detect
        df_std = mapper.map_kaggle(df)   # force Kaggle
        df_std = mapper.map_vnstock(df)  # force VN
    """

    def __init__(self):
        self.source_detected: Optional[str] = None

    # ------------------------------------------------------------------ #
    # PUBLIC API                                                           #
    # ------------------------------------------------------------------ #

    def map(self, df: pd.DataFrame) -> pd.DataFrame:
        """Auto-detect source và map về schema chuẩn."""
        source = self._detect_source(df)
        self.source_detected = source
        logger.info(f"Detected data source: {source}")

        if source == 'kaggle':
            return self.map_kaggle(df)
        elif source == 'vnstock':
            return self.map_vnstock(df)
        else:
            return self.map_generic(df)

    def map_kaggle(self, df: pd.DataFrame) -> pd.DataFrame:
        """Map Kaggle corporate_credit_rating dataset."""
        df = df.copy()

        # 1. Rename columns
        df = df.rename(columns={k: v for k, v in KAGGLE_COL_MAP.items() if k in df.columns})

        # 2. Deduplicate — Kaggle có nhiều rating agencies + nhiều năm per company
        #    Lấy rating mới nhất (Date desc), sau đó dedup theo Symbol
        if 'Date' in df.columns:
            df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
            df = df.sort_values('Date', ascending=False)
        df = df.drop_duplicates(subset=['Ticker'], keep='first')
        logger.info(f"After dedup by Ticker: {len(df)} records")

        # 3. Normalize Credit_Rating
        if 'Credit_Rating' in df.columns:
            df['Credit_Rating'] = df['Credit_Rating'].map(RATING_NORMALIZE_MAP)
            before = len(df)
            df = df.dropna(subset=['Credit_Rating'])
            logger.info(f"Valid ratings: {len(df)}/{before}")

        # 4. Map Sector
        if 'Sector' in df.columns:
            df['Sector'] = df['Sector'].map(KAGGLE_SECTOR_MAP).fillna('Other')

        # 5. Tạo Ticker nếu thiếu
        if 'Ticker' not in df.columns and 'Name' in df.columns:
            df['Ticker'] = df['Name'].str[:6].str.upper().str.replace(' ', '')

        # 6. Winsorize outliers (Kaggle có rất nhiều extreme values)
        df = self._winsorize(df, KAGGLE_CLIP_BOUNDS)

        # 7. Xử lý scale issues — Kaggle ROA/ROE ở dạng ratio (0.05)
        #    nhưng có outlier nên cần check sau khi winsorize
        pct_cols = ['ROA', 'ROE', 'ROCE', 'EBIT_Margin']
        for col in pct_cols:
            if col in df.columns:
                # Sau winsorize, median nên nằm trong (-1, 1) → đây là ratio, giữ nguyên
                # Không nhân 100 vì toàn bộ pipeline sẽ dùng dạng ratio
                pass

        # 8. Tính derived features còn thiếu
        df = self._derive_kaggle_features(df)

        # 9. Tạo Ticker dạng US_xxx để phân biệt với VN
        df['data_source'] = 'kaggle'

        logger.info(f"Kaggle mapped: {len(df)} records, {df.shape[1]} columns")
        logger.info(f"Rating distribution:\n{df['Credit_Rating'].value_counts()}")
        return df

    def map_vnstock(self, df: pd.DataFrame) -> pd.DataFrame:
        """Map output từ vn_data_adapter (đã gần chuẩn, chỉ cần verify)."""
        df = df.copy()
        df = df.rename(columns={k: v for k, v in VNSTOCK_COL_MAP.items() if k in df.columns})

        if 'Credit_Rating' in df.columns:
            df['Credit_Rating'] = df['Credit_Rating'].map(
                lambda x: RATING_NORMALIZE_MAP.get(str(x).strip(), x) if pd.notna(x) else x
            )

        df['data_source'] = 'vietnam'
        return df

    def map_generic(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Cố gắng map file không rõ nguồn.
        Thử case-insensitive match cho canonical feature names.
        """
        df = df.copy()
        col_lower = {c.lower().replace(' ', '_').replace('/', '_'): c for c in df.columns}

        rename_map = {}
        for canonical in CANONICAL_FEATURES + CANONICAL_META:
            key = canonical.lower().replace('/', '_')
            if key in col_lower and col_lower[key] != canonical:
                rename_map[col_lower[key]] = canonical

        df = df.rename(columns=rename_map)

        if 'Credit_Rating' in df.columns:
            df['Credit_Rating'] = df['Credit_Rating'].map(
                lambda x: RATING_NORMALIZE_MAP.get(str(x).strip(), x) if pd.notna(x) else x
            )

        df['data_source'] = 'unknown'
        logger.warning(f"Generic mapping applied. Missing canonical cols: "
                       f"{[c for c in CANONICAL_FEATURES if c not in df.columns]}")
        return df

    # ------------------------------------------------------------------ #
    # PRIVATE HELPERS                                                      #
    # ------------------------------------------------------------------ #

    def _detect_source(self, df: pd.DataFrame) -> str:
        """Phát hiện nguồn dữ liệu dựa vào tên cột đặc trưng."""
        cols = set(df.columns)

        # Kaggle: có camelCase columns đặc trưng
        kaggle_signals = {'returnOnAssets', 'returnOnEquity', 'debtRatio',
                          'currentRatio', 'assetTurnover', 'Rating Agency Name'}
        if len(kaggle_signals & cols) >= 3:
            return 'kaggle'

        # VNStock output từ adapter: có _Is_Bank, Has_Rating
        vnstock_signals = {'_Is_Bank', 'Has_Rating', '_Revenue', '_Net_Income'}
        if len(vnstock_signals & cols) >= 2:
            return 'vnstock'

        # Đã map sẵn: có canonical column names
        canonical_signals = {'ROA', 'ROE', 'EBIT_Margin', 'Current_Ratio', 'Debt/Assets'}
        if len(canonical_signals & cols) >= 4:
            return 'canonical'

        return 'generic'

    def _winsorize(self, df: pd.DataFrame, bounds: dict) -> pd.DataFrame:
        """Clip các cột theo bounds đã định nghĩa."""
        for col, (lo, hi) in bounds.items():
            if col in df.columns:
                n_clipped = ((df[col] < lo) | (df[col] > hi)).sum()
                if n_clipped > 0:
                    logger.debug(f"  Winsorize {col}: {n_clipped} values clipped to [{lo}, {hi}]")
                df[col] = df[col].clip(lower=lo, upper=hi)
        return df

    def _derive_kaggle_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Tính các features cần thiết mà Kaggle không có sẵn."""

        # EBIT_Margin: dùng ebitPerRevenue nếu operatingProfitMargin thiếu
        if 'EBIT_Margin' not in df.columns and '_ebitPerRevenue' in df.columns:
            df['EBIT_Margin'] = df['_ebitPerRevenue']
        elif 'EBIT_Margin' not in df.columns and '_netProfitMargin' in df.columns:
            df['EBIT_Margin'] = df['_netProfitMargin']

        # Net_Debt/EBITDA — Kaggle không có trực tiếp, ước tính
        # Proxy: Debt/Assets / max(EBIT_Margin, 0.01)
        if 'Net_Debt/EBITDA' not in df.columns:
            if 'Debt/Assets' in df.columns and 'EBIT_Margin' in df.columns:
                ebit_safe = df['EBIT_Margin'].clip(lower=0.01)
                # Scale: D/A = tổng nợ/tài sản; EBIT_margin = EBIT/Revenue
                # Hệ số 3 là ước tính thô dựa trên Asset Turnover median ~0.7
                df['Net_Debt/EBITDA'] = (df['Debt/Assets'] * 3) / ebit_safe
                df['Net_Debt/EBITDA'] = df['Net_Debt/EBITDA'].clip(-5, 30)
            else:
                df['Net_Debt/EBITDA'] = 3.0

        # WCTA — từ Current_Ratio và Debt/Assets
        if 'WCTA' not in df.columns and 'Current_Ratio' in df.columns:
            # WC/TA ≈ (CR - 1) / (CR + leverage_factor)
            cr = df['Current_Ratio'].clip(lower=0.1)
            df['WCTA'] = ((cr - 1) / cr * 0.3).clip(-0.5, 0.8)

        # RETA — không có retained earnings → proxy = ROE * 0.6 (payout ratio ~40%)
        if 'RETA' not in df.columns and 'ROE' in df.columns:
            df['RETA'] = (df['ROE'] * 0.6).clip(-1.0, 1.0)

        # Market_to_Book — Kaggle không có, dùng companyEquityMultiplier - 1
        if 'Market_to_Book' not in df.columns:
            if '_equityMultiplier' in df.columns:
                df['Market_to_Book'] = (df['_equityMultiplier'] - 1).clip(0, 20)
            else:
                df['Market_to_Book'] = 2.0

        # Revenue_Growth_YoY — không có trong Kaggle (single snapshot per ticker after dedup)
        if 'Revenue_Growth_YoY' not in df.columns:
            # Gán median theo sector (growth khác nhau đáng kể)
            sector_growth = {
                'Technology': 0.15, 'Healthcare': 0.08, 'Energy': 0.05,
                'Manufacturing': 0.06, 'Retail': 0.04, 'FinancialServices': 0.07,
                'Transportation': 0.05, 'Other': 0.05,
            }
            df['Revenue_Growth_YoY'] = df['Sector'].map(sector_growth).fillna(0.06)

        # Log_Revenue — Kaggle không có doanh thu tuyệt đối
        # Dùng cashPerShare và operatingCashFlowPerShare làm proxy size
        if 'Log_Revenue' not in df.columns:
            if '_cash_per_share' in df.columns:
                safe_val = df['_cash_per_share'].clip(lower=0.01)
                df['Log_Revenue'] = np.log1p(safe_val).clip(2, 14)
            else:
                # Fallback: US firms median revenue ~$5B → log(5B) ≈ 22
                df['Log_Revenue'] = 8.5

        # Market_Cap — không có, dùng enterprise value proxy
        if 'Market_Cap' not in df.columns:
            df['Market_Cap'] = 1e9  # Placeholder; không ảnh hưởng nhiều

        return df

    # ------------------------------------------------------------------ #
    # UTILITY                                                              #
    # ------------------------------------------------------------------ #

    def validate(self, df: pd.DataFrame, require_rating: bool = True) -> dict:
        """
        Kiểm tra DataFrame sau khi map.
        Trả về dict với missing columns và data quality info.
        """
        missing_features = [c for c in CANONICAL_FEATURES if c not in df.columns]
        missing_meta     = [c for c in CANONICAL_META if c not in df.columns]

        if require_rating:
            missing_meta = [c for c in missing_meta if c != 'Credit_Rating']

        null_pct = {}
        for col in CANONICAL_FEATURES:
            if col in df.columns:
                pct = df[col].isna().mean() * 100
                if pct > 0:
                    null_pct[col] = round(pct, 1)

        report = {
            'n_records': len(df),
            'missing_feature_cols': missing_features,
            'missing_meta_cols': missing_meta,
            'null_percentage': null_pct,
            'is_valid': len(missing_features) == 0,
        }

        if report['missing_feature_cols']:
            logger.warning(f"Missing canonical features: {missing_features}")
        if null_pct:
            logger.warning(f"Null percentages: {null_pct}")

        return report

    def summary(self, df: pd.DataFrame) -> str:
        lines = [
            f"  Source:   {df.get('data_source', pd.Series(['unknown'])).iloc[0] if 'data_source' in df.columns else 'unknown'}",
            f"  Records:  {len(df)}",
            f"  Columns:  {df.shape[1]}",
        ]
        if 'Sector' in df.columns:
            lines.append(f"  Sectors:\n{df['Sector'].value_counts().to_string()}")
        if 'Credit_Rating' in df.columns:
            lines.append(f"  Ratings:\n{df['Credit_Rating'].value_counts().to_string()}")
        return '\n'.join(lines)
