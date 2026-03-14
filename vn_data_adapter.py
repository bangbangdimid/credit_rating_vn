"""
vn_data_adapter.py
==================
Chuyển đổi dữ liệu BCTC từ vnstock → Feature vector chuẩn.

FIX v7 (2026-03) — vnstock 3.4.2 KBS source:
  - [CRITICAL] Thêm KBS item_id vào tất cả BALANCE_SHEET_MAP, INCOME_STATEMENT_MAP,
    CASHFLOW_MAP, RATIO_MAP để _find_col() có thể tìm đúng cột sau khi
    _normalize_kbs_df() chuyển item_id thành column names.
  - KBS balance sheet items (từ debug output thực tế VCB):
      assets, i.cash_gold_and_silver_precious_stones,
      iii.placements_at_and_loans_to_other_credit_institutions,
      viii.loans_and_advances_to_customers, ...
  - KBS income statement items:
      i.net_interest_income, n_1.interest_income_and_similar_income,
      n_2.interest_expense_and_similar_expenses, ...
  - KBS ratio items: trailing_eps, book_value_per_share_bvps, p_e, p_b,
      dividend_yield, roe, roa, ...
  - Giữ nguyên toàn bộ logic v6 (sector normalization, ROCE/CR fallback chains)
"""

import pandas as pd
import numpy as np
import logging
from pathlib import Path
from typing import Dict, Optional, Any

logger = logging.getLogger(__name__)

# ================================================================
# COLUMN MAPS
# ================================================================

BALANCE_SHEET_MAP = {
    'total_assets': [
        # KBS confirmed
        'total_assets',                          # VCB exact
        'total_liabilities_and_owners_equity',   # VCB fallback
        'total_owners_equity_and_liabilities',   # VNM exact
        'assets',
        # Legacy
        'TOTAL ASSETS (Bn. VND)', 'totalAssets', 'Total Assets', 'tong_tai_san',
    ],
    'total_liabilities': [
        # KBS confirmed
        'total_liabilities',    # VCB exact
        'a.liabilities',        # VNM exact
        'liabilities',
        # Legacy
        'LIABILITIES (Bn. VND)', 'totalLiabilities', 'Total Liabilities', 'TOTAL RESOURCES (Bn. VND)',
    ],
    'total_equity': [
        # KBS confirmed
        'b.owners_equity',            # VNM exact
        'viii.capital_and_reserves',  # VCB exact
        'i.owners_equity',            # VNM sub-item
        'total_equity', 'equity', 'owners_equity',
        # Legacy
        "OWNER'S EQUITY(Bn.VND)", 'stockholders_equity', 'Owner Equity',
        'totalStockholdersEquity', 'von_chu_so_huu',
    ],
    'short_term_assets': [
        # KBS confirmed VNM
        'a.short_term_assets',
        'short_term_assets', 'current_assets',
        # Legacy
        'SHORT-TERM ASSETS (Bn. VND)', 'Total Current Assets',
        'currentAssets', 'tai_san_ngan_han',
        'Current Assets', 'Short-term Assets', 'Short Term Assets',
    ],
    'short_term_liabilities': [
        # KBS confirmed VNM
        'i.short_term_liabilities',
        'short_term_liabilities', 'current_liabilities',
        # Legacy
        'SHORT-TERM LIABILITIES (Bn. VND)', 'Total Current Liabilities',
        'currentLiabilities', 'no_ngan_han',
        'Current Liabilities', 'Short-term Liabilities', 'Short Term Liabilities',
    ],
    'long_term_debt': [
        # KBS item_id
        'long_term_borrowings', 'long_term_debt', 'longTermDebt',
        'ii.long_term_liabilities', 'long_term_liabilities',
        'ix.convertible_bonds_cds_and_other_valuable_papers_issued',
        # Legacy
        'no_dai_han', 'Convertible bonds/CDs and other valuable papers issued',
        'Long-term Debt', 'Long Term Borrowings',
    ],
    'short_term_debt': [
        # KBS item_id (bank)
        'ii.deposits_and_borrowings_from_other_credit_institutions',
        'deposits_and_borrowings_from_other_credit_institutions',
        'i.due_to_gov_and_borrowings_from_sbv',
        # KBS item_id (non-bank)
        'short_term_borrowings', 'short_term_debt',
        # Legacy
        'shortTermDebt', 'Short Term Borrowings',
        'Deposits and borrowings from other credit institutions',
        'Due to Gov and borrowings from SBV',
        'Short-term Debt', 'Short-term Borrowings',
    ],
    'cash': [
        # KBS confirmed
        'i.cash_and_cash_equivalents',              # VNM exact
        'i.cash_gold_and_silver_precious_stones',   # VCB exact
        'cash_and_cash_equivalents', 'cash',
        # Legacy
        'Cash and cash equivalents (Bn. VND)', 'cashAndCashEquivalents',
        'tien_va_tuong_duong_tien', 'CASH (Bn. VND)',
    ],
    'retained_earnings': [
        # KBS confirmed
        'n_11.undistributed_earnings_after_tax',                  # VNM exact
        'n_5.undistributed_earnings_after_tax_accumulated_loss',  # VCB exact
        'undistributed_earnings_in_this_period',
        'undistributed_earnings', 'retained_earnings', 'retainedEarnings',
        # Legacy
        'Undistributed earnings (Bn. VND)', 'Retained Earnings',
        'loi_nhuan_chua_phan_phoi', 'Undistributed Earnings', 'Retained Profit',
    ],
    'long_term_assets': [
        # KBS confirmed VNM
        'b.long_term_assets',
        'long_term_assets', 'non_current_assets',
        # Legacy
        'LONG-TERM ASSETS (Bn. VND)', 'Total Non-Current Assets',
        'nonCurrentAssets', 'tai_san_dai_han',
    ],
    'customer_deposits': [
        # KBS confirmed VCB
        'iii.deposits_from_customers',
        'deposits_from_customers', 'customer_deposits', 'deposits',
        # Legacy
        'Deposits from customers', 'Customer Deposits', 'Deposits from Customers',
    ],
    'loans_to_customers': [
        # KBS confirmed VCB
        'vi.loans_advances_and_finance_leases_to_customers',
        'n_1.loans_advances_and_finance_leases_to_customers',
        'loans_and_advances_to_customers', 'loans_to_customers', 'net_loans',
        # Legacy
        'Loans and advances to customers, net',
        'Loans and advances to customers',
        'Loan and Advances to Customers',
    ],
    'fixed_assets': [
        # KBS confirmed VNM + VCB
        'ii.fixed_assets',        # VNM
        'x.fixed_assets',         # VCB
        'fixed_assets', 'tangible_fixed_assets', 'property_plant_and_equipment',
        # Legacy
        'Fixed Assets', 'Property Plant Equipment',
        'Tangible Fixed Assets', 'FIXED ASSETS (Bn. VND)',
    ],
    'inventory': [
        # KBS confirmed VNM
        'iv.inventories',
        'inventories', 'inventory',
        # Legacy
        'Inventories', 'Inventory', 'INVENTORIES (Bn. VND)', 'hang_ton_kho',
    ],
    'accounts_receivable': [
        # KBS confirmed VNM
        'iii.short_term_receivables',
        'short_term_receivables', 'trade_receivables', 'accounts_receivable',
        # Legacy
        'Short-term Receivables', 'Accounts Receivable',
        'Trade Receivables', 'SHORT-TERM RECEIVABLES (Bn. VND)', 'phai_thu_ngan_han',
    ],
    'short_term_investments': [
        # KBS item_id
        'short_term_financial_investments', 'short_term_investments',
        # Legacy
        'Short-term Investments', 'SHORT-TERM INVESTMENTS (Bn. VND)',
        'Short-term Financial Investments',
    ],
}

INCOME_STATEMENT_MAP = {
    'revenue': [
        # KBS confirmed VNM
        'n_3.net_revenue',
        # KBS confirmed VCB — computed later as sum of income lines
        # Fallback direct lookup:
        'ix.operating_profit_before_provision_for_credit_losses',
        'i.net_interest_income',
        'net_revenue', 'revenue', 'net_sales',
        # Legacy
        'Revenue (Bn. VND)', 'Net Revenue', 'Net Sales', 'netRevenue',
        'doanh_thu_thuan', 'Total operating revenue', 'REVENUE (Bn. VND)',
    ],
    'revenue_yoy': [
        # KBS ratio: non-bank='net_revenue', bank='net_interest_income'
        'net_revenue',   # VNM: 3.02 = 3.02% YoY
        'Revenue YoY (%)',
    ],
    'net_income': [
        # KBS confirmed VNM + VCB
        'n_18.net_profit_after_tax',                                          # VNM
        'xiii.net_profit_after_tax',                                          # VCB
        'xv.net_profit_atttributable_to_the_equity_holders_of_the_bank',      # VCB parent
        'profit_after_tax_for_shareholders_of_parent_company',                # VNM parent
        'profit_after_tax', 'net_profit_after_tax', 'net_income', 'net_profit',
        # Legacy
        'Net Profit For the Year', 'Attributable to parent company',
        'netIncome', 'loi_nhuan_sau_thue', 'Net Income', 'Profit After Tax',
    ],
    'ebit': [
        # KBS confirmed VNM + VCB
        'n_11.operating_profit',                                   # VNM exact
        'ix.operating_profit_before_provision_for_credit_losses',  # VCB exact
        'operating_profit', 'profit_from_operations', 'ebit',
        # Legacy
        'Operating Profit before Provision', 'operating_income', 'Operating Income', 'EBIT',
        'operatingIncome', 'Operating Profit', 'Earnings Before Interest and Tax',
        'Operating Profit (Loss)', 'EBIT (Bn. VND)', 'Profit from Operations',
    ],
    'profit_before_tax': [
        # KBS confirmed VNM + VCB
        'n_15.profit_before_tax',   # VNM exact
        'xi.profit_before_tax',     # VCB exact
        'profit_before_tax', 'pre_tax_profit', 'earnings_before_tax',
        # Legacy
        'Profit before tax', 'Pre-tax Profit', 'Earnings Before Tax', 'EBT',
        'Profit Before Tax (Bn. VND)',
    ],
    'interest_expense': [
        # KBS confirmed VNM + VCB
        'of_which_interest_expenses',                # VNM income statement
        'interest_expense',                          # VNM cashflow
        'n_2.interest_expense_and_similar_expenses', # VCB income statement
        # Legacy
        'Interest and Similar Expenses', 'interestExpense', 'chi_phi_lai_vay',
        'Finance Costs', 'Interest Expense', 'Financial Expenses',
    ],
    'gross_profit': [
        # KBS confirmed VNM
        'n_5.gross_profit',
        'gross_profit', 'gross_profit_from_sales',
        # Legacy
        'Gross Profit', 'grossProfit', 'Gross Profit (Bn. VND)', 'loi_nhuan_gop',
    ],
    'cogs': [
        # KBS confirmed VNM
        'n_4.cost_of_goods_sold',
        'cost_of_goods_sold', 'direct_costs', 'cogs',
        # Legacy
        'Cost of Goods Sold', 'COGS', 'Cost of Sales', 'Direct Costs',
        'COST OF GOODS SOLD (Bn. VND)',
    ],
    'net_interest_income': [
        # KBS confirmed VCB
        'i.net_interest_income',
        'net_interest_income',
        # Legacy
        'Net Interest Income', 'netInterestIncome', 'thu_nhap_lai_thuan',
    ],
    'total_operating_income': [
        # KBS item_id
        'total_operating_income', 'total_operating_revenue',
        # Legacy
        'Total operating revenue', 'totalOperatingIncome',
    ],
    'depreciation': [
        # KBS confirmed VNM cashflow
        'depreciation_of_fixed_assets_and_properties_investment',
        'depreciation_and_amortization', 'depreciation',
        # Legacy
        'depreciationAndAmortization', 'khau_hao', 'D&A',
        'Depreciation & Amortization', 'Depreciation And Amortization',
    ],
    'selling_expenses': [
        # KBS confirmed VNM
        'n_9.selling_expenses',
        'selling_expenses', 'selling_and_distribution_expenses',
        # Legacy
        'Selling Expenses', 'Selling and Distribution Expenses',
    ],
    'admin_expenses': [
        # KBS confirmed VNM
        'n_10.general_and_administrative_expenses',
        'general_and_administrative_expenses', 'admin_expenses', 'administrative_expenses',
        # Legacy
        'General and Administrative Expenses', 'Administrative Expenses', 'G&A Expenses',
    ],
}

CASHFLOW_MAP = {
    'operating_cashflow': [
        # KBS confirmed VCB + VNM (same key!)
        'net_cash_flows_from_operating_activities',
        'i.cash_flows_from_operating_activities',
        'cash_flows_from_operating_activities', 'operating_cashflow', 'operating_activities',
        # Legacy
        'Net Cash From Operating Activities', 'netCashOperatingActivities',
        'luong_tien_tu_hoat_dong_kinh_doanh',
    ],
}

RATIO_MAP = {
    'ebit_margin_direct': [
        'ebit_margin',   # VNM confirmed: 18.82
    ],
    'roe': [
        # KBS item_id — từ debug output ratio
        'roe', 'return_on_equity_roe',
        # Legacy
        'Chỉ tiêu khả năng sinh lợi_ROE (%)', 'ROE', 'returnOnEquity',
    ],
    'roa': [
        # KBS item_id
        'roa', 'return_on_assets_roa',
        # Legacy
        'Chỉ tiêu khả năng sinh lợi_ROA (%)', 'ROA', 'returnOnAssets',
    ],
    'net_profit_margin': [
        # KBS item_id
        'net_profit_margin', 'profit_margin',
        # Legacy
        'Chỉ tiêu khả năng sinh lợi_Net Profit Margin (%)', 'netProfitMargin',
    ],
    'market_cap': [
        # KBS item_id
        'market_cap', 'market_capitalization',
        # Legacy
        'Chỉ tiêu định giá_Market Capital (Bn. VND)', 'marketCap', 'MarketCap',
    ],
    'price_to_book': [
        # KBS item_id — từ debug: p_b
        'p_b', 'price_to_book', 'pb',
        # Legacy
        'Chỉ tiêu định giá_P/B', 'priceToBook', 'P/B',
    ],
    'price_to_earnings': [
        # KBS item_id — từ debug: p_e
        'p_e', 'price_to_earnings', 'pe',
        # Legacy
        'Chỉ tiêu định giá_P/E', 'P/E',
    ],
    'current_ratio_direct': [
        # KBS confirmed VNM
        'short_term_ratio',
        'current_ratio', 'liquidity_ratio',
        # Legacy
        'Chỉ tiêu thanh khoản_Current Ratio',
        'Chỉ tiêu thanh khoản_Hệ số thanh toán hiện hành',
        'currentRatio', 'Current Ratio', 'Liquidity Ratio',
    ],
    'roce_direct': [
        # KBS confirmed VNM
        'return_on_capital_employed_roce',
        'roce', 'return_on_capital_employed',
        # Legacy
        'Chỉ tiêu sinh lợi_ROCE (%)', 'ROCE', 'returnOnCapitalEmployed',
    ],
}

# ======================== SECTOR CONFIG ========================

FINANCIAL_SECTORS = {'Banking', 'Insurance', 'Securities', 'FinancialServices', 'Finance'}

SECTOR_NORMALIZE = {
    # Tiếng Việt
    'Ngân hàng': 'Banking',
    'Ngân Hàng': 'Banking',
    'NGÂN HÀNG': 'Banking',
    'Tài chính': 'FinancialServices',
    'Tài Chính': 'FinancialServices',
    'Tài chính - Ngân hàng': 'Banking',
    'Bảo hiểm': 'Insurance',
    'Chứng khoán': 'Securities',
    'Dịch vụ tài chính': 'FinancialServices',
    'Công nghệ thông tin': 'Technology',
    'Phần mềm & Dịch vụ': 'Technology',
    'Phần mềm': 'Technology',
    'Công Nghệ': 'Technology',
    'Bất động sản': 'RealEstate',
    'Bất Động Sản': 'RealEstate',
    'Thép': 'Manufacturing',
    'Hóa chất': 'Manufacturing',
    'Vật liệu xây dựng': 'Manufacturing',
    'Dệt may': 'Manufacturing',
    'Thực phẩm & Đồ uống': 'Manufacturing',
    'Thực phẩm': 'Manufacturing',
    'Dược phẩm': 'Healthcare',
    'Điện tử & Linh kiện': 'Manufacturing',
    'Bán lẻ': 'Retail',
    'Dầu khí': 'Energy',
    'Dầu Khí': 'Energy',
    'Điện & Năng lượng tái tạo': 'Energy',
    'Điện lực': 'Energy',
    'Năng lượng': 'Energy',
    'Vận tải': 'Transportation',
    'Vận Tải': 'Transportation',
    'Nông nghiệp': 'Agriculture',
    'Xây dựng': 'Construction',
    'Y tế': 'Healthcare',
    'Viễn thông': 'Telecom',
    'Du lịch & Giải trí': 'Tourism',
    # Tiếng Anh
    'Banking': 'Banking',
    'Banks': 'Banking',
    'Bank': 'Banking',
    'Finance': 'FinancialServices',
    'Financial Services': 'FinancialServices',
    'Insurance': 'Insurance',
    'Securities': 'Securities',
    'Technology': 'Technology',
    'Information Technology': 'Technology',
    'Software': 'Technology',
    'Real Estate': 'RealEstate',
    'Real estate': 'RealEstate',
    'Manufacturing': 'Manufacturing',
    'Industrial': 'Manufacturing',
    'Industrials': 'Manufacturing',
    'Materials': 'Manufacturing',
    'Consumer Staples': 'Manufacturing',
    'Consumer Discretionary': 'Retail',
    'Retail': 'Retail',
    'Energy': 'Energy',
    'Oil & Gas': 'Energy',
    'Utilities': 'Energy',
    'Healthcare': 'Healthcare',
    'Health Care': 'Healthcare',
    'Pharmaceuticals': 'Healthcare',
    'Transportation': 'Transportation',
    'Telecommunication': 'Telecom',
    'Telecom': 'Telecom',
    'Agriculture': 'Agriculture',
    'Construction': 'Construction',
    # icb_name dạng đầy đủ
    'Ngân hàng thương mại': 'Banking',
    'Tài chính tiêu dùng': 'FinancialServices',
    'Môi giới chứng khoán': 'Securities',
    'Quản lý quỹ': 'FinancialServices',
    'Bất động sản nhà ở': 'RealEstate',
    'Bất động sản thương mại': 'RealEstate',
    'Xây dựng & Vật liệu': 'Construction',
    'Hàng tiêu dùng': 'Retail',
    'Bán lẻ thực phẩm': 'Retail',
    'Điện, khí đốt & nước': 'Energy',
    'Khai khoáng': 'Energy',
    'Thép & Kim loại': 'Manufacturing',
    'Hóa chất cơ bản': 'Manufacturing',
    'Dược phẩm & Sinh học': 'Healthcare',
    'Thiết bị y tế': 'Healthcare',
    'Phần mềm & Dịch vụ CNTT': 'Technology',
    'Thiết bị công nghệ': 'Technology',
    'Vận tải biển': 'Transportation',
    'Vận tải hàng không': 'Transportation',
    'Logistics': 'Transportation',
    'Du lịch & Khách sạn': 'Tourism',
    'Truyền thông & Giải trí': 'Tourism',
}

# ======================== HELPERS ========================

def _safe_str(val) -> str:
    if val is None:
        return ''
    if isinstance(val, tuple):
        for item in val:
            s = str(item).strip()
            if s:
                return s
        return ''
    if isinstance(val, (list, np.ndarray)):
        return str(val[0]) if len(val) > 0 else ''
    return str(val).strip()


def _flatten_df(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return df
    if isinstance(df.columns, pd.MultiIndex):
        df = df.copy()
        df.columns = [
            '_'.join(str(lvl) for lvl in col if str(lvl).strip())
            if isinstance(col, tuple) else str(col)
            for col in df.columns
        ]
        return df
    has_tuple = any(isinstance(c, tuple) for c in df.columns)
    if has_tuple:
        df = df.copy()
        new_cols = []
        for c in df.columns:
            if isinstance(c, tuple):
                joined = '_'.join(str(x) for x in c if str(x).strip())
                new_cols.append(joined if joined else str(c))
            else:
                new_cols.append(str(c))
        df.columns = new_cols
    return df


# ======================== ADAPTER CLASS ========================

class VNDataAdapter:

    def __init__(self):
        self.required_features = [
            'ROA', 'ROE', 'ROCE', 'EBIT_Margin', 'Current_Ratio',
            'Revenue_Growth_YoY', 'Log_Revenue',
            'Debt/Assets', 'Debt/Equity', 'Net_Debt/EBITDA',
            'Asset_Turnover', 'WCTA', 'RETA', 'Market_to_Book',
        ]
        self.max_missing = 7

    def _find_col(self, df: pd.DataFrame, candidates: list) -> Optional[str]:
        if df is None or df.empty:
            return None
        import re
        lookup = {}
        for c in df.columns:
            c_str = str(c)
            norm = c_str.lower().strip()
            lookup[norm] = c
            stripped = re.sub(r'\s*\(.*?\)\s*', '', c_str).lower().strip()
            if stripped:
                lookup[stripped] = c
        for cand in candidates:
            if cand in df.columns:
                return cand
            cand_norm = str(cand).lower().strip()
            if cand_norm in lookup:
                return lookup[cand_norm]
            for col_norm, col_orig in lookup.items():
                if cand_norm in col_norm:
                    return col_orig
        return None

    def _get_val(self, df: pd.DataFrame, candidates: list,
                 year_idx: int = 0) -> Optional[float]:
        if df is None or df.empty or year_idx >= len(df):
            return None
        col = self._find_col(df, candidates)
        if col is None:
            return None
        # Bỏ qua cột 'year' (metadata của KBS wide format)
        if str(col).lower() == 'year':
            return None
        try:
            val = df.iloc[year_idx][col]
            if pd.isna(val):
                return None
            return float(val)
        except (ValueError, TypeError):
            return None

    def _get_growth(self, df: pd.DataFrame, candidates: list) -> Optional[float]:
        """Tính YoY growth từ 2 rows liên tiếp (year_idx=0 mới nhất, 1 năm trước)."""
        if df is None or df.empty or len(df) < 2:
            return None
        col = self._find_col(df, candidates)
        if col is None or str(col).lower() == 'year':
            return None
        try:
            v0 = float(df.iloc[0][col])
            v1 = float(df.iloc[1][col])
            if pd.isna(v0) or pd.isna(v1) or abs(v1) < 1e-9:
                return None
            return float(np.clip((v0 - v1) / abs(v1), -0.9, 5.0))
        except (ValueError, TypeError):
            return None

    def _safe_div(self, num, den, default=None):
        if num is None or den is None:
            return default
        try:
            num, den = float(num), float(den)
            if abs(den) < 1e-9:
                return default
            return num / den
        except (TypeError, ValueError):
            return default

    def _normalize_sector(self, sector_vn: str) -> str:
        """
        Chuẩn hóa tên sector từ vnstock về tên chuẩn của pipeline.
        Thứ tự ưu tiên:
          1. Exact match trong SECTOR_NORMALIZE
          2. Case-insensitive exact match
          3. Substring keyword fallback
          4. 'Other'
        """
        if not sector_vn or not isinstance(sector_vn, str):
            return 'Other'
        raw = sector_vn.strip()
        # 1. Exact match
        result = SECTOR_NORMALIZE.get(raw)
        if result:
            return result
        # 2. Case-insensitive exact match
        raw_lower = raw.lower()
        for key, val in SECTOR_NORMALIZE.items():
            if key.lower() == raw_lower:
                logger.debug(f"[sector] case-insensitive: '{raw}' → '{val}'")
                return val
        # 3. Substring keyword fallback
        KW = [
            (['ngân hàng', 'bank', 'tín dụng'],                    'Banking'),
            (['bảo hiểm', 'insur'],                                  'Insurance'),
            (['chứng khoán', 'securit', 'môi giới'],                 'Securities'),
            (['tài chính', 'financ'],                                'FinancialServices'),
            (['bất động sản', 'real estate'],                       'RealEstate'),
            (['công nghệ', 'technology', 'phần mềm', 'software'],  'Technology'),
            (['bán lẻ', 'retail', 'hàng tiêu dùng', 'consumer'],   'Retail'),
            (['dầu khí', 'oil', 'gas', 'khai khoáng',
              'năng lượng', 'energy', 'điện lực', 'utilities'],    'Energy'),
            (['dược phẩm', 'y tế', 'health', 'pharma'],            'Healthcare'),
            (['vận tải', 'transport', 'logist'],                    'Transportation'),
            (['viễn thông', 'telecom'],                             'Telecom'),
            (['xây dựng', 'construct', 'vật liệu'],                 'Construction'),
            (['nông nghiệp', 'agri'],                                'Agriculture'),
            (['thép', 'kim loại', 'hóa chất', 'dệt may',
              'thực phẩm', 'đồ uống', 'manufactur', 'industri'],   'Manufacturing'),
        ]
        for keywords, mapped in KW:
            if any(kw in raw_lower for kw in keywords):
                logger.debug(f"[sector] keyword: '{raw}' → '{mapped}'")
                return mapped
        logger.warning(
            f"[sector] Không nhận ra: '{raw}' → 'Other'. "
            f"Thêm vào SECTOR_NORMALIZE trong vn_data_adapter.py nếu cần."
        )
        return 'Other'

    # ================================================================
    # EBIT — 6 fallback paths
    # ================================================================

    def _compute_ebit(self, ticker: str, bs: pd.DataFrame,
                      inc: pd.DataFrame, is_bank: bool) -> Optional[float]:
        # Path 1: Cột trực tiếp
        ebit = self._get_val(inc, INCOME_STATEMENT_MAP['ebit'])
        if ebit is not None:
            logger.debug(f"[{ticker}] EBIT path 1 (direct): {ebit:.2f}")
            return ebit

        # Path 2: PBT + Interest Expense
        pbt = self._get_val(inc, INCOME_STATEMENT_MAP['profit_before_tax'])
        interest_exp = self._get_val(inc, INCOME_STATEMENT_MAP['interest_expense']) or 0.0
        if pbt is not None:
            ebit = pbt + interest_exp
            logger.debug(f"[{ticker}] EBIT path 2 (PBT + IE): {ebit:.2f}")
            return ebit

        # Path 3: Gross Profit − OPEX
        gross_profit = self._get_val(inc, INCOME_STATEMENT_MAP['gross_profit'])
        selling_exp  = self._get_val(inc, INCOME_STATEMENT_MAP['selling_expenses']) or 0.0
        admin_exp    = self._get_val(inc, INCOME_STATEMENT_MAP['admin_expenses'])   or 0.0
        if gross_profit is not None:
            ebit = gross_profit - selling_exp - admin_exp
            logger.debug(f"[{ticker}] EBIT path 3 (GP - OPEX): {ebit:.2f}")
            return ebit

        # Path 4: Net Income + Interest Expense
        net_income_val = self._get_val(inc, INCOME_STATEMENT_MAP['net_income'])
        if net_income_val is not None and interest_exp != 0.0:
            ebit = net_income_val + interest_exp
            logger.info(f"[{ticker}] EBIT path 4 (NI + IE): {ebit:.2f}")
            return ebit

        # Path 5: Net Income alone (non-banking)
        if net_income_val is not None and not is_bank:
            logger.info(f"[{ticker}] EBIT path 5 (NI proxy): {net_income_val:.2f}")
            return net_income_val

        # Path 6: Banking-specific
        if is_bank:
            op_profit = self._get_val(inc, ['Operating Profit before Provision',
                                             'Profit before tax'])
            if op_profit is not None:
                logger.debug(f"[{ticker}] EBIT path 6 (Banking op_profit): {op_profit:.2f}")
                return op_profit
            net_int = self._get_val(inc, INCOME_STATEMENT_MAP['net_interest_income'])
            if net_int is not None:
                logger.debug(f"[{ticker}] EBIT path 6b (Banking NII proxy): {net_int:.2f}")
                return net_int

        logger.warning(f"[{ticker}] EBIT: không tính được từ bất kỳ path nào")
        return None

    # ================================================================
    # ROCE — 4 fallback paths
    # ================================================================

    def _compute_roce(self, ticker: str, bs: pd.DataFrame,
                      inc: pd.DataFrame, ratio_df: pd.DataFrame,
                      is_bank: bool,
                      ebit: Optional[float],
                      total_assets: Optional[float],
                      current_liabilities: Optional[float],
                      total_equity: Optional[float]) -> Optional[float]:
        # Path 0: Ratio table trực tiếp
        roce_direct = self._get_val(ratio_df, RATIO_MAP.get('roce_direct', []))
        if roce_direct is not None:
            result = roce_direct / 100.0
            logger.debug(f"[{ticker}] ROCE path0 (ratio table): {result:.4f}")
            return result

        if ebit is None:
            logger.warning(f"[{ticker}] ROCE: EBIT=None — không tính được")
            return None

        # CE1: Total Assets − Current Liabilities
        if total_assets is not None and current_liabilities is not None:
            cap_emp = total_assets - current_liabilities
            if cap_emp > 0:
                roce = self._safe_div(ebit, cap_emp)
                logger.debug(f"[{ticker}] ROCE CE1 (TA-CL): {roce:.4f}")
                return roce

        # CE2: tên cột thay thế
        cl_alt = self._get_val(bs, [
            'SHORT-TERM LIABILITIES (Bn. VND)',
            'short_term_liabilities', 'Current Liabilities',
            'Short-term Liabilities', 'Short Term Liabilities',
            'no_ngan_han',
        ])
        if total_assets is not None and cl_alt is not None:
            cap_emp = total_assets - cl_alt
            if cap_emp > 0:
                roce = self._safe_div(ebit, cap_emp)
                logger.info(f"[{ticker}] ROCE CE2 (TA-CL_alt): {roce:.4f}")
                return roce

        # CE3: Total Equity + Long-term Debt
        long_term_debt = self._get_val(bs, BALANCE_SHEET_MAP['long_term_debt']) or 0.0
        if total_equity is not None:
            cap_emp = total_equity + long_term_debt
            if cap_emp > 0:
                roce = self._safe_div(ebit, cap_emp)
                logger.info(f"[{ticker}] ROCE CE3 (Equity+LTD): {roce:.4f}")
                return roce

        # CE4: Proxy TA × 0.6
        if total_assets is not None and total_assets > 0:
            cap_emp = total_assets * 0.6
            roce = self._safe_div(ebit, cap_emp)
            logger.warning(f"[{ticker}] ROCE CE4 (proxy TA×0.6): {roce:.4f} — ước tính thô")
            return roce

        logger.warning(f"[{ticker}] ROCE: không tính được")
        return None

    # ================================================================
    # Current Ratio — nhiều fallback paths, trả về (ratio, ca, cl)
    # ================================================================

    def _compute_current_ratio(self, ticker: str, bs: pd.DataFrame,
                                ratio_df: pd.DataFrame,
                                is_bank: bool,
                                current_assets: Optional[float],
                                current_liabilities: Optional[float],
                                total_assets: Optional[float],
                                total_equity: Optional[float],
                                total_liabilities: Optional[float]):
        # ── BANKING ──
        if is_bank:
            loans    = self._get_val(bs, BALANCE_SHEET_MAP['loans_to_customers'])
            deposits = self._get_val(bs, BALANCE_SHEET_MAP['customer_deposits'])
            if loans is not None and deposits is not None and deposits > 0:
                ldr = loans / deposits
                logger.debug(f"[{ticker}] Current_Ratio CR_B1 (LDR={ldr:.4f})")
                return ldr, loans, deposits
            fixed_assets = self._get_val(bs, BALANCE_SHEET_MAP['fixed_assets']) or 0.0
            if total_assets is not None and deposits is not None and deposits > 0:
                liquid = total_assets - fixed_assets
                ratio  = min(liquid / deposits, 5.0)
                logger.info(f"[{ticker}] Current_Ratio CR_B2 (liquid/deposits): {ratio:.4f}")
                return ratio, liquid, deposits
            cr_rt = self._get_val(ratio_df, RATIO_MAP.get('current_ratio_direct', []))
            if cr_rt is not None:
                logger.info(f"[{ticker}] Current_Ratio CR_B3 (ratio table): {cr_rt:.4f}")
                return cr_rt, None, None
            logger.info(f"[{ticker}] Current_Ratio Banking: dùng LDR ngành 0.85")
            return 0.85, None, None

        # ── NON-BANKING ──
        # CR1: Standard CA / CL
        if current_assets is not None and current_liabilities is not None and current_liabilities > 0:
            cr = current_assets / current_liabilities
            logger.debug(f"[{ticker}] Current_Ratio CR1: {cr:.4f}")
            return cr, current_assets, current_liabilities

        # CR2: tên cột thay thế
        ca_alt = self._get_val(bs, [
            'SHORT-TERM ASSETS (Bn. VND)', 'Current Assets', 'Short-term Assets',
            'Short Term Assets', 'CURRENT ASSETS (Bn. VND)', 'tai_san_ngan_han',
        ])
        cl_alt = self._get_val(bs, [
            'SHORT-TERM LIABILITIES (Bn. VND)', 'Current Liabilities', 'Short-term Liabilities',
            'Short Term Liabilities', 'CURRENT LIABILITIES (Bn. VND)', 'no_ngan_han',
        ])
        if ca_alt is not None and cl_alt is not None and cl_alt > 0:
            cr = ca_alt / cl_alt
            logger.info(f"[{ticker}] Current_Ratio CR2 (alt names): {cr:.4f}")
            return cr, ca_alt, cl_alt

        # CR3: CA gián tiếp = TA - Long-term Assets
        lta = self._get_val(bs, BALANCE_SHEET_MAP['long_term_assets'])
        ca_indirect = None
        if total_assets is not None and lta is not None:
            ca_indirect = total_assets - lta
            if ca_indirect < 0:
                ca_indirect = None
        cl_best = current_liabilities or cl_alt
        if ca_indirect is not None and cl_best is not None and cl_best > 0:
            cr = ca_indirect / cl_best
            logger.info(f"[{ticker}] Current_Ratio CR3 ((TA-LTA)/CL): {cr:.4f}")
            return cr, ca_indirect, cl_best

        # CR4: CL gián tiếp = Total Liabilities - Long-term Debt
        ltd = self._get_val(bs, BALANCE_SHEET_MAP['long_term_debt']) or 0.0
        cl_indirect = None
        if total_liabilities is not None:
            cl_indirect = total_liabilities - ltd
            if cl_indirect <= 0:
                cl_indirect = None
        ca_best = current_assets or ca_alt or ca_indirect
        if ca_best is not None and cl_indirect is not None:
            cr = ca_best / cl_indirect
            logger.info(f"[{ticker}] Current_Ratio CR4 (CA/(TL-LTD)): {cr:.4f}")
            return cr, ca_best, cl_indirect

        # CR5: Ratio table trực tiếp
        cr_rt = self._get_val(ratio_df, RATIO_MAP.get('current_ratio_direct', []))
        if cr_rt is not None:
            logger.info(f"[{ticker}] Current_Ratio CR5 (ratio table): {cr_rt:.4f}")
            return cr_rt, None, None

        # CR6: Proxy CL = Total Liabilities × 0.6
        if ca_best is not None and total_liabilities is not None and total_liabilities > 0:
            cl_est = total_liabilities * 0.6
            cr = ca_best / cl_est
            logger.warning(f"[{ticker}] Current_Ratio CR6 (proxy CL=TL×0.6): {cr:.4f} — ước tính thô")
            return cr, ca_best, cl_est

        logger.warning(f"[{ticker}] Current_Ratio: không tính được")
        return None, None, None

    # ================================================================
    # MAIN TRANSFORM
    # ================================================================

    def transform(
        self,
        ticker: str,
        company_info: Dict,
        statements: Dict[str, pd.DataFrame]
    ) -> Optional[Dict[str, Any]]:
        """Transform BCTC → feature dict."""

        # KBS wide format — đã normalize sẵn, KHÔNG gọi _flatten_df()
        bs       = statements.get('balance',  pd.DataFrame())
        inc      = statements.get('income',   pd.DataFrame())
        cf       = statements.get('cashflow', pd.DataFrame())
        ratio_df = statements.get('ratio',    pd.DataFrame())

        for df_obj in [bs, inc, cf, ratio_df]:
            if df_obj is not None and not df_obj.empty:
                if not isinstance(df_obj.index, pd.RangeIndex):
                    df_obj.reset_index(drop=True, inplace=True)

        logger.debug(f"[{ticker}] BS  cols: {[c for c in bs.columns if c!='year'][:6] if not bs.empty else '[]'}")
        logger.debug(f"[{ticker}] Inc cols: {[c for c in inc.columns if c!='year'][:6] if not inc.empty else '[]'}")
        logger.debug(f"[{ticker}] Ratio cols: {[c for c in ratio_df.columns if c!='year'][:6] if not ratio_df.empty else '[]'}")

        # Sector
        sector_vn = _safe_str(company_info.get('sector_vn', ''))
        sector    = self._normalize_sector(sector_vn)
        is_bank   = sector in FINANCIAL_SECTORS

        # Balance Sheet
        total_assets        = self._get_val(bs, BALANCE_SHEET_MAP['total_assets'])
        total_liabilities   = self._get_val(bs, BALANCE_SHEET_MAP['total_liabilities'])
        total_equity        = self._get_val(bs, BALANCE_SHEET_MAP['total_equity'])
        current_assets      = self._get_val(bs, BALANCE_SHEET_MAP['short_term_assets'])
        current_liabilities = self._get_val(bs, BALANCE_SHEET_MAP['short_term_liabilities'])
        long_term_debt      = self._get_val(bs, BALANCE_SHEET_MAP['long_term_debt'])  or 0.0
        short_term_debt     = self._get_val(bs, BALANCE_SHEET_MAP['short_term_debt']) or 0.0
        cash                = self._get_val(bs, BALANCE_SHEET_MAP['cash'])             or 0.0
        retained_earnings   = self._get_val(bs, BALANCE_SHEET_MAP['retained_earnings'])

        if total_liabilities is None and total_assets and total_equity:
            total_liabilities = total_assets - total_equity

        # Income Statement
        revenue      = self._get_val(inc, INCOME_STATEMENT_MAP['revenue'])
        net_income   = self._get_val(inc, INCOME_STATEMENT_MAP['net_income'])
        depreciation = (
            self._get_val(cf,  INCOME_STATEMENT_MAP['depreciation']) or
            self._get_val(inc, INCOME_STATEMENT_MAP['depreciation']) or
            0.0
        )
        net_int_inc  = self._get_val(inc, INCOME_STATEMENT_MAP['net_interest_income'])

        if is_bank:
            if revenue is None or revenue == 0:
                nii  = self._get_val(inc, INCOME_STATEMENT_MAP['net_interest_income']) or 0
                nfee = self._get_val(inc, ['ii.net_fee_and_commission_income']) or 0
                nfx  = self._get_val(inc, ['iii.net_gain_loss_from_foreign_currencies_and_gold_trading']) or 0
                ntrd = self._get_val(inc, ['iv.net_gain_loss_from_trading_securities']) or 0
                ninv = self._get_val(inc, ['v.net_gain_loss_from_investment_securities']) or 0
                noth = self._get_val(inc, ['vi.net_other_income']) or 0
                ndiv = self._get_val(inc, ['vii.income_from_capital_contribution_and_long_term_investments']) or 0
                total_income = nii + nfee + nfx + ntrd + ninv + noth + ndiv
                if total_income > 0:
                    revenue = total_income
                    logger.debug(f"[{ticker}] Bank revenue = sum of income lines: {revenue:.0f}")
            if revenue is None or revenue == 0:
                op_rev = self._get_val(inc, INCOME_STATEMENT_MAP.get('total_operating_income', []))
                if op_rev and op_rev > 0:
                    revenue = op_rev

        # Ratio Table
        raw_roe  = self._get_val(ratio_df, RATIO_MAP['roe'])
        raw_roa  = self._get_val(ratio_df, RATIO_MAP['roa'])
        vnstock_roe = (raw_roe / 100.0) if raw_roe is not None else None
        vnstock_roa = (raw_roa / 100.0) if raw_roa is not None else None
        market_cap    = self._get_val(ratio_df, RATIO_MAP['market_cap'])
        price_to_book = self._get_val(ratio_df, RATIO_MAP['price_to_book'])

        # Computed features
        ebit = self._compute_ebit(ticker, bs, inc, is_bank)
        total_debt = long_term_debt + short_term_debt
        ebitda     = ((ebit or 0) + depreciation) if ebit is not None else None

        ROA  = self._safe_div(net_income, total_assets) or vnstock_roa
        ROE  = self._safe_div(net_income, total_equity) or vnstock_roe

        ROCE = self._compute_roce(
            ticker, bs, inc, ratio_df, is_bank,
            ebit, total_assets, current_liabilities, total_equity
        )

        EBIT_Margin = None
        em_direct = self._get_val(ratio_df, RATIO_MAP.get('ebit_margin_direct', ['ebit_margin']))
        if em_direct is not None:
            EBIT_Margin = em_direct / 100.0 if abs(em_direct) > 1.0 else em_direct
            logger.debug(f"[{ticker}] EBIT_Margin from ratio: {em_direct}")
        if EBIT_Margin is None and revenue and abs(revenue) > 1e-9 and ebit is not None:
            EBIT_Margin = self._safe_div(ebit, revenue)
        if EBIT_Margin is None and is_bank and net_int_inc and total_assets:
            EBIT_Margin = self._safe_div(net_int_inc, total_assets)  # NIM proxy
            logger.debug(f"[{ticker}] EBIT_Margin bank NIM proxy: {EBIT_Margin}")

        Debt_to_Assets  = self._safe_div(total_liabilities, total_assets)
        Debt_to_Equity  = self._safe_div(total_liabilities, total_equity)
        Net_Debt        = total_debt - cash
        Net_Debt_EBITDA = self._safe_div(Net_Debt, ebitda)

        if is_bank and net_int_inc and total_assets:
            nim = self._safe_div(net_int_inc, total_assets)
            if nim is not None:
                Net_Debt_EBITDA = nim

        Current_Ratio, _ca_resolved, _cl_resolved = self._compute_current_ratio(
            ticker, bs, ratio_df, is_bank,
            current_assets, current_liabilities,
            total_assets, total_equity, total_liabilities
        )

        if _ca_resolved is not None and _cl_resolved is not None and total_assets:
            wc = _ca_resolved - _cl_resolved
            WCTA = self._safe_div(wc, total_assets)
        else:
            WCTA = None

        Asset_Turnover = self._safe_div(revenue, total_assets)

        Revenue_Growth = None
        rg_ratio = self._get_val(ratio_df, ['net_revenue', 'Revenue YoY (%)'])
        if rg_ratio is not None and not np.isnan(rg_ratio):
            Revenue_Growth = float(np.clip(rg_ratio / 100.0, -0.9, 5.0))
            logger.debug(f"[{ticker}] Revenue_Growth from ratio (net_revenue %): {rg_ratio}")
        if Revenue_Growth is None and is_bank:
            for bk_col in ['net_interest_income', 'operating_profit_before_provision_for_credit_losses']:
                rg_bk = self._get_val(ratio_df, [bk_col])
                if rg_bk is not None and not np.isnan(rg_bk):
                    Revenue_Growth = float(np.clip(rg_bk / 100.0, -0.9, 5.0))
                    logger.debug(f"[{ticker}] Revenue_Growth bank proxy {bk_col}: {rg_bk}%")
                    break
        if Revenue_Growth is None:
            Revenue_Growth = self._get_growth(inc, INCOME_STATEMENT_MAP['revenue'])

        Log_Revenue = float(np.log1p(abs(revenue))) if revenue else None
        RETA           = self._safe_div(retained_earnings, total_assets)
        Market_to_Book = price_to_book
        if market_cap is None:
            market_cap = 0.0

        features = {
            'Ticker':             ticker,
            'Sector':             sector,
            'Company_Name':       _safe_str(company_info.get('company_name', ticker)),
            'Exchange':           _safe_str(company_info.get('exchange', '')),
            'ROA':                ROA,
            'ROE':                ROE,
            'ROCE':               ROCE,
            'EBIT_Margin':        EBIT_Margin,
            'Current_Ratio':      Current_Ratio,
            'Revenue_Growth_YoY': Revenue_Growth,
            'Log_Revenue':        Log_Revenue,
            'Market_Cap':         market_cap,
            'Debt/Assets':        Debt_to_Assets,
            'Debt/Equity':        Debt_to_Equity,
            'Net_Debt/EBITDA':    Net_Debt_EBITDA,
            'Asset_Turnover':     Asset_Turnover,
            'WCTA':               WCTA,
            'RETA':               RETA,
            'Market_to_Book':     Market_to_Book,
            '_Revenue':           revenue,
            '_Net_Income':        net_income,
            '_EBIT':              ebit,
            '_Total_Assets':      total_assets,
            '_Total_Equity':      total_equity,
            '_Total_Debt':        total_debt,
            '_Current_Assets':    current_assets,
            '_Current_Liabilities': current_liabilities,
            '_Is_Bank':           is_bank,
        }

        missing = [
            f for f in self.required_features
            if features.get(f) is None
            or (isinstance(features.get(f), float) and np.isnan(features[f]))
        ]

        if len(missing) > self.max_missing:
            logger.warning(
                f"[{ticker}] Thiếu {len(missing)}/{len(self.required_features)} features: "
                f"{missing}. Skip."
            )
            return None

        if missing:
            logger.info(f"[{ticker}] Missing (sẽ impute): {missing}")
        else:
            logger.info(f"[{ticker}] Tất cả features đã được tính từ BCTC thực tế")

        return features

    # ======================== KAGGLE ADAPTER ========================

    @staticmethod
    def adapt_kaggle_data(df: pd.DataFrame) -> pd.DataFrame:
        """Map Kaggle corporate_credit_rating.csv → format chuẩn."""
        KAGGLE_TO_SCRIPT = {
            'returnOnAssets':                     'ROA',
            'returnOnEquity':                     'ROE',
            'returnOnCapitalEmployed':            'ROCE',
            'operatingProfitMargin':              'EBIT_Margin',
            'ebitPerRevenue':                     'EBIT_Margin',
            'currentRatio':                       'Current_Ratio',
            'debtRatio':                          'Debt/Assets',
            'debtEquityRatio':                    'Debt/Equity',
            'assetTurnover':                      'Asset_Turnover',
            'companyEquityMultiplier':            'Market_to_Book',
            'netProfitMargin':                    '_Net_Profit_Margin',
            'grossProfitMargin':                  '_Gross_Margin',
            'freeCashFlowOperatingCashFlowRatio': '_FCF_OCF_Ratio',
        }
        KAGGLE_SECTOR_MAP = {
            'Banking': 'Banking', 'Finance': 'FinancialServices',
            'Insurance': 'Insurance', 'Technology': 'Technology',
            'Consumer Durables': 'Manufacturing', 'Consumer Discretionary': 'Retail',
            'Consumer Staples': 'Manufacturing', 'Healthcare': 'Healthcare',
            'Energy': 'Energy', 'Materials': 'Manufacturing',
            'Industrials': 'Manufacturing', 'Real Estate': 'RealEstate',
            'Utilities': 'Energy', 'Communication Services': 'Telecom',
        }
        df = df.copy()
        df = df.rename(columns=KAGGLE_TO_SCRIPT)
        if 'Rating' in df.columns:
            df = df.rename(columns={'Rating': 'Credit_Rating'})
        if 'Sector' in df.columns:
            df['Sector'] = df['Sector'].map(KAGGLE_SECTOR_MAP).fillna(df['Sector'])
        if 'Ticker' not in df.columns:
            if 'Symbol' in df.columns:
                df['Ticker'] = df['Symbol']
            elif 'Name' in df.columns:
                df['Ticker'] = df['Name'].str[:6].str.upper().str.replace(' ', '')
            else:
                df['Ticker'] = [f'US{i:04d}' for i in range(len(df))]
        if 'Revenue_Growth_YoY' not in df.columns:
            df['Revenue_Growth_YoY'] = 0.08
        if 'Log_Revenue' not in df.columns:
            if 'cashPerShare' in df.columns:
                df['Log_Revenue'] = np.log1p(df['cashPerShare'].abs())
            else:
                df['Log_Revenue'] = 8.0
        if 'Market_Cap' not in df.columns:
            df['Market_Cap'] = 1e9
        if 'WCTA' not in df.columns and 'Current_Ratio' in df.columns:
            df['WCTA'] = (df['Current_Ratio'] - 1) / df['Current_Ratio'].clip(lower=0.1) * 0.3
        if 'RETA' not in df.columns and 'ROE' in df.columns:
            df['RETA'] = df['ROE'] * 0.5
        if 'Net_Debt/EBITDA' not in df.columns:
            if 'Debt/Assets' in df.columns and 'EBIT_Margin' in df.columns:
                df['Net_Debt/EBITDA'] = df['Debt/Assets'] / df['EBIT_Margin'].clip(lower=0.01)
            else:
                df['Net_Debt/EBITDA'] = 3.0
        return df


# ======================== SECTOR ADJUSTER ========================

class SectorAdjuster:
    SECTOR_MEDIANS = {
        'Banking':      {'ROA': 0.015, 'ROE': 0.14, 'ROCE': 0.10,
                         'Debt/Assets': 0.88, 'Debt/Equity': 12.0,
                         'Current_Ratio': 0.85, 'EBIT_Margin': 0.28,
                         'Asset_Turnover': 0.10, 'Net_Debt/EBITDA': 0.03,
                         'WCTA': 0.05, 'RETA': 0.05},
        'RealEstate':   {'ROA': 0.04, 'ROE': 0.09, 'ROCE': 0.07,
                         'Debt/Assets': 0.65, 'Debt/Equity': 2.5,
                         'Current_Ratio': 1.2, 'EBIT_Margin': 0.20,
                         'Asset_Turnover': 0.28, 'Net_Debt/EBITDA': 5.5,
                         'WCTA': 0.10, 'RETA': 0.15},
        'Manufacturing':{'ROA': 0.06, 'ROE': 0.12, 'ROCE': 0.09,
                         'Debt/Assets': 0.45, 'Debt/Equity': 1.0,
                         'Current_Ratio': 1.8, 'EBIT_Margin': 0.09,
                         'Asset_Turnover': 1.0, 'Net_Debt/EBITDA': 2.5,
                         'WCTA': 0.25, 'RETA': 0.25},
        'Technology':   {'ROA': 0.09, 'ROE': 0.16, 'ROCE': 0.13,
                         'Debt/Assets': 0.20, 'Debt/Equity': 0.3,
                         'Current_Ratio': 2.5, 'EBIT_Margin': 0.18,
                         'Asset_Turnover': 0.85, 'Net_Debt/EBITDA': 0.8,
                         'WCTA': 0.35, 'RETA': 0.25},
        'Retail':       {'ROA': 0.07, 'ROE': 0.15, 'ROCE': 0.11,
                         'Debt/Assets': 0.40, 'Debt/Equity': 0.8,
                         'Current_Ratio': 1.5, 'EBIT_Margin': 0.05,
                         'Asset_Turnover': 1.8, 'Net_Debt/EBITDA': 2.0,
                         'WCTA': 0.20, 'RETA': 0.22},
        'Energy':       {'ROA': 0.05, 'ROE': 0.11, 'ROCE': 0.08,
                         'Debt/Assets': 0.50, 'Debt/Equity': 1.2,
                         'Current_Ratio': 1.5, 'EBIT_Margin': 0.15,
                         'Asset_Turnover': 0.5, 'Net_Debt/EBITDA': 3.0,
                         'WCTA': 0.18, 'RETA': 0.20},
    }
    DEFAULT_MEDIAN = SECTOR_MEDIANS['Manufacturing']

    def impute_missing(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        features = list(self.DEFAULT_MEDIAN.keys())
        for feat in features:
            if feat not in df.columns:
                continue
            mask = df[feat].isna()
            if not mask.any():
                continue
            for sector, medians in self.SECTOR_MEDIANS.items():
                sm = (df['Sector'] == sector) & mask
                if sm.any() and feat in medians:
                    df.loc[sm, feat] = medians[feat]
            if df[feat].isna().any():
                gm = df[feat].median()
                df[feat] = df[feat].fillna(gm if not pd.isna(gm)
                                           else self.DEFAULT_MEDIAN.get(feat, 0.0))
        return df


# ======================== UTILITY ========================

def map_kaggle_csv(input_path: str, output_path: str) -> pd.DataFrame:
    logger.info(f"Mapping Kaggle data: {input_path}")
    df = pd.read_csv(input_path)
    df_mapped = VNDataAdapter.adapt_kaggle_data(df)
    if 'Ticker' in df_mapped.columns and 'Credit_Rating' in df_mapped.columns:
        df_mapped = df_mapped.sort_values('Ticker')
        df_mapped = df_mapped.drop_duplicates(
            subset=['Ticker', 'Credit_Rating'], keep='first'
        )
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    df_mapped.to_csv(out, index=False)
    logger.info(f"Saved: {out} ({len(df_mapped)} rows)")
    return df_mapped


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--map_kaggle', type=str)
    parser.add_argument('--output', type=str, default='data/processed/kaggle_mapped.csv')
    args = parser.parse_args()
    if args.map_kaggle:
        map_kaggle_csv(args.map_kaggle, args.output)
    else:
        print("Usage: python vn_data_adapter.py --map_kaggle corporate_credit_rating.csv")
