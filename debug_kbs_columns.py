"""
debug_kbs_columns.py
====================
In ra toàn bộ columns và giá trị thực tế sau khi _normalize_kbs_df()
để biết chính xác tên cột KBS → fix column maps trong vn_data_adapter.py

Usage:
    python debug_kbs_columns.py
    python debug_kbs_columns.py --ticker VNM   # non-bank để so sánh
"""
import argparse
import pandas as pd
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

def print_section(title):
    print(f"\n{'#'*60}\n  {title}\n{'#'*60}")

def normalize_kbs_df(df, stmt_type=''):
    """Copy của _normalize_kbs_df từ vn_crawler.py"""
    if df is None or df.empty:
        return None
    id_col = 'item_id' if 'item_id' in df.columns else ('item' if 'item' in df.columns else None)
    if id_col is None:
        return None
    meta_cols = {'item', 'item_id', 'item_en', 'unit', 'levels', 'row_number'}
    year_cols = [c for c in df.columns if str(c) not in meta_cols]
    if not year_cols:
        return None

    def _yk(c):
        s = str(c)
        if s.isdigit(): return int(s)
        if '-Q' in s:
            p = s.split('-Q')
            try: return int(p[0])*10 + int(p[1])
            except: return 0
        return 0

    year_cols_sorted = sorted(year_cols, key=_yk, reverse=True)
    records = []
    for yc in year_cols_sorted:
        row = {'year': str(yc)}
        for _, item_row in df.iterrows():
            key = str(item_row.get(id_col, '')).strip()
            if not key or key == 'nan':
                continue
            val = item_row.get(yc)
            if val is not None and not (isinstance(val, float) and val != val):
                row[key] = val
        records.append(row)
    if not records:
        return None
    return pd.DataFrame(records).reset_index(drop=True)


def print_wide_df(df, label, top_n=None):
    """In wide DataFrame: mỗi column_name và giá trị năm mới nhất"""
    if df is None or df.empty:
        print(f"  {label}: None / empty")
        return
    print(f"\n  {label} — shape={df.shape}")
    print(f"  Năm có dữ liệu: {list(df['year']) if 'year' in df.columns else 'N/A'}")
    print(f"\n  {'Column name':<60} {'Năm mới nhất':>20}")
    print(f"  {'-'*60} {'-'*20}")
    row = df.iloc[0]
    cols = [c for c in df.columns if c != 'year']
    if top_n:
        cols = cols[:top_n]
    for col in cols:
        val = row.get(col, '')
        if isinstance(val, float):
            val_str = f"{val:,.2f}" if abs(val) < 1e15 else f"{val:.3e}"
        else:
            val_str = str(val)
        print(f"  {col:<60} {val_str:>20}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ticker', default='VCB')
    args = parser.parse_args()
    ticker = args.ticker.upper()

    print(f"\n{'#'*60}")
    print(f"  KBS Column Debug | ticker={ticker}")
    print(f"{'#'*60}")

    from vnstock import Vnstock
    stock = Vnstock().stock(symbol=ticker, source='KBS')

    # ── balance_sheet ─────────────────────────────────────────────
    print_section(f"balance_sheet — raw item_id list")
    try:
        raw_bs = stock.finance.balance_sheet(period='year')
        print(f"  Raw shape: {raw_bs.shape}")
        print(f"  Raw columns: {list(raw_bs.columns)}")
        print("\n  All item_id values:")
        for _, row in raw_bs.iterrows():
            iid = str(row.get('item_id', '')).strip()
            itm = str(row.get('item', '')).strip()
            if iid:
                print(f"    {iid:<60}  ({itm})")
        wide_bs = normalize_kbs_df(raw_bs, 'balance')
        print_wide_df(wide_bs, "balance_sheet WIDE")
    except Exception as e:
        print(f"  ERROR: {e}")

    # ── income_statement ──────────────────────────────────────────
    print_section(f"income_statement — raw item_id list")
    try:
        raw_inc = stock.finance.income_statement(period='year')
        print(f"  Raw shape: {raw_inc.shape}")
        print("\n  All item_id values:")
        for _, row in raw_inc.iterrows():
            iid = str(row.get('item_id', '')).strip()
            itm = str(row.get('item', '')).strip()
            if iid:
                print(f"    {iid:<60}  ({itm})")
        wide_inc = normalize_kbs_df(raw_inc, 'income')
        print_wide_df(wide_inc, "income_statement WIDE")
    except Exception as e:
        print(f"  ERROR: {e}")

    # ── cash_flow ─────────────────────────────────────────────────
    print_section(f"cash_flow — raw item_id list")
    try:
        raw_cf = stock.finance.cash_flow(period='year')
        print(f"  Raw shape: {raw_cf.shape}")
        print("\n  All item_id values:")
        for _, row in raw_cf.iterrows():
            iid = str(row.get('item_id', '')).strip()
            itm = str(row.get('item', '')).strip()
            if iid:
                print(f"    {iid:<60}  ({itm})")
        wide_cf = normalize_kbs_df(raw_cf, 'cashflow')
        print_wide_df(wide_cf, "cash_flow WIDE")
    except Exception as e:
        print(f"  ERROR: {e}")

    # ── ratio ─────────────────────────────────────────────────────
    print_section(f"ratio — raw item_id list")
    try:
        raw_ratio = stock.finance.ratio(period='year')
        print(f"  Raw shape: {raw_ratio.shape}")
        print("\n  All item_id values:")
        for _, row in raw_ratio.iterrows():
            iid = str(row.get('item_id', '')).strip()
            itm = str(row.get('item', '')).strip()
            if iid:
                print(f"    {iid:<60}  ({itm})")
        wide_ratio = normalize_kbs_df(raw_ratio, 'ratio')
        print_wide_df(wide_ratio, "ratio WIDE")
    except Exception as e:
        print(f"  ERROR: {e}")

    # ── Diagnose missing features ──────────────────────────────────
    print_section("Diagnose: Tại sao EBIT_Margin / Revenue / Asset_Turnover bị miss?")
    try:
        wide_inc = normalize_kbs_df(stock.finance.income_statement(period='year'), 'income')
        wide_bs  = normalize_kbs_df(stock.finance.balance_sheet(period='year'), 'balance')

        if wide_inc is not None:
            cols_inc = [c for c in wide_inc.columns if c != 'year']
            # Tìm revenue candidates
            rev_keywords = ['revenue', 'income', 'net', 'total', 'operating']
            print("\n  Income cols chứa keyword revenue/income/net/total/operating:")
            for c in cols_inc:
                if any(kw in c.lower() for kw in rev_keywords):
                    val = wide_inc.iloc[0].get(c, '')
                    print(f"    {c:<60} = {val}")

        if wide_bs is not None:
            cols_bs = [c for c in wide_bs.columns if c != 'year']
            asset_keywords = ['asset', 'total']
            print("\n  Balance cols chứa keyword asset/total:")
            for c in cols_bs:
                if any(kw in c.lower() for kw in asset_keywords):
                    val = wide_bs.iloc[0].get(c, '')
                    print(f"    {c:<60} = {val}")

    except Exception as e:
        print(f"  ERROR: {e}")

    print("\n\n" + "="*60)
    print("  DONE — Paste output này cho Claude")
    print("="*60)

if __name__ == '__main__':
    main()
