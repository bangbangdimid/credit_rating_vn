"""
crawl_vn_rated.py  v2
=====================
Cào BCTC cho tất cả công ty trong vn_rated.csv.

FIXES v2:
  - KHÔNG dùng lang= kwarg (KBS không hỗ trợ → lỗi)
  - Cache dùng CSV thay vì parquet (không cần pyarrow)
  - Rate limit: 20 req/phút (Guest) → delay đúng
      4 calls/ticker × delay = không vượt limit
  - Auto-wait 55s khi bị rate limit, tự retry

Cách chạy:
    python crawl_vn_rated.py

Output:
    data/raw/vn_rated_with_financials.csv   ← dùng cho finetune
    data/processed/vn_firms_crawled.csv     ← dùng cho pipeline

Yêu cầu:
    pip install vnstock pandas numpy
"""

import sys
import os
import time
import logging
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, Dict, List

# ── Paths ──────────────────────────────────────────────────────────
ROOT         = Path(__file__).parent
VN_RATED_CSV = ROOT / "data/raw/vn_rated.csv"
OUTPUT_RAW   = ROOT / "data/raw/vn_rated_with_financials.csv"
OUTPUT_PROC  = ROOT / "data/processed/vn_firms_crawled.csv"
CACHE_DIR    = ROOT / "data/cache"

for d in [OUTPUT_RAW.parent, OUTPUT_PROC.parent, CACHE_DIR, ROOT / "logs"]:
    d.mkdir(parents=True, exist_ok=True)

# ── Logging ────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("logs/crawl_vn_rated.log", encoding="utf-8"),
    ]
)
logger = logging.getLogger(__name__)

# ── Rate limit (KBS Guest: 20 req/phút) ───────────────────────────
# Mỗi ticker cần 4 calls (BS + IS + CF + Ratio)
# 20 req/60s = 3s/req → 4 calls × 3s = 12s/ticker tối thiểu
# Dùng 16s để có buffer, không bao giờ vượt limit
INTER_CALL_DELAY = 4.0    # giây giữa mỗi API call trong 1 ticker
INTER_TICKER_DELAY = 5.0  # giây thêm sau khi xong 1 ticker
RATE_LIMIT_WAIT  = 60.0   # giây chờ khi bị rate limit (thông báo chờ 50s → dùng 60 cho chắc)
MAX_RETRIES      = 3


# ── Sector mapping ─────────────────────────────────────────────────
TICKER_SECTOR = {
    "ACB":"Banking","BAB":"Banking","HDB":"Banking","LPB":"Banking",
    "MBB":"Banking","MSB":"Banking","NAB":"Banking","OCB":"Banking",
    "TPB":"Banking","VAB":"Banking","VCB":"Banking","VIB":"Banking",
    "ABB":"Banking","TCX":"Banking",
    "APG":"FinancialServices","BSI":"FinancialServices","DSC":"FinancialServices",
    "MBS":"FinancialServices","ORS":"FinancialServices","PSI":"FinancialServices",
    "SHS":"FinancialServices","TVS":"FinancialServices","VCI":"FinancialServices",
    "VCK":"FinancialServices","VDS":"FinancialServices","VND":"FinancialServices",
    "F88":"FinancialServices","TIN":"FinancialServices",
    "AGG":"RealEstate","BCG":"RealEstate","BCM":"RealEstate","KDH":"RealEstate",
    "NLG":"RealEstate","SGR":"RealEstate","VHM":"RealEstate","VIC":"RealEstate",
    "VPI":"RealEstate",
    "CTD":"Manufacturing","GEX":"Manufacturing","HUT":"Manufacturing",
    "HDG":"Manufacturing","CMX":"Manufacturing",
    "GEG":"Energy","VCP":"Energy",
}


# ==================================================================
# NORMALIZE KBS RESPONSE
# ==================================================================

def _normalize_kbs(df: pd.DataFrame) -> pd.DataFrame:
    """
    KBS trả về dạng: rows=items, cols=[item, item_id, 2025, 2024, ...]
    Pivot thành: rows=năm (mới nhất trước), cols=item_id values.
    """
    if df is None or df.empty:
        return df
    id_col = next((c for c in ["item_id", "item"] if c in df.columns), None)
    if id_col is None:
        return df  # không phải KBS format → trả về nguyên
    meta = {"item", "item_id", "item_en", "unit", "levels", "row_number"}
    year_cols = [c for c in df.columns if str(c) not in meta]
    if not year_cols:
        return df

    def _sort_key(c):
        s = str(c)
        if s.isdigit():
            return int(s)
        if "-Q" in s:
            p = s.split("-Q")
            try: return int(p[0]) * 10 + int(p[1])
            except: return 0
        return 0

    year_cols = sorted(year_cols, key=_sort_key, reverse=True)
    records = []
    for yc in year_cols:
        row = {"year": str(yc)}
        for _, r in df.iterrows():
            key = str(r.get(id_col, "")).strip()
            if not key or key == "nan":
                continue
            val = r.get(yc)
            if val is not None and not (isinstance(val, float) and val != val):
                row[key] = val
        records.append(row)
    return pd.DataFrame(records).reset_index(drop=True) if records else df


# ==================================================================
# CRAWLER
# ==================================================================

class VNRatedCrawler:

    def __init__(self):
        try:
            from vnstock import Vnstock
            self._Vnstock = Vnstock
            logger.info("vnstock import OK (source=KBS)")
        except ImportError:
            logger.error("Chưa cài vnstock: pip install vnstock")
            sys.exit(1)

    # ── Gọi 1 finance method an toàn ──────────────────────────────

    def _call(self, fn, period="year") -> Optional[pd.DataFrame]:
        """
        Gọi fn(period=period) — KHÔNG truyền lang.
        Tự động chờ nếu rate limit.
        """
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                df = fn(period=period)
                time.sleep(INTER_CALL_DELAY)  # delay sau mỗi call
                if df is not None and not df.empty:
                    return _normalize_kbs(df)
                return None
            except Exception as e:
                err = str(e)
                if "Rate limit" in err or "20/20" in err or "GIỚI HẠN" in err or "429" in err:
                    logger.warning(f"    Rate limit → chờ {RATE_LIMIT_WAIT}s...")
                    time.sleep(RATE_LIMIT_WAIT)
                    # Không tính vào attempt
                elif attempt < MAX_RETRIES:
                    logger.debug(f"    Retry {attempt}: {err[:80]}")
                    time.sleep(INTER_CALL_DELAY)
                else:
                    logger.debug(f"    Failed: {err[:80]}")
                    return None
        return None

    # ── Lấy 1 giá trị từ DataFrame ────────────────────────────────

    def _get(self, df: Optional[pd.DataFrame], keys: List[str],
             idx: int = 0) -> Optional[float]:
        if df is None or df.empty or idx >= len(df):
            return None
        for k in keys:
            if k in df.columns:
                val = df.iloc[idx][k]
                if pd.notna(val):
                    try: return float(val)
                    except: continue
        # fuzzy — tìm cột chứa từ khóa
        for k in keys:
            kl = k.lower()
            for col in df.columns:
                if kl in str(col).lower():
                    val = df.iloc[idx][col]
                    if pd.notna(val):
                        try: return float(val)
                        except: continue
        return None

    # ── Crawl 1 ticker ─────────────────────────────────────────────

    def fetch(self, ticker: str) -> Optional[Dict]:
        # Check cache (CSV)
        cache = CACHE_DIR / f"{ticker}.csv"
        if cache.exists():
            try:
                df_c = pd.read_csv(cache)
                if not df_c.empty:
                    logger.info(f"  {ticker}: cache hit")
                    return df_c.iloc[0].to_dict()
            except Exception:
                cache.unlink(missing_ok=True)

        stock = self._Vnstock().stock(symbol=ticker, source="KBS")
        finance = stock.finance

        bs    = self._call(finance.balance_sheet)
        inc   = self._call(finance.income_statement)
        cf    = self._call(finance.cash_flow)
        ratio = self._call(finance.ratio)

        g = self._get  # shorthand

        # ── Balance sheet ─────────────────────────────────────────
        total_assets = g(bs,["total_assets","total_liabilities_and_owners_equity",
                              "total_owners_equity_and_liabilities","assets",
                              "TOTAL ASSETS (Bn. VND)","totalAssets"])
        total_liab   = g(bs,["total_liabilities","a.liabilities","liabilities",
                              "LIABILITIES (Bn. VND)","totalLiabilities"])
        total_equity = g(bs,["b.owners_equity","viii.capital_and_reserves",
                              "i.owners_equity","total_equity","equity",
                              "OWNER'S EQUITY(Bn.VND)","totalStockholdersEquity"])
        cur_assets   = g(bs,["a.short_term_assets","current_assets",
                              "SHORT-TERM ASSETS (Bn. VND)","currentAssets"])
        cur_liab     = g(bs,["i.short_term_liabilities","current_liabilities",
                              "SHORT-TERM LIABILITIES (Bn. VND)","currentLiabilities"])
        lt_debt      = g(bs,["long_term_debt","long_term_loans","longTermDebt",
                              "c.long_term_liabilities","ii.long_term_liabilities"])
        st_debt      = g(bs,["short_term_loans","short_term_debt","shortTermDebt",
                              "i.2.short_term_loans"])
        retained     = g(bs,["retained_earnings","retainedEarnings",
                              "iv.undistributed_profit_after_tax","undistributed_profit"])
        cash         = g(bs,["cash","cash_and_cash_equivalents",
                              "i.cash_and_cash_equivalents",
                              "i.cash_gold_and_silver_precious_stones"])
        prev_equity  = g(bs,["b.owners_equity","viii.capital_and_reserves",
                              "total_equity","equity"], idx=1)

        # ── Income statement ──────────────────────────────────────
        revenue      = g(inc,["revenue","net_revenue","NET REVENUE (Bn. VND)",
                               "net_sales","1.revenue",
                               "i.1.revenues_from_sales_of_goods"])
        ebit         = g(inc,["ebit","EBIT (Bn. VND)","operating_profit",
                               "3.operating_profit"])
        net_profit   = g(inc,["net_profit","net_income","NET PROFIT (Bn. VND)",
                               "profit_after_tax","net_profit_after_tax",
                               "9.profit_after_tax","15.profit_after_tax"])
        gross_profit = g(inc,["gross_profit","GROSS PROFIT (Bn. VND)",
                               "3.gross_profit","2.gross_profit_on_sales"])
        prev_rev     = g(inc,["revenue","net_revenue","NET REVENUE (Bn. VND)",
                               "net_sales","1.revenue"], idx=1)

        # ── Ratio (có thể đã có ROA/ROE trực tiếp) ───────────────
        roa_direct = g(ratio,["roa","ROA","return_on_assets"])
        roe_direct = g(ratio,["roe","ROE","return_on_equity"])

        # ── Tính features ─────────────────────────────────────────
        def div(a, b):
            if a is None or b is None or b == 0: return None
            return a / b

        # ROA / ROE — từ ratio nếu có, tính từ BCTC nếu không
        roa = roa_direct
        if roa is None: roa = div(net_profit, total_assets)
        if roa is not None and abs(roa) > 1: roa /= 100  # ratio API trả %

        roe = roe_direct
        if roe is None: roe = div(net_profit, total_equity)
        if roe is not None and abs(roe) > 1: roe /= 100

        # ROCE = EBIT / Capital Employed
        cap_emp = (total_assets - cur_liab) if (total_assets and cur_liab) else None
        roce = div(ebit, cap_emp)

        # EBIT Margin
        ebit_margin = div(ebit, revenue)

        # Current Ratio
        current_ratio = div(cur_assets, cur_liab)

        # Debt/Assets, Debt/Equity
        total_debt = None
        if lt_debt is not None and st_debt is not None:
            total_debt = lt_debt + st_debt
        elif total_liab is not None:
            total_debt = total_liab
        debt_assets  = div(total_debt, total_assets) or div(total_liab, total_assets)
        debt_equity  = div(total_debt, total_equity) or div(total_liab, total_equity)

        # Net Debt / EBITDA
        ebitda = None
        if ebit is not None:
            dep_est = max(0, (gross_profit or ebit) - ebit)
            ebitda = ebit + dep_est * 0.15 if dep_est == 0 else ebit + dep_est
        net_debt = (total_debt - cash) if (total_debt and cash) else total_debt
        net_debt_ebitda = div(net_debt, ebitda)

        # Asset Turnover
        asset_turnover = div(revenue, total_assets)

        # WCTA
        wcta = None
        if cur_assets and cur_liab and total_assets:
            wcta = (cur_assets - cur_liab) / total_assets

        # RETA
        reta = None
        if retained and total_assets:
            reta = retained / total_assets
        elif net_profit and prev_equity and total_assets:
            reta = (net_profit + prev_equity) / total_assets

        # Revenue Growth
        rev_growth = div((revenue or 0) - (prev_rev or 0), abs(prev_rev or 1)) \
                     if (revenue and prev_rev and prev_rev != 0) else None

        # Log Revenue
        log_rev = float(np.log(revenue)) if (revenue and revenue > 0) else None

        features = {
            "Ticker":             ticker,
            "Sector":             TICKER_SECTOR.get(ticker, "Other"),
            "ROA":                roa,
            "ROE":                roe,
            "ROCE":               roce,
            "EBIT_Margin":        ebit_margin,
            "Current_Ratio":      current_ratio,
            "Debt/Assets":        debt_assets,
            "Debt/Equity":        debt_equity,
            "Net_Debt/EBITDA":    net_debt_ebitda,
            "Asset_Turnover":     asset_turnover,
            "WCTA":               wcta,
            "RETA":               reta,
            "Market_to_Book":     None,   # cần market cap
            "Revenue_Growth_YoY": rev_growth,
            "Log_Revenue":        log_rev,
            "Market_Cap":         None,
        }

        n_ok = sum(1 for k, v in features.items()
                   if k not in ("Ticker","Sector") and v is not None)

        if n_ok < 4:
            logger.warning(f"  {ticker}: chỉ {n_ok}/15 features → skip")
            return None

        logger.info(f"  {ticker}: {n_ok}/15 features OK")

        # Lưu cache CSV
        pd.DataFrame([features]).to_csv(cache, index=False, encoding="utf-8-sig")
        return features

    # ── Main crawl loop ─────────────────────────────────────────────

    def crawl_all(self) -> pd.DataFrame:
        df_rated = pd.read_csv(VN_RATED_CSV)
        df_rated["date"] = pd.to_datetime(df_rated["date"], dayfirst=True, errors="coerce")
        df_rated = (df_rated.sort_values("date", ascending=False)
                             .drop_duplicates(subset=["ticker"], keep="first")
                             .reset_index(drop=True))
        tickers    = df_rated["ticker"].tolist()
        rating_map = dict(zip(df_rated["ticker"], df_rated["credit_rating"]))

        logger.info(f"Sẽ crawl {len(tickers)} tickers: {tickers}")
        logger.info(f"Rate limit KBS Guest: 20 req/phút")
        logger.info(f"Delay: {INTER_CALL_DELAY}s/call + {INTER_TICKER_DELAY}s/ticker")
        logger.info(f"Thời gian ước tính: ~{len(tickers) * (INTER_CALL_DELAY*4 + INTER_TICKER_DELAY) / 60:.0f} phút\n")

        results, failed = [], []

        for i, ticker in enumerate(tickers, 1):
            logger.info(f"[{i:2d}/{len(tickers)}] {ticker}...")
            row = self.fetch(ticker)
            if row:
                row["credit_rating"] = rating_map.get(ticker, "")
                results.append(row)
            else:
                failed.append(ticker)

            # Delay giữa các ticker
            if i < len(tickers):
                time.sleep(INTER_TICKER_DELAY)

        logger.info(f"\n{'='*60}")
        logger.info(f"Xong: {len(results)}/{len(tickers)} thành công")
        if failed:
            logger.warning(f"Thất bại: {failed}")

        if not results:
            return pd.DataFrame()

        df = pd.DataFrame(results)
        col_order = ["Ticker","Sector","credit_rating",
                     "ROA","ROE","ROCE","EBIT_Margin","Current_Ratio",
                     "Debt/Assets","Debt/Equity","Net_Debt/EBITDA",
                     "Asset_Turnover","WCTA","RETA",
                     "Market_to_Book","Revenue_Growth_YoY","Log_Revenue","Market_Cap"]
        df = df[[c for c in col_order if c in df.columns]]
        return df


# ==================================================================
# MAIN
# ==================================================================

def main():
    logger.info("=" * 60)
    logger.info("VN RATED COMPANIES — BCTC CRAWLER v2")
    logger.info("=" * 60)

    if not VN_RATED_CSV.exists():
        logger.error(f"Không tìm thấy: {VN_RATED_CSV}")
        sys.exit(1)

    crawler = VNRatedCrawler()
    df = crawler.crawl_all()

    if df.empty:
        logger.error("Không có dữ liệu nào!")
        sys.exit(1)

    df.to_csv(OUTPUT_RAW,  index=False, encoding="utf-8-sig")
    df.to_csv(OUTPUT_PROC, index=False, encoding="utf-8-sig")
    logger.info(f"✓ Saved: {OUTPUT_RAW}")
    logger.info(f"✓ Saved: {OUTPUT_PROC}")

    # Summary
    feat_cols = ["ROA","ROE","ROCE","EBIT_Margin","Current_Ratio",
                 "Debt/Assets","Debt/Equity","Net_Debt/EBITDA","Asset_Turnover","WCTA","RETA"]
    present = [c for c in feat_cols if c in df.columns]
    print(f"\n{'='*60}")
    print(f"  Firms crawled   : {len(df)}")
    print(f"  Features        : {len(present)}/{len(feat_cols)}")
    print(f"\n  Null % per feature:")
    for col in present:
        pct = df[col].isna().mean()
        s = "✓" if pct < 0.3 else ("⚠" if pct < 0.6 else "✗")
        print(f"    {s} {col:<22}: {pct*100:.0f}% null")
    print(f"\n  Rating distribution:")
    print(df["credit_rating"].value_counts().to_string())
    print(f"\n✅ File output: {OUTPUT_RAW}")
    print(f"\n  Bước tiếp theo:")
    print(f"    python main.py train")
    print(f"    python main.py finetune --rated {OUTPUT_RAW}")
    print("=" * 60)


if __name__ == "__main__":
    main()
