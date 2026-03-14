"""
vn_crawler.py
=============
Thu thập BCTC của các doanh nghiệp niêm yết VN qua vnstock.

FIX v6 (2026-03) — vnstock 3.4.2:
  - [CRITICAL] source VCI → KBS (VCI server trả {} rỗng, chết hoàn toàn)
  - [CRITICAL] KBS finance API không nhận lang= và dropna= → bỏ hoàn toàn
  - [CRITICAL] KBS trả về DataFrame dạng transposed:
      rows = items, cols = [item, item_id, 2025, 2024, 2023, 2022]
    → _normalize_kbs_df() pivot thành wide (1 row/năm, cols=item_id)
  - TCBS deprecated 401 → loại khỏi choices
  - Giữ nguyên rate limit / retry / batch logic
"""

import pandas as pd
import numpy as np
import logging
import time
import sys
from pathlib import Path
from typing import List, Optional, Dict

logger = logging.getLogger(__name__)

# ================================================================
# Pandas compat: applymap đổi tên thành map trong pandas >= 2.1
# ================================================================
if not hasattr(pd.DataFrame, 'applymap'):
    pd.DataFrame.applymap = pd.DataFrame.map

# ================================================================
# Rate limit config
# ================================================================
REQUEST_DELAY   = 4.0
BATCH_PAUSE     = 70.0
BATCH_SIZE      = 4
RATE_LIMIT_WAIT = 70.0
MAX_RETRIES     = 3
DEFAULT_SOURCE  = 'KBS'   # VCI: server chết; TCBS: deprecated 401


# ================================================================
# KBS DataFrame normalizer
# ================================================================

def _normalize_kbs_df(df, ticker='', stmt_type=''):
    """
    KBS trả về: rows=items, cols=['item','item_id','2025','2024',...]
    Pivot thành: rows=năm (mới nhất idx 0), cols=item_id values.
    """
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


# ================================================================
# HELPERS
# ================================================================

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
        df.columns = [
            '_'.join(str(x) for x in c if str(x).strip())
            if isinstance(c, tuple) else str(c)
            for c in df.columns
        ]
    return df


def _is_rate_limit_error(e: Exception) -> bool:
    err_str = str(e).lower()
    return any(kw in err_str for kw in [
        'rate limit', 'ratelimit', 'rate_limit',
        '429', 'too many requests', 'giới hạn',
        'exceeded', 'quota',
    ])


def _safe_call(func, *args, **kwargs):
    original_exit = sys.exit
    rate_limit_triggered = [False]

    def fake_exit(code=0):
        rate_limit_triggered[0] = True
        raise SystemExit(code)

    sys.exit = fake_exit
    try:
        result = func(*args, **kwargs)
        sys.exit = original_exit
        return result, None
    except SystemExit:
        sys.exit = original_exit
        if rate_limit_triggered[0]:
            return None, 'rate_limit'
        return None, 'system_exit'
    except Exception as e:
        sys.exit = original_exit
        if _is_rate_limit_error(e):
            return None, 'rate_limit'
        return None, str(e)
    finally:
        sys.exit = original_exit


# ================================================================
# SECTOR MAP — tải một lần từ listing.symbols_by_industries()
# ================================================================

def _load_sector_map(source: str = DEFAULT_SOURCE) -> Dict[str, str]:
    """
    Tải map {ticker → industry_name} từ vnstock.
    Trả về dict rỗng nếu không kết nối được (crawler vẫn chạy,
    sector sẽ là 'Other' — có thể fix sau bằng cách chạy lại).
    """
    try:
        from vnstock import Vnstock
        stock = Vnstock().stock(symbol='VCB', source=source)
        raw = stock.listing.symbols_by_industries()

        # KBS returns DataFrame directly; guard against dict wrapping
        if isinstance(raw, dict):
            df = raw.get('data', raw.get('result', None))
            if df is None:
                for v in raw.values():
                    if isinstance(v, pd.DataFrame):
                        df = v
                        break
        else:
            df = raw

        if df is None or (hasattr(df, 'empty') and df.empty):
            logger.warning("symbols_by_industries() trả về rỗng")
            return {}

        df = _flatten_df(df)

        # Tìm cột symbol và industry_name
        sym_col = next(
            (c for c in df.columns if c.lower() in ('symbol', 'ticker', 'ma', 'mã')),
            None
        )
        ind_col = next(
            (c for c in df.columns if 'industry_name' in c.lower()
             or c.lower() in ('industry_name', 'ten_nganh', 'nganh', 'icb_name', 'industry')),
            None
        )

        # Fallback: nếu không tìm được tên cột đúng, thử vị trí
        if sym_col is None:
            sym_col = df.columns[0]
        if ind_col is None:
            # Tìm cột nào không phải số (likely là tên ngành)
            for c in df.columns[1:]:
                if df[c].dtype == object:
                    ind_col = c
                    break

        if sym_col is None or ind_col is None:
            logger.warning(f"Không xác định được cột symbol/industry. Columns: {list(df.columns)}")
            return {}

        sector_map = {
            str(row[sym_col]).upper(): str(row[ind_col]).strip()
            for _, row in df.iterrows()
            if pd.notna(row[ind_col]) and str(row[ind_col]).strip() not in ('', 'nan', 'None')
        }

        logger.info(f"✓ Sector map: {len(sector_map)} tickers từ symbols_by_industries()")
        # Log sample để verify
        sample = {k: v for k, v in list(sector_map.items())[:5]}
        logger.info(f"  Sample: {sample}")
        return sector_map

    except Exception as e:
        logger.warning(f"Không tải được sector map: {e}")
        logger.warning("  → Sector sẽ là 'Other' cho tất cả tickers. Bỏ qua nếu OK.")
        return {}


# ================================================================
# CRAWLER
# ================================================================

class VNStockCrawler:

    def __init__(self, source: str = DEFAULT_SOURCE):
        self.source = source
        self._check_vnstock()
        # Load sector map một lần duy nhất
        logger.info("Đang tải sector map từ listing.symbols_by_industries()...")
        self._sector_map: Dict[str, str] = _load_sector_map(source)

    def _check_vnstock(self):
        try:
            from vnstock import Vnstock  # noqa
            logger.info("✓ vnstock đã được cài đặt")
        except ImportError:
            raise ImportError("Chưa cài vnstock. Chạy: pip install vnstock")

    def _wait_for_rate_limit(self, wait_secs: float = RATE_LIMIT_WAIT):
        logger.warning(f"  ⚠ Rate limit! Chờ {wait_secs:.0f}s rồi tiếp tục...")
        for remaining in range(int(wait_secs), 0, -10):
            logger.info(f"    Còn {remaining}s...")
            time.sleep(min(10, remaining))
        logger.info("  ✓ Tiếp tục crawl...")

    def get_company_info(self, ticker: str) -> Dict:
        """
        Lấy thông tin tổng quan công ty.
        Sector lấy từ _sector_map (đã tải sẵn), không gọi overview() nữa
        vì overview() không có column sector.
        """
        from vnstock import Vnstock

        def _fetch():
            stock = Vnstock().stock(symbol=ticker, source=self.source)
            overview = stock.company.overview()
            return overview

        overview, err = _safe_call(_fetch)

        if err == 'rate_limit':
            self._wait_for_rate_limit()
            overview, err = _safe_call(_fetch)

        # Lấy exchange và company_name từ overview
        exchange     = ''
        company_name = ticker

        if overview is not None and err is None:
            try:
                if isinstance(overview, pd.DataFrame):
                    overview = _flatten_df(overview)
                    if not overview.empty:
                        row = overview.iloc[0]
                        # exchange
                        for key in ['exchange', 'Exchange']:
                            if key in row.index:
                                exchange = _safe_str(row[key])
                                if exchange:
                                    break
                        # company_name
                        for key in ['short_name', 'shortName', 'company_name',
                                    'companyName', 'organ_name', 'organName', 'symbol']:
                            if key in row.index:
                                v = _safe_str(row[key])
                                if v:
                                    company_name = v
                                    break
                elif isinstance(overview, dict):
                    exchange     = _safe_str(overview.get('exchange', overview.get('Exchange', '')))
                    company_name = _safe_str(
                        overview.get('short_name') or overview.get('shortName') or
                        overview.get('company_name') or overview.get('companyName') or ticker
                    )
            except Exception as e:
                logger.debug(f"[{ticker}] parse overview error: {e}")

        # Sector từ sector_map (không từ overview)
        sector_vn = self._sector_map.get(ticker.upper(), '')
        if not sector_vn:
            logger.debug(f"[{ticker}] sector không có trong sector_map → 'Other'")

        return {
            'exchange':     exchange,
            'sector_vn':    sector_vn,
            'company_name': company_name,
        }

    def _fetch_statement(self, ticker: str, stmt_type: str) -> Optional[pd.DataFrame]:
        from vnstock import Vnstock

        def _fetch():
            stock   = Vnstock().stock(symbol=ticker, source=self.source)
            finance = stock.finance
            # KBS: chỉ nhận period=, không nhận lang= hay dropna=
            if stmt_type == 'balance':
                return finance.balance_sheet(period='year')
            elif stmt_type == 'income':
                return finance.income_statement(period='year')
            elif stmt_type == 'cashflow':
                return finance.cash_flow(period='year')
            elif stmt_type == 'ratio':
                return finance.ratio(period='year')

        for attempt in range(MAX_RETRIES):
            result, err = _safe_call(_fetch)

            if err is None:
                if result is not None and not (hasattr(result, 'empty') and result.empty):
                    df = _normalize_kbs_df(result, ticker, stmt_type)
                    return df  # None nếu normalize thất bại
                return None

            elif err == 'rate_limit':
                wait = RATE_LIMIT_WAIT + (attempt * 30)
                self._wait_for_rate_limit(wait)
                continue

            else:
                logger.debug(f"[{ticker}] {stmt_type} error: {err}")
                return None

        return None

    def get_financial_statements(self, ticker: str) -> Optional[Dict[str, pd.DataFrame]]:
        results = {}

        bs = self._fetch_statement(ticker, 'balance')
        if bs is not None:
            results['balance'] = bs
        time.sleep(1.5)

        inc = self._fetch_statement(ticker, 'income')
        if inc is not None:
            results['income'] = inc
        time.sleep(1.5)

        if 'balance' not in results or 'income' not in results:
            logger.warning(f"[{ticker}] Thiếu balance/income sheet → skip")
            return None

        cf = self._fetch_statement(ticker, 'cashflow')
        if cf is not None:
            results['cashflow'] = cf
        time.sleep(1.5)

        ratio = self._fetch_statement(ticker, 'ratio')
        if ratio is not None:
            results['ratio'] = ratio
        time.sleep(1.5)

        return results

    def get_all_tickers(self, exchange: str = 'HOSE') -> List[str]:
        from vnstock import Vnstock

        def _fetch():
            stock = Vnstock().stock(symbol='VCB', source=self.source)
            return stock.listing.symbols_by_exchange()

        df, err = _safe_call(_fetch)
        if err or df is None:
            logger.error(f"Lỗi lấy danh sách ticker: {err}")
            return []

        df = _flatten_df(df)
        exc_col = next((c for c in df.columns if 'exchange' in c.lower()), None)
        sym_col = next((c for c in df.columns if 'symbol' in c.lower()), None)
        if exc_col and sym_col:
            tickers = df[df[exc_col].astype(str).str.upper() == exchange.upper()][sym_col].tolist()
        else:
            tickers = df.iloc[:, 0].astype(str).tolist()

        logger.info(f"Tìm thấy {len(tickers)} mã tại {exchange}")
        return tickers

    def crawl_tickers(
        self,
        tickers: List[str],
        rated_map: Optional[Dict[str, str]] = None,
    ) -> pd.DataFrame:
        from vn_data_adapter import VNDataAdapter
        adapter = VNDataAdapter()

        records = []
        failed  = []

        logger.info(f"\n{'='*60}")
        logger.info(f"Crawl {len(tickers)} tickers | Delay={REQUEST_DELAY}s | "
                    f"Batch={BATCH_SIZE} tickers → nghỉ {BATCH_PAUSE}s")
        logger.info(f"Tổng thời gian ước tính: "
                    f"~{len(tickers) * (REQUEST_DELAY + 6) / 60:.0f} phút")
        logger.info(f"{'='*60}")

        for i, ticker in enumerate(tickers, 1):
            logger.info(f"\n[{i}/{len(tickers)}] ── {ticker} ──")

            success = False
            for attempt in range(MAX_RETRIES):
                try:
                    company_info = self.get_company_info(ticker)
                    time.sleep(1.0)

                    statements = self.get_financial_statements(ticker)
                    if statements is None:
                        if attempt < MAX_RETRIES - 1:
                            logger.info(f"  Retry {attempt+1}/{MAX_RETRIES}...")
                            time.sleep(REQUEST_DELAY * 2)
                            continue
                        else:
                            failed.append(ticker)
                            break

                    features = adapter.transform(ticker, company_info, statements)
                    if features is None:
                        failed.append(ticker)
                        break

                    if rated_map and ticker in rated_map:
                        features['Credit_Rating'] = rated_map[ticker]
                        features['Has_Rating']    = True
                    else:
                        features['Has_Rating'] = False

                    records.append(features)
                    logger.info(
                        f"  ✓ Sector={features.get('Sector','?')} "
                        f"(raw='{company_info.get('sector_vn','')}') | "
                        f"ROA={features.get('ROA') or 'N/A'} | "
                        f"Rating={'✓' if features.get('Has_Rating') else '–'}"
                    )
                    success = True
                    break

                except SystemExit:
                    logger.warning(f"  SystemExit caught (rate limit) cho {ticker}")
                    self._wait_for_rate_limit()
                    continue

                except Exception as e:
                    if _is_rate_limit_error(e):
                        self._wait_for_rate_limit()
                        continue
                    else:
                        logger.error(f"  ✗ Lỗi: {e}")
                        if attempt < MAX_RETRIES - 1:
                            time.sleep(REQUEST_DELAY)
                            continue
                        failed.append(ticker)
                        break

            if not success and ticker not in failed:
                failed.append(ticker)

            if i % BATCH_SIZE == 0 and i < len(tickers):
                remaining = len(tickers) - i
                logger.info(
                    f"\n  ⏸ Batch {i//BATCH_SIZE} hoàn thành. "
                    f"Còn {remaining} tickers. "
                    f"Nghỉ {BATCH_PAUSE:.0f}s..."
                )
                time.sleep(BATCH_PAUSE)
            else:
                time.sleep(REQUEST_DELAY)

        logger.info(f"\n{'='*60}")
        logger.info(f"KẾT QUẢ: ✓ {len(records)} thành công | ✗ {len(failed)} thất bại")
        if failed:
            logger.info(f"Thất bại: {', '.join(failed)}")
        logger.info(f"{'='*60}")

        if not records:
            logger.error("Không crawl được dữ liệu nào!")
            return pd.DataFrame()

        return pd.DataFrame(records)


# ================================================================
# RATED DATA LOADER
# ================================================================

def load_rated_firms(filepath: str) -> Dict[str, str]:
    RATING_NORMALIZE = {
        'AAA': 'AAA',
        'AA+': 'AA', 'AA': 'AA', 'AA-': 'AA',
        'A+':  'A',  'A':  'A',  'A-':  'A',
        'BBB+': 'BBB', 'BBB': 'BBB', 'BBB-': 'BBB',
        'BB+': 'BB',  'BB':  'BB',  'BB-':  'BB',
        'B+':  'B',   'B':   'B',   'B-':   'B',
        'CCC+': 'CCC', 'CCC': 'CCC', 'CCC-': 'CCC',
        'CC': 'CC', 'C': 'C', 'D': 'D', 'SD': 'D',
    }
    try:
        df = pd.read_csv(filepath)
        ticker_col = next(
            (c for c in df.columns if c.lower() in ['ticker', 'symbol', 'ma', 'mã']),
            None
        )
        rating_col = next(
            (c for c in df.columns if c.lower() in ['credit_rating', 'rating', 'xephangtindung']),
            None
        )
        if not ticker_col or not rating_col:
            logger.error(f"File {filepath} cần cột ticker và credit_rating")
            return {}

        rated_map = {}
        for _, row in df.iterrows():
            ticker     = _safe_str(row[ticker_col]).upper()
            raw_rating = _safe_str(row[rating_col]).upper()
            normalized = RATING_NORMALIZE.get(raw_rating, raw_rating)
            if ticker:
                rated_map[ticker] = normalized

        logger.info(f"Loaded {len(rated_map)} rated firms từ {filepath}")
        logger.info(f"Ratings: {pd.Series(list(rated_map.values())).value_counts().to_dict()}")
        return rated_map

    except Exception as e:
        logger.error(f"Lỗi load rated firms: {e}")
        return {}


# ================================================================
# MAIN
# ================================================================

def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--tickers',    type=str, default=None)
    parser.add_argument('--exchange',   type=str, default=None,
                        choices=['HOSE', 'HNX', 'UPCOM'])
    parser.add_argument('--rated_file', type=str, default=None)
    parser.add_argument('--output',     type=str,
                        default='data/processed/vn_firms_crawled.csv')
    parser.add_argument('--source',     type=str, default=DEFAULT_SOURCE,
                        choices=['KBS'])  # VCI: server chết; TCBS: deprecated 401
    args = parser.parse_args()

    if not args.tickers and not args.exchange:
        args.tickers = 'VCB,BID,CTG,VNM,FPT,HPG,VIC,HDB,TCB,MWG,VHM,MSN,GAS,REE,VPB'

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    rated_map = {}
    if args.rated_file:
        rated_map = load_rated_firms(args.rated_file)

    crawler = VNStockCrawler(source=args.source)
    tickers = (
        crawler.get_all_tickers(args.exchange)
        if args.exchange
        else [t.strip().upper() for t in args.tickers.split(',') if t.strip()]
    )

    if not tickers:
        logger.error("Không có ticker nào!")
        return

    df = crawler.crawl_tickers(tickers, rated_map)
    if df.empty:
        logger.error("Không có dữ liệu")
        return

    df.to_csv(output_path, index=False, encoding='utf-8-sig')
    logger.info(f"✓ Saved {len(df)} records → {output_path}")


if __name__ == '__main__':
    main()
