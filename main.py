"""
main.py
=======
Entry point duy nhất cho VN Corporate Credit Rating System.

Điều phối tất cả các bước:
  prep      → Map Kaggle CSV về schema chuẩn
  crawl     → Thu thập BCTC VN firms qua vnstock
  merge     → Gộp Kaggle + VN data thành training set
  train     → Stage 1: Train base model trên Kaggle (US) data
  finetune  → Stage 2: Fine-tune model trên VN rated firms (giải quyết domain gap)
  predict   → Predict credit rating cho VN firms mới
  all       → Chạy toàn bộ pipeline (bao gồm cả finetune nếu có VN rated data)

Quan hệ giữa các module:
  main.py
    ├── column_mapper.py    (map cột từ mọi nguồn → chuẩn)
    ├── script.py           (DataProcessor, FeatureEngineer, Model, Pipelines)
    │     ├── CreditRatingPipeline   (single-stage, Kaggle only)
    │     └── TwoStagePipeline       (two-stage: Kaggle → VN fine-tune)
    ├── vn_crawler.py       (thu thập BCTC qua vnstock)
    └── vn_data_adapter.py  (transform raw BCTC → features)

Cách dùng (Two-Stage — RECOMMENDED):
  # Step-by-step
  python main.py prep
  python main.py crawl --tickers VCB,VNM,FPT --rated data/raw/vn_rated.csv
  python main.py train                              # Stage 1: train trên Kaggle
  python main.py finetune --rated data/raw/vn_rated.csv  # Stage 2: fine-tune VN
  python main.py predict --tickers HPG,REE,PNJ

  # Full pipeline tự động
  python main.py all --tickers VCB,BID,VNM,FPT,HPG --rated data/raw/vn_rated.csv

  # Quick train Kaggle only (không fine-tune, không cần VN rated)
  python main.py prep
  python main.py train --train_data data/processed/kaggle_mapped.csv
  python main.py predict --predict_data data/processed/vn_unrated.csv
"""

import sys
import logging
import argparse
import pickle
import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Optional

# ================================================================
# SETUP — paths, logging, imports
# ================================================================

# Đảm bảo src/ trong path
ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'src'))

# Thư mục tạo bởi Paths.makedirs() — gọi sau khi load config

# ================================================================
# PATHS CONFIG
# ================================================================

# Paths đọc từ config.yaml qua config_loader
from config_loader import load_config, Paths as _PathsCls, setup_logging
_cfg   = load_config()
Paths  = _PathsCls(_cfg)
setup_logging(_cfg)
logger = logging.getLogger(__name__)  # instance thay vì class với attrs tĩnh


# ================================================================
# STEP 1 — PREP: Map Kaggle CSV
# ================================================================

def cmd_prep(args) -> bool:
    """Map Kaggle corporate_credit_rating.csv → schema chuẩn."""
    logger.info("\n" + "="*60)
    logger.info("STEP: PREP — Map Kaggle Data")
    logger.info("="*60)

    kaggle_path = Path(getattr(args, 'kaggle_data', None) or Paths.kaggle_raw)

    if not kaggle_path.exists():
        logger.error(f"Không tìm thấy Kaggle data tại: {kaggle_path}")
        logger.info("Download tại: https://www.kaggle.com/datasets/kirtandelwadia/corporate-credit-rating")
        logger.info(f"Đặt vào: {kaggle_path}")
        return False

    from column_mapper import ColumnMapper
    mapper = ColumnMapper()

    df = pd.read_csv(kaggle_path, low_memory=False)
    logger.info(f"Loaded Kaggle: {len(df)} rows, {df.shape[1]} cols")

    df_mapped = mapper.map_kaggle(df)
    report    = mapper.validate(df_mapped, require_rating=True)

    logger.info(f"\nMapping summary:\n{mapper.summary(df_mapped)}")

    if report['missing_feature_cols']:
        logger.warning(f"Missing features: {report['missing_feature_cols']}")

    Paths.kaggle_mapped.parent.mkdir(parents=True, exist_ok=True)
    df_mapped.to_csv(Paths.kaggle_mapped, index=False, encoding='utf-8-sig')
    logger.info(f"\n✓ Saved: {Paths.kaggle_mapped}  ({len(df_mapped)} records)")
    return True


# ================================================================
# STEP 2 — CRAWL: Thu thập BCTC VN firms
# ================================================================

def cmd_crawl(args) -> bool:
    """Crawl BCTC VN firms qua vnstock."""
    logger.info("\n" + "="*60)
    logger.info("STEP: CRAWL — Thu thập BCTC VN firms")
    logger.info("="*60)

    # Xác định tickers
    tickers = _parse_tickers(args)
    if not tickers:
        logger.warning("Không có tickers → dùng default 20 blue-chip VN firms")
        tickers = ('VCB,BID,CTG,VNM,FPT,HPG,VIC,HDB,TCB,MWG,'
                   'VHM,MSN,GAS,REE,VPB,ACB,MBB,SSI,PNJ,DGC').split(',')

    # Load rated map
    rated_map = _load_rated_map(args)

    # Crawl
    try:
        from vn_crawler import VNStockCrawler
    except ImportError:
        logger.error("Không import được vn_crawler. Đảm bảo file vn_crawler.py tồn tại.")
        return False

    crawler = VNStockCrawler(source=getattr(args, 'source', 'KBS'))
    df = crawler.crawl_tickers(tickers, rated_map)

    if df.empty:
        logger.error("Crawl thất bại — không có dữ liệu")
        return False

    Paths.vn_crawled.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(Paths.vn_crawled, index=False, encoding='utf-8-sig')
    logger.info(f"\n✓ Saved: {Paths.vn_crawled}  ({len(df)} firms)")
    logger.info(f"  Có rating: {df.get('Has_Rating', pd.Series(False)).sum()}")
    return True


# ================================================================
# STEP 3 — MERGE: Gộp tất cả datasets
# ================================================================

def cmd_merge(args) -> bool:
    """
    Gộp Kaggle + VN crawled → training set chuẩn hóa.
    Logic thực tế nằm trong src/data_merger.py::DataMerger.
    """
    logger.info("\n" + "="*60)
    logger.info("STEP: MERGE — Gộp datasets")
    logger.info("="*60)

    from data_merger import run_merge

    kaggle_path = str(Paths.kaggle_mapped) if Paths.kaggle_mapped.exists() else None
    vn_path     = str(Paths.vn_crawled)    if Paths.vn_crawled.exists()    else None

    if not kaggle_path:
        logger.warning(f"Không có Kaggle data ({Paths.kaggle_mapped}). Chạy 'prep' trước.")
    if not vn_path:
        logger.warning(f"Không có VN crawled data ({Paths.vn_crawled}). Chạy 'crawl' trước.")

    ok, n_train, n_predict = run_merge(
        kaggle_path    = kaggle_path,
        vn_path        = vn_path,
        train_output   = str(Paths.training_data),
        predict_output = str(Paths.vn_unrated),
        vn_weight      = 3.0,
    )

    if ok:
        logger.info(f"\n✓ Merge hoàn thành: {n_train} train / {n_predict} predict records")
    return ok


# ================================================================
# STEP 4 — TRAIN (Stage 1: Kaggle base model)
# ================================================================

def cmd_train(args) -> bool:
    """
    Stage 1: Train base model trên Kaggle (US) data.

    Dùng TwoStagePipeline thay vì CreditRatingPipeline để:
      - DomainAdaptationWeighter upweight US firms gần VN profile
      - Model sẵn sàng cho Stage 2 fine-tune (finetune command)

    Sau bước này nên chạy tiếp:
      python main.py finetune --rated data/raw/vn_rated.csv
    """
    logger.info("\n" + "="*60)
    logger.info("STEP: TRAIN — Stage 1 (Kaggle/US base model)")
    logger.info("="*60)

    train_path = Path(getattr(args, 'train_data', None) or Paths.kaggle_mapped)

    if not train_path.exists():
        # Fallback: thử training_data (merged) nếu kaggle_mapped không có
        fallback = Path(Paths.training_data)
        if fallback.exists():
            logger.info(f"Không tìm thấy {train_path}, dùng {fallback}")
            train_path = fallback
        else:
            logger.error(f"Không tìm thấy training data: {train_path}")
            logger.info("Chạy 'prep' trước để tạo kaggle_mapped.csv")
            return False

    from script import TwoStagePipeline
    from config_loader import make_model_config
    config   = make_model_config()
    pipeline = TwoStagePipeline(config)

    try:
        eval_results = pipeline.train(str(train_path))
        _print_eval_summary(eval_results, label="Stage 1 (Kaggle/US)")
        logger.info(f"\n✓ Stage 1 model saved → {config.MODEL_DIR}")
        logger.info("→ Tiếp theo: python main.py finetune --rated data/raw/vn_rated.csv")
        return True
    except Exception as e:
        logger.error(f"Training Stage 1 failed: {e}", exc_info=True)
        return False


# ================================================================
# STEP 4b — FINETUNE (Stage 2: VN domain adaptation)
# ================================================================

def cmd_finetune(args) -> bool:
    """
    Stage 2: Fine-tune model đã train (Stage 1) trên VN rated firms.

    Giải quyết domain gap giữa US Kaggle data và VN market:
      - RF:  warm_start thêm VN-specific trees
      - XGB: continue training thêm boosting rounds với VN data
      - GB:  warm_start thêm estimators, learning rate thấp hơn

    Yêu cầu:
      - Đã chạy 'train' (Stage 1) trước
      - Có file VN rated firms (--rated hoặc data/raw/vn_rated.csv)
        File này phải có cột: Ticker, Credit_Rating, và các BCTC features

    Output:
      - Model fine-tuned được lưu đè lên Stage 1 model
      - Domain Gap Report: so sánh acc_us vs acc_vn
    """
    logger.info("\n" + "="*60)
    logger.info("STEP: FINETUNE — Stage 2 (VN Domain Adaptation)")
    logger.info("="*60)

    from config_loader import make_model_config
    config = make_model_config()

    # Kiểm tra Stage 1 model tồn tại
    model_path = config.MODEL_DIR / 'best_model.pkl'
    if not model_path.exists():
        logger.error(f"Không tìm thấy Stage 1 model: {model_path}")
        logger.info("Chạy 'train' trước để tạo base model")
        return False

    # Xác định VN rated data path
    # Ưu tiên: --rated arg → vn_crawled (đã crawl và có rating) → vn_rated raw
    rated_arg  = getattr(args, 'rated', None)
    vn_sources = [
        rated_arg,
        str(Paths.vn_crawled),     # crawled data (có thể đã có Credit_Rating)
        str(Paths.vn_rated),       # raw rated file
    ]
    vn_path = None
    for src in vn_sources:
        if src and Path(src).exists():
            vn_path = src
            break

    if not vn_path:
        logger.error("Không tìm thấy VN rated data. Thử các nguồn:")
        for src in vn_sources:
            if src:
                logger.error(f"  - {src} → không tồn tại")
        logger.info("Chạy 'crawl --rated data/raw/vn_rated.csv' trước")
        return False

    logger.info(f"VN rated data: {vn_path}")

    from script import TwoStagePipeline
    try:
        pipeline = TwoStagePipeline.load_trained(config)
    except Exception as e:
        logger.error(f"Không load được Stage 1 model: {e}")
        return False

    try:
        eval_results = pipeline.fine_tune(vn_path)
        if eval_results:
            _print_eval_summary(eval_results, label="Stage 2 (VN Fine-tuned)")
        logger.info(f"\n✓ Fine-tuned model saved → {config.MODEL_DIR}")
        return True
    except Exception as e:
        logger.error(f"Fine-tuning Stage 2 failed: {e}", exc_info=True)
        return False



def cmd_predict(args) -> bool:
    """Predict credit rating cho VN firms."""
    logger.info("\n" + "="*60)
    logger.info("STEP: PREDICT")
    logger.info("="*60)

    from config_loader import make_model_config
    config = make_model_config()

    if not (config.MODEL_DIR / 'best_model.pkl').exists():
        logger.error(f"Không tìm thấy model tại {config.MODEL_DIR}. Chạy 'train' trước.")
        return False

    # Load pipeline — TwoStagePipeline (fine-tuned) ưu tiên hơn CreditRatingPipeline
    try:
        from script import TwoStagePipeline
        pipeline = TwoStagePipeline.load_trained(config)
        logger.info("Loaded TwoStagePipeline (two-stage fine-tuned model)")
    except Exception:
        from script import CreditRatingPipeline
        pipeline = CreditRatingPipeline.load_trained(config)
        logger.info("Loaded CreditRatingPipeline (single-stage base model)")

    # Xác định input data
    tickers = _parse_tickers(args)
    predict_input = Path(getattr(args, 'predict_data', None) or Paths.vn_unrated)

    if tickers:
        logger.info(f"Crawling {len(tickers)} tickers trước khi predict...")
        temp_path = Path('data/processed/_predict_temp.csv')
        ok = _crawl_to_file(tickers, temp_path)
        if not ok:
            return False
        predict_input = temp_path

    elif not predict_input.exists():
        logger.error(f"Không tìm thấy predict input: {predict_input}")
        logger.info("Chạy 'crawl' trước, hoặc chỉ định --tickers / --predict_data")
        return False

    output_path = Path(getattr(args, 'output', None) or Paths.predictions)
    try:
        df_result = pipeline.predict(str(predict_input), str(output_path))
        print("\n" + "="*70)
        print("VIETNAM CORPORATE CREDIT RATING PREDICTIONS")
        print("="*70)
        display_cols = ['Ticker', 'Sector', 'Predicted_Rating', 'Confidence']
        avail = [c for c in display_cols if c in df_result.columns]
        print(df_result[avail].to_string(index=False))
        print("="*70)
        print(f"\n✓ Full results saved → {output_path}")
        return True
    except Exception as e:
        logger.error(f"Prediction failed: {e}", exc_info=True)
        return False


# ================================================================
# MANUAL PREDICT — nhập tay ROA, ROE, ... không cần crawl
# ================================================================

# Danh sách features người dùng có thể nhập (canonical names)
MANUAL_FEATURES = [
    ("ROA",               "Return on Assets",                "VD: 0.05 = 5%"),
    ("ROE",               "Return on Equity",                "VD: 0.12 = 12%"),
    ("ROCE",              "Return on Capital Employed",       "VD: 0.08"),
    ("EBIT_Margin",       "EBIT / Revenue",                  "VD: 0.10 = 10%"),
    ("Current_Ratio",     "Current Assets / Current Liab",   "VD: 1.5"),
    ("Debt/Assets",       "Total Debt / Total Assets",       "VD: 0.45"),
    ("Debt/Equity",       "Total Debt / Equity",             "VD: 1.2"),
    ("Net_Debt/EBITDA",   "Net Debt / EBITDA",               "VD: 3.0 (âm = tiền nhiều hơn nợ)"),
    ("Asset_Turnover",    "Revenue / Total Assets",          "VD: 0.8"),
    ("WCTA",              "Working Capital / Total Assets",  "VD: 0.15"),
    ("RETA",              "Retained Earnings / Total Assets","VD: 0.20"),
    ("Market_to_Book",    "Market Cap / Book Value",         "VD: 2.0  (bỏ qua nếu không niêm yết)"),
    ("Revenue_Growth_YoY","Revenue growth year-over-year",   "VD: 0.08 = 8%  (bỏ qua nếu không có)"),
    ("Log_Revenue",       "ln(Revenue in Bn VND)",           "VD: ln(5000) ≈ 8.52  (bỏ qua nếu không có)"),
]

SECTOR_CHOICES = [
    "Banking", "FinancialServices", "RealEstate",
    "Manufacturing", "Energy", "Technology",
    "Healthcare", "Retail", "Transportation", "Other"
]


def _prompt_float(prompt: str, hint: str, required: bool = False) -> Optional[float]:
    """Nhập 1 số thực, trả None nếu bỏ qua."""
    while True:
        raw = input(f"  {prompt} [{hint}]: ").strip()
        if raw == "":
            if required:
                print("    ⚠ Trường này bắt buộc, vui lòng nhập.")
                continue
            return None
        try:
            return float(raw)
        except ValueError:
            print("    ⚠ Giá trị không hợp lệ. Nhập số thực (VD: 0.08) hoặc Enter để bỏ qua.")


def cmd_manual(args) -> bool:
    """
    Predict credit rating bằng cách nhập tay các chỉ số tài chính.
    Không cần ticker, không cần crawl.

    Hỗ trợ 2 chế độ:
      --interactive : hỏi từng chỉ số qua terminal (mặc định)
      --csv <file>  : đọc từ CSV đã điền sẵn (batch predict)
    """
    logger.info("\n" + "="*60)
    logger.info("STEP: MANUAL PREDICT — Nhập tay chỉ số tài chính")
    logger.info("="*60)

    from config_loader import make_model_config
    config = make_model_config()

    if not (config.MODEL_DIR / 'best_model.pkl').exists():
        logger.error(f"Chưa có model. Chạy 'train' (và 'finetune') trước.")
        return False

    try:
        from script import TwoStagePipeline
        pipeline = TwoStagePipeline.load_trained(config)
        logger.info("Loaded TwoStagePipeline")
    except Exception:
        try:
            from script import CreditRatingPipeline
            pipeline = CreditRatingPipeline.load_trained(config)
            logger.info("Loaded CreditRatingPipeline")
        except Exception as e:
            logger.error(f"Không load được model: {e}")
            return False

    # ── Chế độ CSV batch ──────────────────────────────────────────
    csv_path = getattr(args, 'csv', None)
    if csv_path:
        return _manual_from_csv(pipeline, csv_path, args, config)

    # ── Chế độ interactive ────────────────────────────────────────
    all_results = []
    session = 1

    print("\n" + "╔" + "═"*58 + "╗")
    print("║   MANUAL CREDIT RATING — Nhập chỉ số tài chính         ║")
    print("║   Nhấn Enter để bỏ qua chỉ số không có                 ║")
    print("║   Gõ 'done' sau khi nhập xong, 'quit' để thoát         ║")
    print("╚" + "═"*58 + "╝\n")

    while True:
        print(f"\n{'─'*60}")
        print(f"  CÔNG TY / FIRM #{session}")
        print(f"{'─'*60}")

        # Tên / mã công ty
        name = input("  Tên hoặc mã công ty (tùy chọn): ").strip() or f"Firm_{session}"

        # Sector
        print(f"\n  Sector — chọn số tương ứng:")
        for i, s in enumerate(SECTOR_CHOICES, 1):
            print(f"    {i:2d}. {s}")
        sec_raw = input("  Sector [Enter=Other]: ").strip()
        try:
            sector = SECTOR_CHOICES[int(sec_raw) - 1] if sec_raw else "Other"
        except (ValueError, IndexError):
            sector = "Other"

        # Nhập từng chỉ số tài chính
        print(f"\n  Nhập các chỉ số tài chính:")
        row: dict = {"Ticker": name, "Sector": sector}
        n_filled = 0

        for feat, label, hint in MANUAL_FEATURES:
            val = _prompt_float(f"{feat:<22} ({label})", hint)
            row[feat] = val
            if val is not None:
                n_filled += 1

        if n_filled < 3:
            print(f"\n  ⚠ Chỉ có {n_filled} chỉ số — cần ít nhất 3 để predict có nghĩa.")
            cont = input("  Tiếp tục dự đoán? [y/N]: ").strip().lower()
            if cont != 'y':
                print("  Bỏ qua firm này.")
                session += 1
                continue

        # ── Predict ──────────────────────────────────────────────
        result = _predict_single_row(pipeline, row, config)
        _print_single_result(result, n_filled)
        all_results.append(result)

        session += 1
        again = input("\n  Nhập thêm công ty khác? [Y/n]: ").strip().lower()
        if again == 'n':
            break

    # ── Lưu tất cả kết quả ───────────────────────────────────────
    if all_results:
        output_path = Path(getattr(args, 'output', None) or
                           'data/output/manual_predictions.csv')
        output_path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(all_results).to_csv(output_path, index=False, encoding='utf-8-sig')
        print(f"\n✓ Kết quả lưu tại: {output_path}")

    return True


def _manual_from_csv(pipeline, csv_path: str, args, config) -> bool:
    """
    Batch predict từ CSV nhập tay.
    CSV cần có cột: Ticker (hoặc Name), Sector, ROA, ROE, ...
    Các cột không có sẽ được điền NaN.

    Ví dụ CSV:
        Ticker,Sector,ROA,ROE,Debt/Assets,Current_Ratio
        CongTyA,Manufacturing,0.05,0.12,0.45,1.5
        CongTyB,Banking,0.008,0.14,0.82,
    """
    p = Path(csv_path)
    if not p.exists():
        # In ra template CSV nếu file không tồn tại
        logger.error(f"File không tồn tại: {p}")
        template_path = Path('data/raw/manual_input_template.csv')
        template_path.parent.mkdir(parents=True, exist_ok=True)
        feat_names = [f[0] for f in MANUAL_FEATURES]
        pd.DataFrame(columns=["Ticker", "Sector"] + feat_names).to_csv(
            template_path, index=False, encoding='utf-8-sig')
        logger.info(f"✓ Template CSV đã tạo tại: {template_path}")
        logger.info(f"  Điền dữ liệu vào template rồi chạy lại:")
        logger.info(f"  python main.py manual --csv {template_path}")
        return False

    df_input = pd.read_csv(p)
    logger.info(f"Loaded {len(df_input)} firms từ {p}")

    results = []
    for _, row in df_input.iterrows():
        row_dict = row.to_dict()
        if "Ticker" not in row_dict and "Name" in row_dict:
            row_dict["Ticker"] = row_dict["Name"]
        result = _predict_single_row(pipeline, row_dict, config)
        results.append(result)
        _print_single_result(result, n_filled=None)

    output_path = Path(getattr(args, 'output', None) or
                       'data/output/manual_predictions.csv')
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(results).to_csv(output_path, index=False, encoding='utf-8-sig')

    print(f"\n{'='*70}")
    print(f"BATCH PREDICT — {len(results)} firms")
    print(f"{'='*70}")
    df_out = pd.DataFrame(results)
    show_cols = [c for c in ['Ticker','Sector','Predicted_Rating','Confidence',
                              'Rating_Range','ROA','ROE','Debt/Assets'] if c in df_out.columns]
    print(df_out[show_cols].to_string(index=False))
    print(f"\n✓ Saved: {output_path}")
    return True


def _predict_single_row(pipeline, row: dict, config) -> dict:
    """Predict 1 row dict → trả về dict kết quả."""
    import numpy as np
    from pathlib import Path as _P

    feat_names = [f[0] for f in MANUAL_FEATURES]

    # Build single-row DataFrame với đầy đủ cột cần thiết
    df_row = pd.DataFrame([row])

    # Đảm bảo có Ticker và Sector
    if "Ticker" not in df_row.columns:
        df_row["Ticker"] = "Unknown"
    if "Sector" not in df_row.columns:
        df_row["Sector"] = "Other"

    # Thêm các cột còn thiếu với NaN
    for feat in feat_names:
        if feat not in df_row.columns:
            df_row[feat] = np.nan

    # Lưu temp file để dùng pipeline.predict (giữ nguyên flow xử lý)
    temp_path = _P('data/processed/_manual_temp.csv')
    temp_path.parent.mkdir(parents=True, exist_ok=True)
    df_row.to_csv(temp_path, index=False, encoding='utf-8-sig')

    out_path = _P('data/output/_manual_temp_out.csv')
    try:
        df_result = pipeline.predict(str(temp_path), str(out_path))
        if df_result is not None and not df_result.empty:
            res = df_result.iloc[0].to_dict()
        else:
            res = {"Ticker": row.get("Ticker", "?"), "Predicted_Rating": "N/A",
                   "Confidence": 0.0}
    except Exception as e:
        logger.error(f"Predict failed: {e}")
        res = {"Ticker": row.get("Ticker", "?"), "Predicted_Rating": "ERROR",
               "Confidence": 0.0, "Error": str(e)}

    # Thêm các input features vào kết quả để dễ xem lại
    for feat in ["ROA", "ROE", "Debt/Assets", "Current_Ratio", "EBIT_Margin"]:
        if feat in row and row[feat] is not None and not (
                isinstance(row[feat], float) and np.isnan(row[feat])):
            res[feat] = row[feat]

    return res


def _print_single_result(result: dict, n_filled: Optional[int]):
    """In kết quả predict 1 firm ra terminal."""
    rating = result.get("Predicted_Rating", "N/A")
    conf   = result.get("Confidence", 0.0)
    ticker = result.get("Ticker", "?")
    sector = result.get("Sector", "")

    # Màu rating (ASCII-safe fallback)
    rating_emoji = {
        "AAA":"🟢","AA":"🟢","A":"🟢",
        "BBB":"🟡","BB":"🟡",
        "B":"🟠","CCC":"🔴","CC":"🔴","C":"🔴","D":"🔴"
    }.get(rating, "⚪")

    print(f"\n  ┌{'─'*48}┐")
    print(f"  │  {ticker:<20}  {sector:<22}  │")
    print(f"  │  Xếp hạng dự đoán: {rating_emoji} {rating:<6}  (confidence: {conf:.1%})  │")
    if n_filled is not None:
        print(f"  │  Số chỉ số đã nhập: {n_filled}/14                         │")

    # Hiển thị key inputs để kiểm tra
    key_feats = [("ROA", "%.3f"), ("ROE", "%.3f"),
                 ("Debt/Assets", "%.3f"), ("Current_Ratio", "%.2f")]
    feat_strs = []
    for feat, fmt in key_feats:
        v = result.get(feat)
        if v is not None and not (isinstance(v, float) and np.isnan(v)):
            feat_strs.append(f"{feat}={fmt % v}")
    if feat_strs:
        line = "  " + " | ".join(feat_strs)
        print(f"  │  {line:<46}  │")
    print(f"  └{'─'*48}┘")


# ================================================================
# HELPERS
# ================================================================

def _parse_tickers(args) -> List[str]:
    """Parse --tickers argument → list of uppercase ticker strings."""
    raw = getattr(args, 'tickers', None)
    if not raw:
        return []
    return [t.strip().upper() for t in raw.split(',') if t.strip()]


def _load_rated_map(args) -> dict:
    """Load VN firms đã có credit rating từ file CSV."""
    rated_file = getattr(args, 'rated', None) or Paths.vn_rated
    rated_file = Path(rated_file)
    if not rated_file.exists():
        logger.info(f"Không có rated file ({rated_file}) → crawl không có ground truth VN")
        return {}

    try:
        from vn_crawler import load_rated_firms
        rated = load_rated_firms(str(rated_file))
        logger.info(f"Loaded {len(rated)} rated VN firms từ {rated_file}")
        return rated
    except Exception as e:
        logger.warning(f"Lỗi load rated file: {e}")
        return {}


def _crawl_to_file(tickers: List[str], output_path: Path) -> bool:
    """Crawl tickers và lưu vào file."""
    try:
        from vn_crawler import VNStockCrawler
        from data_merger import DataMerger

        crawler = VNStockCrawler(source='KBS')
        df = crawler.crawl_tickers(tickers)
        if df.empty:
            return False

        merger = DataMerger()
        df = merger.impute_missing(df)

        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_path, index=False)
        return True
    except Exception as e:
        logger.error(f"Crawl failed: {e}", exc_info=True)
        return False


def _print_eval_summary(eval_results: dict, label: str = ""):
    """In tóm tắt kết quả evaluation."""
    title = f"EVALUATION SUMMARY — {label}" if label else "EVALUATION SUMMARY"
    print("\n" + "="*55)
    print(title)
    print("="*55)
    print(f"  Accuracy:       {eval_results.get('accuracy', 0):.4f}")
    print(f"  Kappa (quad):   {eval_results.get('kappa', 0):.4f}")
    print(f"  F1 (macro):     {eval_results.get('f1_macro', 0):.4f}")
    print(f"  Within 1 notch: {eval_results.get('within_1_notch', 0):.4f}")
    print(f"  Within 2 notch: {eval_results.get('within_2_notch', 0):.4f}")
    print("="*55)


# ================================================================
# CLI
# ================================================================

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog='main.py',
        description='VN Corporate Credit Rating System',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
RECOMMENDED: Two-Stage Pipeline (giải quyết US→VN domain gap)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  Step 1 — Chuẩn bị Kaggle data:
    python main.py prep

  Step 2 — Thu thập BCTC VN firms (bao gồm firms đã có rating):
    python main.py crawl --tickers VCB,BID,VNM,FPT,HPG --rated data/raw/vn_rated.csv

  Step 3 — Stage 1: Train base model trên Kaggle (US) data:
    python main.py train

  Step 4 — Stage 2: Fine-tune trên VN rated firms:
    python main.py finetune --rated data/raw/vn_rated.csv

  Step 5 — Predict cho VN firms mới:
    python main.py predict --tickers REE,PNJ,DGC

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Full pipeline tự động (bao gồm cả finetune):
  python main.py all --tickers VCB,BID,VNM,FPT,HPG --rated data/raw/vn_rated.csv

Quick train Kaggle only (không fine-tune, không cần VN rated):
  python main.py prep
  python main.py train --train_data data/processed/kaggle_mapped.csv
  python main.py predict --predict_data data/processed/vn_unrated.csv
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        """,
    )

    subparsers = parser.add_subparsers(dest='command', metavar='COMMAND')
    subparsers.required = True

    # -- prep --
    p_prep = subparsers.add_parser('prep', help='Map Kaggle CSV → schema chuẩn')
    p_prep.add_argument('--kaggle_data', type=str,
                        help=f'Path to Kaggle CSV (default: {Paths.kaggle_raw})')

    # -- crawl --
    p_crawl = subparsers.add_parser('crawl', help='Thu thập BCTC VN firms qua vnstock')
    p_crawl.add_argument('--tickers', type=str,
                         help='Danh sách tickers, phân cách bằng dấu phẩy. VD: VCB,VNM,FPT')
    p_crawl.add_argument('--exchange', type=str, choices=['HOSE', 'HNX', 'UPCOM'],
                         help='Crawl toàn bộ sàn (thay cho --tickers)')
    p_crawl.add_argument('--rated', type=str,
                         help=f'File CSV VN firms đã có rating (default: {Paths.vn_rated})')
    p_crawl.add_argument('--source', type=str, default='KBS', choices=['KBS'],
                         help='Nguồn dữ liệu vnstock (default: KBS)')

    # -- merge --
    subparsers.add_parser('merge', help='Gộp Kaggle + VN data → training set')

    # -- train (Stage 1) --
    p_train = subparsers.add_parser(
        'train',
        help='Stage 1: Train base model trên Kaggle/US data',
    )
    p_train.add_argument(
        '--train_data', type=str,
        help=f'Path to training CSV (default: {Paths.kaggle_mapped})',
    )

    # -- finetune (Stage 2) --  [NEW]
    p_ft = subparsers.add_parser(
        'finetune',
        help='Stage 2: Fine-tune trên VN rated firms (chạy sau train)',
    )
    p_ft.add_argument(
        '--rated', type=str,
        help=(f'Path to VN rated firms CSV '
              f'(default: {Paths.vn_crawled} → {Paths.vn_rated})'),
    )

    # -- manual predict --
    p_manual = subparsers.add_parser(
        'manual',
        help='Predict bằng cách nhập tay ROA, ROE, ... (không cần ticker/crawl)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description="""
Predict credit rating bằng nhập tay các chỉ số tài chính.
Không cần mã chứng khoán, không cần kết nối vnstock.

Chế độ interactive (hỏi từng chỉ số):
  python main.py manual

Chế độ CSV batch (nhập nhiều công ty 1 lúc):
  python main.py manual --csv data/raw/my_firms.csv
  python main.py manual --template          ← tạo file template trống

Ví dụ CSV input:
  Ticker,Sector,ROA,ROE,Debt/Assets,Current_Ratio,EBIT_Margin
  CongTyA,Manufacturing,0.05,0.12,0.45,1.5,0.08
  CongTyB,Banking,0.008,0.14,0.82,,0.25
        """
    )
    p_manual.add_argument('--csv', type=str,
                          help='Path đến CSV nhập tay (batch predict)')
    p_manual.add_argument('--template', action='store_true',
                          help='Tạo file CSV template rồi thoát')
    p_manual.add_argument('--output', type=str,
                          help='Output path (default: data/output/manual_predictions.csv)')

    # -- predict --
    p_pred = subparsers.add_parser('predict', help='Predict credit rating')
    p_pred.add_argument('--tickers', type=str,
                        help='Tickers cần predict (sẽ crawl trước)')
    p_pred.add_argument('--predict_data', type=str,
                        help=f'Path to predict CSV (default: {Paths.vn_unrated})')
    p_pred.add_argument('--output', type=str,
                        help=f'Output path (default: {Paths.predictions})')

    # -- all --
    p_all = subparsers.add_parser(
        'all',
        help='Full pipeline: prep → crawl → train → finetune → predict',
    )
    p_all.add_argument('--tickers', type=str, help='VN firm tickers')
    p_all.add_argument('--rated', type=str,
                       help='VN rated firms file (dùng cho finetune)')
    p_all.add_argument('--kaggle_data', type=str)
    p_all.add_argument('--source', type=str, default='KBS')
    p_all.add_argument('--skip_crawl', action='store_true',
                       help='Skip crawl step (dùng data đã crawl trước)')
    p_all.add_argument('--skip_finetune', action='store_true',
                       help='Skip finetune step (chạy Stage 1 only)')
    p_all.add_argument('--train_data', type=str,
                       help='Override training data path cho Stage 1')

    return parser


# ================================================================
# MAIN
# ================================================================

def main():
    parser = build_parser()
    args = parser.parse_args()

    Paths.makedirs()
    logger.info(f"\n{'='*60}")
    logger.info(f"VN Corporate Credit Rating System")
    logger.info(f"Command: {args.command}")
    logger.info(f"{'='*60}")

    dispatch = {
        'prep':     cmd_prep,
        'crawl':    cmd_crawl,
        'merge':    cmd_merge,
        'train':    cmd_train,
        'finetune': cmd_finetune,   # [NEW v7]
        'predict':  cmd_predict,
        'manual':   cmd_manual,     # [NEW v8] nhập tay chỉ số tài chính
    }

    if args.command == 'manual' and getattr(args, 'template', False):
        # Tạo template CSV rồi thoát luôn
        template_path = Path('data/raw/manual_input_template.csv')
        template_path.parent.mkdir(parents=True, exist_ok=True)
        feat_names = [f[0] for f in MANUAL_FEATURES]
        pd.DataFrame(columns=["Ticker", "Sector"] + feat_names).to_csv(
            template_path, index=False, encoding='utf-8-sig')
        print(f"✓ Template CSV tạo tại: {template_path}")
        print(f"  Điền dữ liệu rồi chạy:")
        print(f"  python main.py manual --csv {template_path}")
        sys.exit(0)

    if args.command in dispatch:
        ok = dispatch[args.command](args)
        sys.exit(0 if ok else 1)

    elif args.command == 'all':
        # ── Full Two-Stage Pipeline ──────────────────────────────────────
        # prep → crawl → train (S1) → finetune (S2) → predict
        steps = [('prep', cmd_prep)]

        if not args.skip_crawl:
            steps.append(('crawl', cmd_crawl))

        steps.append(('train', cmd_train))  # Stage 1

        # Stage 2: chỉ chạy finetune nếu có rated data VÀ không skip
        has_rated = bool(
            getattr(args, 'rated', None)
            or Path(Paths.vn_crawled).exists()
            or Path(Paths.vn_rated).exists()
        )
        if not getattr(args, 'skip_finetune', False) and has_rated:
            steps.append(('finetune', cmd_finetune))
        elif getattr(args, 'skip_finetune', False):
            logger.info("Bỏ qua finetune (--skip_finetune)")
        else:
            logger.info(
                "Bỏ qua finetune (không có VN rated data). "
                "Sau khi có data, chạy: python main.py finetune --rated <file>"
            )

        steps.append(('predict', cmd_predict))

        for step_name, step_fn in steps:
            logger.info(f"\n{'#'*60}")
            logger.info(f"  Running: {step_name.upper()}")
            logger.info(f"{'#'*60}")
            ok = step_fn(args)
            if not ok:
                if step_name in ('prep', 'crawl', 'finetune'):
                    logger.warning(f"Step '{step_name}' có vấn đề, tiếp tục...")
                else:
                    logger.error(f"Step '{step_name}' thất bại. Dừng pipeline.")
                    sys.exit(1)

        logger.info("\n✅ Full two-stage pipeline hoàn thành!")


if __name__ == '__main__':
    main()
