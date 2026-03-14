"""
src/config_loader.py
====================
Đọc config.yaml và trả về các object config cho từng module.

Mọi hardcode path/hyperparameter đều phải đọc từ đây,
không được hardcode trực tiếp trong main.py hay script.py.

Usage:
    from src.config_loader import load_config, make_model_config, Paths

    cfg   = load_config()                 # raw dict từ yaml
    paths = Paths(cfg)                    # typed path object
    model_cfg = make_model_config(cfg)    # Config dataclass cho script.py
"""

import yaml
import logging
from pathlib import Path
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

_CONFIG_FILE = Path(__file__).parent.parent / 'config.yaml'
_cached_cfg: Optional[Dict] = None


def load_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    """
    Đọc config.yaml và trả về dict.
    Cache lại sau lần đọc đầu tiên.

    Args:
        config_path: Override path mặc định (cho testing)
    """
    global _cached_cfg
    if _cached_cfg is not None and config_path is None:
        return _cached_cfg

    path = Path(config_path) if config_path else _CONFIG_FILE

    if not path.exists():
        logger.warning(f"config.yaml không tìm thấy tại {path}. Dùng default values.")
        return _default_config()

    try:
        with open(path, encoding='utf-8') as f:
            cfg = yaml.safe_load(f)
        logger.info(f"Config loaded từ: {path}")
        _cached_cfg = cfg
        return cfg
    except Exception as e:
        logger.error(f"Lỗi đọc config.yaml: {e}. Dùng defaults.")
        return _default_config()


class Paths:
    """
    Typed path object — thay thế class Paths hardcode trong main.py.

    Usage:
        cfg   = load_config()
        paths = Paths(cfg)
        df    = pd.read_csv(paths.kaggle_raw)
    """

    def __init__(self, cfg: Optional[Dict] = None):
        if cfg is None:
            cfg = load_config()
        p = cfg.get('paths', {})

        self.kaggle_raw     = Path(p.get('kaggle_raw',    'data/raw/corporate_credit_rating.csv'))
        self.vn_rated       = Path(p.get('vn_rated',      'data/raw/vn_rated.csv'))
        self.kaggle_mapped  = Path(p.get('kaggle_mapped', 'data/processed/kaggle_mapped.csv'))
        self.vn_crawled     = Path(p.get('vn_crawled',    'data/processed/vn_firms_crawled.csv'))
        self.training_data  = Path(p.get('training_data', 'data/processed/training_data.csv'))
        self.vn_unrated     = Path(p.get('vn_unrated',    'data/processed/vn_unrated.csv'))
        self.predictions    = Path(p.get('predictions',   'data/output/predictions.csv'))
        self.model          = Path(p.get('model',         'models/best_model.pkl'))
        self.processor      = Path(p.get('processor',     'models/processor.pkl'))
        self.plots_dir      = Path(p.get('plots_dir',     'plots'))
        self.logs_dir       = Path(p.get('logs_dir',      'logs'))

    def makedirs(self):
        """Tạo tất cả thư mục cần thiết."""
        for d in ['data/raw', 'data/processed', 'data/output',
                  'models', str(self.plots_dir), str(self.logs_dir)]:
            Path(d).mkdir(parents=True, exist_ok=True)


def make_model_config(cfg: Optional[Dict] = None):
    """
    Tạo Config dataclass cho src/script.py từ config.yaml.

    Returns:
        src.script.Config instance với values từ config.yaml
    """
    from src.script import Config

    if cfg is None:
        cfg = load_config()

    t = cfg.get('training', {})
    f = cfg.get('features', {})
    p = cfg.get('paths', {})

    rf  = t.get('random_forest', {})
    xgb = t.get('xgboost', {})
    gb  = t.get('gradient_boosting', {})

    return Config(
        RATING_SCALE          = t.get('rating_scale', None),
        UNIVERSAL_FEATURES    = f.get('universal', None),
        SECTOR_DEPENDENT      = f.get('sector_dependent', None),
        RANDOM_STATE          = t.get('random_state', 42),
        TEST_SIZE             = t.get('test_size', 0.20),
        CV_FOLDS              = t.get('cv_folds', 5),
        N_ESTIMATORS_RF       = rf.get('n_estimators', 300),
        MAX_DEPTH_RF          = rf.get('max_depth', 10),
        MIN_SAMPLES_SPLIT_RF  = rf.get('min_samples_split', 5),
        N_ESTIMATORS_XGB      = xgb.get('n_estimators', 300),
        MAX_DEPTH_XGB         = xgb.get('max_depth', 8),
        LEARNING_RATE_XGB     = xgb.get('learning_rate', 0.03),
        SUBSAMPLE_XGB         = xgb.get('subsample', 0.8),
        COLSAMPLE_XGB         = xgb.get('colsample_bytree', 0.8),
        N_ESTIMATORS_GB       = gb.get('n_estimators', 200),
        MAX_DEPTH_GB          = gb.get('max_depth', 6),
        LEARNING_RATE_GB      = gb.get('learning_rate', 0.05),
        OUTPUT_DIR            = Path(p.get('logs_dir', 'logs')),
        MODEL_DIR             = Path(p.get('model', 'models/best_model.pkl')).parent,
        PLOTS_DIR             = Path(p.get('plots_dir', 'plots')),
    )


def get_crawl_config(cfg: Optional[Dict] = None) -> Dict:
    """Trả về crawl config dict."""
    if cfg is None:
        cfg = load_config()
    return cfg.get('crawl', {
        'source': 'VCI',
        'request_delay': 1.5,
        'batch_pause': 3.0,
        'batch_size': 10,
        'max_missing_features': 5,
        'default_tickers': ['VCB', 'VNM', 'FPT', 'HPG', 'TCB'],
    })


def setup_logging(cfg: Optional[Dict] = None):
    """Cấu hình logging từ config.yaml."""
    if cfg is None:
        cfg = load_config()

    log_cfg = cfg.get('logging', {})
    level   = getattr(logging, log_cfg.get('level', 'INFO').upper(), logging.INFO)
    fmt     = log_cfg.get('format', '%(asctime)s | %(levelname)-8s | %(message)s')
    logfile = log_cfg.get('file', 'logs/main.log')

    Path(logfile).parent.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(
        level=level,
        format=fmt,
        handlers=[
            logging.FileHandler(logfile, encoding='utf-8'),
            logging.StreamHandler(),
        ],
        force=True,
    )


# ================================================================
# DEFAULT CONFIG (fallback nếu không có config.yaml)
# ================================================================

def _default_config() -> Dict:
    return {
        'paths': {
            'kaggle_raw':    'data/raw/corporate_credit_rating.csv',
            'vn_rated':      'data/raw/vn_rated.csv',
            'kaggle_mapped': 'data/processed/kaggle_mapped.csv',
            'vn_crawled':    'data/processed/vn_firms_crawled.csv',
            'training_data': 'data/processed/training_data.csv',
            'vn_unrated':    'data/processed/vn_unrated.csv',
            'predictions':   'data/output/predictions.csv',
            'model':         'models/best_model.pkl',
            'processor':     'models/processor.pkl',
            'plots_dir':     'plots',
            'logs_dir':      'logs',
        },
        'crawl': {
            'source': 'VCI',
            'request_delay': 1.5,
            'batch_pause': 3.0,
            'batch_size': 10,
            'max_missing_features': 5,
            'default_tickers': ['VCB', 'BID', 'VNM', 'FPT', 'HPG',
                                 'VIC', 'TCB', 'MWG', 'MSN', 'GAS'],
        },
        'merge': {'vn_weight': 3.0},
        'training': {
            'random_state': 42, 'test_size': 0.20, 'cv_folds': 5,
            'rating_scale': ['AAA', 'AA', 'A', 'BBB', 'BB', 'B', 'CCC', 'CC', 'C', 'D'],
            'random_forest': {'n_estimators': 300, 'max_depth': 10, 'min_samples_split': 5},
            'xgboost': {'n_estimators': 300, 'max_depth': 8, 'learning_rate': 0.03,
                        'subsample': 0.8, 'colsample_bytree': 0.8},
            'gradient_boosting': {'n_estimators': 200, 'max_depth': 6, 'learning_rate': 0.05},
        },
        'features': {
            'universal': ['ROA', 'ROE', 'ROCE', 'EBIT_Margin', 'Current_Ratio',
                          'Revenue_Growth_YoY', 'Log_Revenue', 'Market_Cap'],
            'sector_dependent': ['Debt/Assets', 'Debt/Equity', 'Net_Debt/EBITDA',
                                  'Asset_Turnover', 'WCTA', 'RETA', 'Market_to_Book'],
        },
        'logging': {'level': 'INFO', 'file': 'logs/main.log'},
    }
