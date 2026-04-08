# src/utils/io.py
"""파일 읽기/쓰기 유틸리티"""

import json
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd
import yaml


def load_config(station_id: str, config_root: str = "config") -> Dict:
    """관측소 설정 + 처리 옵션을 합쳐서 반환"""
    root = Path(config_root)

    station_file = root / "stations" / f"{station_id}.yaml"
    if not station_file.exists():
        raise FileNotFoundError(f"Station config not found: {station_file}")

    options_file = root / "processing_options.yaml"
    if not options_file.exists():
        raise FileNotFoundError(f"Processing options not found: {options_file}")

    with open(station_file, encoding="utf-8") as f:
        station_cfg = yaml.safe_load(f)

    with open(options_file, encoding="utf-8") as f:
        options_cfg = yaml.safe_load(f)

    return {"station": station_cfg, "options": options_cfg}


def save_json(data: Dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    def _convert(obj: Any) -> Any:
        import numpy as np
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        raise TypeError(f"Not serializable: {type(obj)}")

    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False, default=_convert)


def load_json(path: Path) -> Optional[Dict]:
    if not path.exists():
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def save_excel(df: pd.DataFrame, path: Path, index: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_excel(path, index=index, engine="openpyxl")


def load_excel(path: Path, **kwargs) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Excel file not found: {path}")
    return pd.read_excel(path, **kwargs)