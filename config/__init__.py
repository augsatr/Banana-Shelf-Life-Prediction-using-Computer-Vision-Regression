from pathlib import Path
from typing import Dict, Any
import yaml
import os


class Config:
    _instance = None

    def __new__(cls, path: str = None):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._loaded = False
        return cls._instance

    def __init__(self, path: str = None):
        if self._loaded:
            return
        base = Path(__file__).resolve().parent / "config.yaml"
        with open(base) as f:
            self.cfg: Dict[str, Any] = yaml.safe_load(f)
        if path:
            with open(path) as f:
                override = yaml.safe_load(f)
                self._deep_merge(self.cfg, override)
        self._resolve_paths()
        self._loaded = True

    def _resolve_paths(self):
        base = Path(__file__).resolve().parent.parent
        for key in self.cfg.get("paths", {}):
            rel = self.cfg["paths"][key]
            self.cfg["paths"][key] = base / rel
            os.makedirs(self.cfg["paths"][key], exist_ok=True)

    def _deep_merge(self, base: dict, override: dict):
        for k, v in override.items():
            if k in base and isinstance(base[k], dict) and isinstance(v, dict):
                self._deep_merge(base[k], v)
            else:
                base[k] = v

    def __getitem__(self, key):
        return self.cfg[key]

    def __getattr__(self, name):
        return self.cfg.get(name, {})

    def get(self, *keys, default=None):
        val = self.cfg
        for k in keys:
            if isinstance(val, dict):
                val = val.get(k)
                if val is None:
                    return default
            else:
                return default
        return val if val is not None else default


cfg = Config()
