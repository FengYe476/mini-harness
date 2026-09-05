import os
import fnmatch

from pathlib import Path

from mini_harness.config import CONFIG

def _resolve_file(file_path: Path, cfg = CONFIG) -> Path:
    path = Path(file_path)
    if not path.is_absolute():
        path = cfg.work_space/path
    return path.resolve()

def is_denied(real_path: Path, cfg = CONFIG) -> bool:
    if any(part for part in cfg.deny_dir if part in real_path.parts[:-1]):
        return True
    return any(fnmatch.fnmatch(real_path.name, part) for part in cfg.deny_name)

def validate_read(file_path: Path, cfg = CONFIG) -> Path:
    real_path = _resolve_file(file_path, cfg = cfg)
    if not cfg.guard_read:
        return real_path
    if not real_path.is_relative_to(cfg.work_space):
        raise PermissionError(f'[Access denied]: the reading file is restricted in {cfg.work_space}, got {real_path}')
    if is_denied(real_path, cfg = cfg):
        raise PermissionError(f'[Access denied]: the reading file is denied, got {real_path}')

    return real_path

def validate_write(file_path: Path, cfg = CONFIG) -> Path:
    real_path = _resolve_file(file_path, cfg = cfg)
    if not cfg.guard_write:
        return real_path

    if not real_path.is_relative_to(cfg.sandbox_dir):
        raise PermissionError(f'[Access denied]: the writing file is restricted in {cfg.sandbox_dir}, got {real_path}')

    return real_path