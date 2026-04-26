"""
agent_logger.py
---------------
Centralized logging for the Healthcare Multi-Agent system.

All agents and the orchestrator write structured, human-readable entries to:
    <project_root>/logs/healthcare_agents.log

Log format (each line):
    [YYYY-MM-DD HH:MM:SS] [LEVEL   ] [COMPONENT          ] MESSAGE

Usage:
    from .agent_logger import get_logger
    log = get_logger(__name__)
    log.info("Something happened")
"""

from __future__ import annotations

import logging
import os
from logging.handlers import RotatingFileHandler
from pathlib import Path

# ---------------------------------------------------------------------------
# Resolve the project root regardless of where Python is run
# ---------------------------------------------------------------------------
# This file lives at:  src/healthcare_support_agents/agent_logger.py
# Project root is:     ../../  (two levels up)
_THIS_FILE = Path(__file__).resolve()
_PROJECT_ROOT = _THIS_FILE.parent.parent.parent          # healthcare_multi_agent_tutorial/
_LOG_FILE = _PROJECT_ROOT / "agent_operations.log"       # root-level log file


def _setup_root_logger() -> logging.Logger:
    """Create (once) the shared file+console logger for all agent modules."""
    logger = logging.getLogger("healthcare_agents")
    if logger.handlers:
        # Already configured – return as-is (handles reimports)
        return logger

    logger.setLevel(logging.DEBUG)

    # --- File handler (rotating, max 5 MB × 3 backups) ---
    file_fmt = logging.Formatter(
        fmt="[%(asctime)s] [%(levelname)-8s] [%(name)-32s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    file_handler = RotatingFileHandler(
        _LOG_FILE,
        maxBytes=5 * 1024 * 1024,   # 5 MB
        backupCount=3,
        encoding="utf-8",
    )
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(file_fmt)

    # --- Console handler (INFO and above only) ---
    console_fmt = logging.Formatter(
        fmt="%(asctime)s [%(levelname)-8s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(console_fmt)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    logger.propagate = False
    return logger


# Initialise once at import time
_root_logger = _setup_root_logger()


def get_logger(name: str) -> logging.Logger:
    """
    Return a child logger under the 'healthcare_agents' hierarchy.

    Parameters
    ----------
    name : str
        Typically ``__name__`` from the calling module.
        Example: 'healthcare_support_agents.llm_agents'  →  logged as
                 'healthcare_agents.llm_agents'
    """
    # Strip the package prefix so child names stay short in the log
    short_name = name.split(".")[-1] if "." in name else name
    return _root_logger.getChild(short_name)


def log_separator(label: str = "") -> None:
    """Write a visual separator line to the log file (helps distinguish sessions)."""
    border = "=" * 72
    _root_logger.info(border)
    if label:
        _root_logger.info("  %s", label)
        _root_logger.info(border)


def get_log_file_path() -> Path:
    """Return the absolute path to the active log file (useful for UI display)."""
    return _LOG_FILE
