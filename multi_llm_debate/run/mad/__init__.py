"""
MAD (Multi-Agent Debate) Benchmark Module

This module provides benchmark integration for the MAD framework.
"""

from .main import main
from .evaluate import evaluate_mad_results
from .run_debate import process_mad_dataset
from .utils import load_mad_dataset

__all__ = ["main", "evaluate_mad_results", "process_mad_dataset", "load_mad_dataset"] 