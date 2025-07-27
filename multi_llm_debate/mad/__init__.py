"""
MAD (Multi-Agent Debate) Framework

This module implements the MAD framework for multi-agent debates.
It provides a structured approach to conducting debates between multiple AI agents
with a moderator to evaluate and reach consensus.
"""

from .agent import Agent
from .debate import Debate, DebatePlayer

__all__ = ["Debate", "DebatePlayer", "Agent"]
