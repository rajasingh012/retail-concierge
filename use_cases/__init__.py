"""Application core: orchestration of MAF ChatAgents."""
from .discovery_agent import build_discovery_agent
from .synthesis_agent import build_synthesis_agent

__all__ = ["build_discovery_agent", "build_synthesis_agent"]