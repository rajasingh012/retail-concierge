"""Application core: builders for the three collaborating MAF agents."""
from .critic_agent import build_critic_agent
from .discovery_agent import build_discovery_agent
from .research_agent import build_research_agent

__all__ = [
    "build_discovery_agent",
    "build_research_agent",
    "build_critic_agent",
]
