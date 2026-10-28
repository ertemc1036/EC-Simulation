"""NanoCrab Lab research workflow (offline fixtures).

Does not call a model provider and does not modify the simulation engine.
MiroFish source is not included. See docs/nanocrab-research-api-v1.md.
"""

__all__ = ["ResearchService"]


def __getattr__(name: str):
    if name == "ResearchService":
        from backend.research.service import ResearchService

        return ResearchService
    raise AttributeError(name)
