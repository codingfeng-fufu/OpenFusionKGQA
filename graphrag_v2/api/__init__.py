"""Production runtime API for OpenFusionKGQA."""

from graphrag_v2.api.app import app, create_app
from graphrag_v2.api.settings import ApiRuntimeSettings

__all__ = ["ApiRuntimeSettings", "app", "create_app"]
