"""Extractor interface."""

from __future__ import annotations

from abc import ABC, abstractmethod

from graphrag_v2.document.models import TextUnit
from graphrag_v2.extraction.models import ExtractionResult


class BaseExtractor(ABC):
    """Base class for text-unit extractors."""

    @abstractmethod
    async def extract(self, text_unit: TextUnit) -> ExtractionResult:
        """Extract candidate knowledge from a text unit."""
