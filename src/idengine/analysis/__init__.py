"""Metadata analysis: language, keywords, entities, dates."""

from idengine.analysis.entities import Entity, EntityExtractor
from idengine.analysis.keywords import extract_keywords, rake, tfidf_keywords
from idengine.analysis.language import LanguageGuess, detect_language
from idengine.analysis.metadata import DocumentMetadata, MetadataExtractor

__all__ = [
    "DocumentMetadata",
    "Entity",
    "EntityExtractor",
    "LanguageGuess",
    "MetadataExtractor",
    "detect_language",
    "extract_keywords",
    "rake",
    "tfidf_keywords",
]
