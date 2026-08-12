"""Offline test suite for llm-extractor.

Every test runs without network access: model calls go through
``tests._fakes.FakeProvider`` and HTTP sources through injected fetchers.
"""
