"""Generic plugin registry with entry-point discovery.

Every pluggable axis of the system — API providers, document sources, pipeline
stages, extraction templates — is a :class:`Registry`. Third-party packages
extend the system *without forking it* by publishing an entry point:

.. code-block:: toml

    # in a separate package, e.g. llm-extractor-patents
    [project.entry-points."llm_extractor.sources"]
    uspto = "llm_extractor_patents.uspto:USPTOSource"

The registry lazily loads those entry points the first time it is queried, so
importing the core package stays cheap.
"""
from __future__ import annotations

from dataclasses import dataclass, field


class RegistryError(KeyError):
    """Raised when a requested plugin name is not registered."""


@dataclass
class Registry:
    """A named collection of plugins backed by an entry-point group."""

    kind: str
    entry_point_group: str = ""
    _items: dict = field(default_factory=dict)
    _loaded: bool = False

    def register(self, name: str, obj=None):
        """Register ``obj`` under ``name``; usable as a decorator."""
        if obj is not None:
            self._items[name] = obj
            return obj

        def deco(target):
            self._items[name] = target
            return target

        return deco

    def _load_entry_points(self) -> None:
        if self._loaded or not self.entry_point_group:
            self._loaded = True
            return
        self._loaded = True
        try:
            from importlib.metadata import entry_points

            selected = entry_points(group=self.entry_point_group)
        except Exception:  # pragma: no cover - metadata unavailable
            return
        for ep in selected:
            if ep.name in self._items:
                continue
            try:
                self._items[ep.name] = ep.load()
            except Exception:  # pragma: no cover - a broken plugin must not break the run
                continue

    def get(self, name: str):
        self._load_entry_points()
        try:
            return self._items[name]
        except KeyError:
            raise RegistryError(
                f"unknown {self.kind} '{name}'; available: {self.names()}"
            ) from None

    def names(self) -> list:
        self._load_entry_points()
        return sorted(self._items)

    def items(self) -> dict:
        self._load_entry_points()
        return dict(self._items)

    def __contains__(self, name: str) -> bool:
        self._load_entry_points()
        return name in self._items
