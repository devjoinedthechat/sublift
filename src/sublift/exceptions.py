"""Error types.

All of them subclass both :class:`SubliftError` and a builtin, so existing
``except ValueError`` handlers keep working while ``except SubliftError`` lets
you separate "sublift refused" from "numpy blew up".

The distinction worth making is between data that is malformed and an analysis
that is not identified. A :class:`PanelError` means the input does not describe a
well-formed experiment and something upstream needs fixing. A
:class:`NotIdentifiedError` means the data is fine but the question cannot be
answered from it -- a horizon past the follow-up, a stratification with empty
cells. The second is not a bug report; it is the library declining to make
something up.
"""

from __future__ import annotations

__all__ = ["SubliftError", "PanelError", "NotIdentifiedError", "EstimationError"]


class SubliftError(Exception):
    """Base class for every error sublift raises deliberately."""


class PanelError(SubliftError, ValueError):
    """The input data does not describe a well-formed retention experiment."""


class NotIdentifiedError(SubliftError, ValueError):
    """The data is well-formed, but it cannot answer the question as asked."""


class EstimationError(SubliftError, RuntimeError):
    """An estimator could not be computed -- empty strata, a model that would not fit."""
