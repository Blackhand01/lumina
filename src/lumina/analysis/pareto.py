"""Small Pareto-frontier helper."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from typing import TypeVar


T = TypeVar("T")


def pareto_frontier(items: Iterable[T], *, x: Callable[[T], float], y: Callable[[T], float]) -> tuple[T, ...]:
    """Return non-dominated items where higher x and higher y are better."""

    values = list(items)
    frontier: list[T] = []
    for item in values:
        item_x = x(item)
        item_y = y(item)
        dominated = any(x(other) >= item_x and y(other) >= item_y and (x(other), y(other)) != (item_x, item_y) for other in values)
        if not dominated:
            frontier.append(item)
    return tuple(frontier)
