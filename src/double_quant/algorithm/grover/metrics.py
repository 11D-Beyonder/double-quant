from __future__ import annotations

import math


def grover_success_probability(
    *,
    num_items: int,
    iterations: int,
    marked_items: int = 1,
) -> float:
    """Return the ideal success probability after ``iterations`` Grover steps."""
    if num_items <= 0:
        raise ValueError("num_items must be positive")
    if marked_items <= 0 or marked_items > num_items:
        raise ValueError("marked_items must be in [1, num_items]")
    if iterations < 0:
        raise ValueError("iterations must be non-negative")
    theta = math.asin(math.sqrt(marked_items / num_items))
    return math.sin((2 * iterations + 1) * theta) ** 2
