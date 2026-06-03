"""Flask route blueprints."""

from itertools import chain


def prime_stream(gen):
    iterator = iter(gen)
    try:
        first = next(iterator)
    except StopIteration:
        return iter(())
    return chain([first], iterator)
