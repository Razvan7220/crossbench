"""Vendored / ported execution scaffolds.

Each scaffold pins an upstream commit SHA in its module docstring and locks
its load-bearing string constants with frozen-hash tests so an upstream
prompt change can't silently drift our benchmark numbers.
"""
