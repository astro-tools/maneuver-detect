"""The data layer — public-catalog fetch, clean, and per-object mean-element assembly.

Catalogue fetchers with on-disk caching and rate-limit discipline, elset cleaning (epoch
dedup, bad-elset rejection, gap handling), and per-NORAD mean-element time-series assembly.
Built behind a stable internal interface so the historical-TLE pipeline can later be extracted.
"""

from __future__ import annotations
