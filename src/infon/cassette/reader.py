"""Range-get reader. Bytes-only interface so S3 swaps in unchanged.

Two things make this S3-friendly without actually hitting S3:
  1. LocalFetcher mimics HTTP Range semantics — open+seek+read.
  2. RangeFetcher logs (requests, bytes) so the MCTS lab can measure
     the same metrics you'd see against real S3.

For S3: replace LocalFetcher with an s3fs/boto3 backend that issues
`GET ... Range: bytes=offset-offset+length-1`.  The Hit→Infon path
above stays identical.
"""

from __future__ import annotations

import gzip
import io
import json
import struct
from dataclasses import dataclass, field
from typing import Protocol

from ..atom import Infon
from .format import MAGIC, Footer, _U32, _U64
from .index import Hit, Manifest


class RangeFetcher(Protocol):
    def fetch(self, path: str, offset: int, length: int) -> bytes: ...
    def size(self, path: str) -> int: ...


@dataclass
class LocalFetcher:
    """Local-FS implementation with request/byte counters for lab metrics."""
    requests: int = 0
    bytes_read: int = 0
    _footer_cache: dict = field(default_factory=dict)

    def fetch(self, path: str, offset: int, length: int) -> bytes:
        self.requests += 1
        with open(path, "rb") as f:
            f.seek(offset)
            buf = f.read(length)
        self.bytes_read += len(buf)
        return buf

    def size(self, path: str) -> int:
        import os
        return os.path.getsize(path)

    def reset_counters(self):
        self.requests = 0
        self.bytes_read = 0

    # ── bootstrap: load footer via two range gets ────────────────────────
    def load_footer(self, path: str) -> Footer:
        if path in self._footer_cache:
            return self._footer_cache[path]
        size = self.size(path)
        tail = self.fetch(path, size - 16, 16)
        if tail[8:] != MAGIC:
            raise ValueError(f"bad cassette trailer at {path}")
        foot_offset = _U64.unpack(tail[:8])[0]
        # footer len + footer body; we don't know footer len, so fetch
        # [foot_offset .. size-16)
        foot_buf = self.fetch(path, foot_offset, size - 16 - foot_offset)
        foot_len = _U32.unpack(foot_buf[:_U32.size])[0]
        foot = Footer.from_json(foot_buf[_U32.size:_U32.size + foot_len])
        self._footer_cache[path] = foot
        return foot


@dataclass
class FsspecFetcher:
    """fsspec-backed RangeFetcher for S3, GCS, local, anything fsspec speaks.

    Uses fsspec.open(path, "rb") which translates path schemes (s3://,
    gs://, local) to the right backend. Range reads are issued via
    seek()+read(); s3fs converts these to HTTP Range GETs automatically.

    Same counter semantics as LocalFetcher — a "request" is one fetch()
    call, which maps to one HTTP GET on S3.
    """
    requests: int = 0
    bytes_read: int = 0

    def fetch(self, path: str, offset: int, length: int) -> bytes:
        import fsspec
        self.requests += 1
        with fsspec.open(path, "rb") as f:
            f.seek(offset)
            buf = f.read(length)
        self.bytes_read += len(buf)
        return buf

    def size(self, path: str) -> int:
        import fsspec
        fs, rel = fsspec.core.url_to_fs(path)
        return fs.size(rel)

    def reset_counters(self):
        self.requests = 0
        self.bytes_read = 0


def hydrate_locs(fetcher: RangeFetcher, manifest: Manifest,
                 hits: list[Hit]) -> list[Infon]:
    """Fetch + decode each hit. Groups by cassette so a smart fetcher could
    coalesce adjacent ranges (left as a TODO; LocalFetcher issues one GET
    per hit, matching naive S3 Range behavior)."""
    by_cassette: dict[str, list[Hit]] = {}
    for h in hits:
        by_cassette.setdefault(h.cassette_id, []).append(h)

    infons: list[Infon] = []
    for cid, hs in by_cassette.items():
        path = manifest.cassette_path(cid)
        for h in hs:
            frame = fetcher.fetch(path, h.loc.offset, h.loc.length)
            # frame = [LEN u32][GZIP(json)]
            flen = _U32.unpack(frame[:_U32.size])[0]
            body = gzip.decompress(frame[_U32.size:_U32.size + flen])
            infons.append(Infon.from_dict(json.loads(body.decode())))
    return infons
