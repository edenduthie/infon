"""Cassette binary format — writer and reader.

A cassette is an immutable append-closed file of infons. Each record is a
gzip-compressed JSON blob, individually decompressable via byte-range read.
The footer carries per-record locators and aggregate stats; the trailer
lets a reader bootstrap from a 16-byte tail fetch.
"""

from __future__ import annotations

import gzip
import hashlib
import io
import json
import struct
import time
from dataclasses import dataclass, asdict, field
from typing import BinaryIO, Iterable, Iterator

from ..atom import Infon


MAGIC = b"INFC0001"
CASSETTE_VERSION = 1

_U32 = struct.Struct("<I")
_U64 = struct.Struct("<Q")


@dataclass
class RecordLoc:
    """Where one infon record lives inside a cassette + minimal pushdown fields."""
    infon_id: str
    offset: int                 # absolute byte offset in cassette
    length: int                 # compressed frame length (excl. LEN prefix)
    subject: str = ""
    predicate: str = ""
    object: str = ""
    polarity: int = 1
    confidence: float = 0.0
    timestamp: str | None = None


@dataclass
class Footer:
    """Cassette footer: locators + aggregate stats for pruning."""
    version: int = CASSETTE_VERSION
    cassette_id: str = ""
    created_at: str = ""
    n_records: int = 0
    records: list[RecordLoc] = field(default_factory=list)
    subjects: list[str] = field(default_factory=list)      # deduped
    predicates: list[str] = field(default_factory=list)
    objects: list[str] = field(default_factory=list)
    t_min: str | None = None
    t_max: str | None = None
    sha256_body: str = ""
    schema_ref: str = ""                                   # anchor schema hash/URI

    def to_json(self) -> bytes:
        return json.dumps({
            **asdict(self),
            "records": [asdict(r) for r in self.records],
        }, separators=(",", ":")).encode()

    @classmethod
    def from_json(cls, buf: bytes) -> "Footer":
        d = json.loads(buf.decode())
        d["records"] = [RecordLoc(**r) for r in d.get("records", [])]
        return cls(**d)


class CassetteWriter:
    """Streaming writer: add infons, then close() to seal footer + trailer."""

    def __init__(self, fp: BinaryIO, cassette_id: str, schema_ref: str = ""):
        self.fp = fp
        self.footer = Footer(
            cassette_id=cassette_id,
            created_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            schema_ref=schema_ref,
        )
        self._body_hash = hashlib.sha256()
        self._body_start: int = 0
        self._subjects: set[str] = set()
        self._predicates: set[str] = set()
        self._objects: set[str] = set()
        self._write_header()

    def _write_header(self):
        self.fp.write(MAGIC)
        header = {
            "version": CASSETTE_VERSION,
            "cassette_id": self.footer.cassette_id,
            "schema_ref": self.footer.schema_ref,
            "created_at": self.footer.created_at,
        }
        hjson = json.dumps(header, separators=(",", ":")).encode()
        self.fp.write(_U32.pack(len(hjson)))
        self.fp.write(hjson)
        self._body_start = self.fp.tell()

    def add(self, infon: Infon) -> RecordLoc:
        payload = json.dumps(infon.to_dict(), separators=(",", ":")).encode()
        frame = gzip.compress(payload, compresslevel=6, mtime=0)

        offset = self.fp.tell()
        self.fp.write(_U32.pack(len(frame)))
        self.fp.write(frame)
        self._body_hash.update(frame)

        loc = RecordLoc(
            infon_id=infon.infon_id,
            offset=offset,
            length=len(frame) + _U32.size,  # full frame incl. LEN prefix
            subject=infon.subject,
            predicate=infon.predicate,
            object=infon.object,
            polarity=infon.polarity,
            confidence=infon.confidence,
            timestamp=infon.timestamp,
        )
        self.footer.records.append(loc)
        self._subjects.add(infon.subject)
        self._predicates.add(infon.predicate)
        self._objects.add(infon.object)
        if infon.timestamp:
            if self.footer.t_min is None or infon.timestamp < self.footer.t_min:
                self.footer.t_min = infon.timestamp
            if self.footer.t_max is None or infon.timestamp > self.footer.t_max:
                self.footer.t_max = infon.timestamp
        return loc

    def close(self) -> Footer:
        self.footer.n_records = len(self.footer.records)
        self.footer.subjects = sorted(self._subjects)
        self.footer.predicates = sorted(self._predicates)
        self.footer.objects = sorted(self._objects)
        self.footer.sha256_body = self._body_hash.hexdigest()

        foot = self.footer.to_json()
        foot_offset = self.fp.tell()
        self.fp.write(_U32.pack(len(foot)))
        self.fp.write(foot)
        self.fp.write(_U64.pack(foot_offset))
        self.fp.write(MAGIC)
        self.fp.flush()
        return self.footer


class CassetteReader:
    """Random-access reader. Works over any file-like that supports seek/read.

    For S3, wrap an fsspec file or use RangeFetcher (see reader.py).
    """

    def __init__(self, fp: BinaryIO):
        self.fp = fp
        self._footer: Footer | None = None

    @property
    def footer(self) -> Footer:
        if self._footer is None:
            self._footer = self._load_footer()
        return self._footer

    def _load_footer(self) -> Footer:
        self.fp.seek(-16, io.SEEK_END)
        tail = self.fp.read(16)
        if tail[8:] != MAGIC:
            raise ValueError("bad cassette trailer")
        foot_offset = _U64.unpack(tail[:8])[0]
        self.fp.seek(foot_offset)
        foot_len = _U32.unpack(self.fp.read(_U32.size))[0]
        return Footer.from_json(self.fp.read(foot_len))

    def read_at(self, offset: int, length: int) -> Infon:
        """Read a single record given its (offset, length) from the footer."""
        self.fp.seek(offset)
        frame_len = _U32.unpack(self.fp.read(_U32.size))[0]
        frame = self.fp.read(frame_len)
        return Infon.from_dict(json.loads(gzip.decompress(frame).decode()))

    def iter_records(self) -> Iterator[Infon]:
        for loc in self.footer.records:
            yield self.read_at(loc.offset, loc.length)


def write_cassette(path: str, infons: Iterable[Infon], cassette_id: str,
                   schema_ref: str = "") -> Footer:
    """Convenience: stream infons into a new cassette file.

    Accepts local paths or fsspec URIs (s3://, gs://, ...)."""
    if "://" in path:
        import fsspec
        with fsspec.open(path, "wb") as f:
            w = CassetteWriter(f, cassette_id=cassette_id, schema_ref=schema_ref)
            for inf in infons:
                w.add(inf)
            return w.close()
    with open(path, "wb") as f:
        w = CassetteWriter(f, cassette_id=cassette_id, schema_ref=schema_ref)
        for inf in infons:
            w.add(inf)
        return w.close()


def open_cassette(path: str) -> CassetteReader:
    """Random-access read. Accepts local paths or fsspec URIs."""
    if "://" in path:
        import fsspec
        return CassetteReader(fsspec.open(path, "rb").open())
    return CassetteReader(open(path, "rb"))
