"""Claim-based coordination for extracting a corpus that is still downloading.

Two things make a large harvest awkward. The first is that waiting for every
PDF before extracting anything wastes hours: papers arrive over a long tail, and
the workers could have been busy. The second follows from fixing the first — as
soon as several workers share a growing folder, two of them will reach for the
same new file.

The fix is a claim taken before any work starts. A claim is a file created with
``O_CREAT|O_EXCL``, which the kernel guarantees will succeed for exactly one
caller even across nodes on a shared filesystem. That primitive is used rather
than a database because SQLite's locking is not dependable over NFS/Lustre,
which is precisely where this corpus lives.

Claims are keyed by *content hash*, not filename, so the same paper downloaded
twice under different names is still extracted once.

A worker that dies leaves its claim behind. Rather than blocking that paper
forever, a claim older than :data:`STALE_AFTER_S` with no completion marker is
treated as abandoned and may be taken again — at worst the paper is extracted
twice, which the response cache makes nearly free, and which is far better than
losing it.
"""
from __future__ import annotations

import hashlib
import os
import time
from dataclasses import dataclass
from pathlib import Path

#: A claim with no result after this long is assumed to belong to a dead worker.
STALE_AFTER_S = 1800.0

CLAIM_SUFFIX = ".claim"
DONE_SUFFIX = ".done"


def content_key(path) -> str:
    """Stable identity for a file: hash of its bytes, not its name.

    Read in chunks so a large PDF does not have to be held in memory, and so
    that hashing a directory of thousands stays predictable.
    """
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()[:32]


@dataclass
class ClaimStore:
    """Coordinates which worker extracts which document."""

    root: Path

    def __post_init__(self) -> None:
        self.root = Path(self.root)
        self.root.mkdir(parents=True, exist_ok=True)

    def _claim_path(self, key: str) -> Path:
        return self.root / f"{key}{CLAIM_SUFFIX}"

    def _done_path(self, key: str) -> Path:
        return self.root / f"{key}{DONE_SUFFIX}"

    def is_done(self, key: str) -> bool:
        return self._done_path(key).exists()

    def claim(self, key: str, owner: str = "") -> bool:
        """Try to take ownership of one document. True only for the winner."""
        if self.is_done(key):
            return False

        path = self._claim_path(key)
        try:
            handle = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
        except FileExistsError:
            return self._steal_if_stale(path, key, owner)
        except OSError:
            return False

        with os.fdopen(handle, "w", encoding="utf-8") as stream:
            stream.write(f"{owner or os.getpid()}\n{time.time()}\n")
        return True

    def _steal_if_stale(self, path: Path, key: str, owner: str) -> bool:
        """Reclaim a claim whose owner appears to have died."""
        try:
            age = time.time() - path.stat().st_mtime
        except OSError:
            return False
        if age < STALE_AFTER_S or self.is_done(key):
            return False
        try:
            # Re-stat after unlink to make sure we are not racing another
            # worker that is also trying to steal the same stale claim.
            path.unlink()
        except OSError:
            return False
        return self.claim(key, owner)

    def complete(self, key: str, note: str = "") -> None:
        """Mark a document finished so it is never claimed again."""
        done = self._done_path(key)
        try:
            handle = os.open(done, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
        except FileExistsError:
            return
        except OSError:
            return
        with os.fdopen(handle, "w", encoding="utf-8") as stream:
            stream.write(f"{time.time()}\n{note}\n")

    def release(self, key: str) -> None:
        """Give up a claim without completing, so another worker may retry."""
        try:
            self._claim_path(key).unlink()
        except OSError:
            pass

    def stats(self) -> dict:
        claims = sum(1 for _ in self.root.glob(f"*{CLAIM_SUFFIX}"))
        done = sum(1 for _ in self.root.glob(f"*{DONE_SUFFIX}"))
        return {"claimed": claims, "completed": done}


def iter_new_files(folder, claims: ClaimStore, extensions=(".pdf",),
                   poll_s: float = 20.0, quiet_rounds: int = 3,
                   max_wait_s: float = 0.0, owner: str = ""):
    """Yield files to work on as they appear, claiming each one first.

    The folder is still filling while this runs, so a single listing is not
    enough. Polling continues until the folder has produced nothing new for
    ``quiet_rounds`` consecutive passes, which is the signal that the download
    side has finished — no coordination with the downloader is required, which
    keeps the two halves independent and separately restartable.

    Only fully-written files are offered: a file whose size is still changing
    between passes is skipped this round rather than parsed half-downloaded.
    """
    folder = Path(folder)
    seen_sizes: dict = {}
    idle = 0
    started = time.time()

    while True:
        produced = False
        for path in sorted(folder.glob("*")):
            if path.suffix.lower() not in extensions or not path.is_file():
                continue
            try:
                size = path.stat().st_size
            except OSError:
                continue
            if size == 0:
                continue
            # Still growing: leave it for the next pass.
            if seen_sizes.get(path) != size:
                seen_sizes[path] = size
                continue

            key = content_key(path)
            if claims.is_done(key) or not claims.claim(key, owner):
                continue
            produced = True
            yield path, key

        idle = 0 if produced else idle + 1
        if idle >= quiet_rounds:
            return
        if max_wait_s and (time.time() - started) > max_wait_s:
            return
        time.sleep(poll_s)
