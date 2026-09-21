"""Explicit file snapshots; callers own inventories, identifiers and disclosure policy."""

from dataclasses import dataclass, field
import hashlib
from pathlib import Path
from types import MappingProxyType
from typing import Mapping


def sha256(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _regular(path):
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"Snapshot requires a regular, non-symlink file: {path}")


@dataclass(frozen=True)
class FileSnapshot:
    """Hash an explicit inventory and later reject changed contents or nonregular files.

    This detects changes between checks, not concurrent-write prevention or a
    filesystem transaction. Added files outside the supplied inventory are not
    covered. Hash labels may be private; hashes() does not redact them.
    """

    _files: Mapping[str, Path] = field(repr=False)
    _hashes: Mapping[str, str] = field(repr=False)

    @classmethod
    def capture(cls, files):
        paths = dict(files)
        if any(not isinstance(name, str) or not name for name in paths):
            raise ValueError("Snapshot labels must be nonempty strings")
        # Anchor paths without resolving symlinks away before validation.
        paths = {name: Path(path).absolute() for name, path in paths.items()}
        digests = {}
        for name, path in paths.items():
            _regular(path)
            digests[name] = sha256(path)
            _regular(path)
        return cls(MappingProxyType(paths), MappingProxyType(digests))

    def hashes(self):
        return dict(self._hashes)

    def verify(self):
        for name, path in self._files.items():
            _regular(path)
            if sha256(path) != self._hashes[name]:
                raise ValueError(f"Snapshot file changed: {name}")
            _regular(path)
