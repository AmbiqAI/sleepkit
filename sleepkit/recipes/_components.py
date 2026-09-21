"""Explicit recipe imports and provenance inventory for locally incubated blocks.

Replace these imports only after a pinned heliaEDGE implementation passes the
same conformance tests. No environment-dependent automatic fallback.
"""

from pathlib import Path

from sleepkit._edge_candidates.classification import ClassificationAccumulator, summarize_confusion
from sleepkit._edge_candidates.evidence import FileSnapshot

__all__ = ["ClassificationAccumulator", "summarize_confusion", "FileSnapshot", "implementation_files"]


def implementation_files(recipe_directory):
    """Selected recipe plus its shared blocks/artifact code, with stable local labels."""
    package = Path(__file__).resolve().parents[1]
    files = {p.name: p for p in sorted(Path(recipe_directory).glob("*.py"))}
    files["components.py"] = Path(__file__)
    for directory, prefix in ((package / "_edge_candidates", "shared"), (package / "artifacts", "artifacts")):
        files.update({f"{prefix}/{p.name}": p for p in sorted(directory.glob("*.py"))})
    return files
