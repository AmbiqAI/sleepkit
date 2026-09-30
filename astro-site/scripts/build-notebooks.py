"""Publish saved notebook cells and outputs without running Python code."""

import base64
import json
import re
import shutil
from pathlib import Path

source = Path("../notebooks/train-detect-model.ipynb")
notebook = json.loads(source.read_text())
assets = Path("public/notebooks")
shutil.rmtree(assets, ignore_errors=True)
assets.mkdir(parents=True, exist_ok=True)
shutil.copyfile(source, assets / source.name)
shutil.copyfile(Path("../docs/guides/train-detect-model.ipynb"), assets / "previous-docs-train-detect-model.ipynb")
parts = [
    "---\ntitle: Train Sleep Detection Model\ndescription: Train a wrist-based sleep detection model with saved notebook code and outputs.\n---",
    f'<div class="sleepkit-actions"><a class="md-button" href="/sleepkit/notebooks/{source.name}">Download notebook</a> <a class="md-button" href="https://github.com/AmbiqAI/sleepkit/blob/main/notebooks/{source.name}">View source</a> <a class="md-button" href="https://colab.research.google.com/github/AmbiqAI/sleepkit/blob/main/notebooks/{source.name}">Open in Colab</a></div>',
    "This page uses the repository notebook, whose saved training and export logs date to December 2025. The [previous documentation copy](/sleepkit/notebooks/previous-docs-train-detect-model.ipynb) is retained as an archive; it contains the same code but mixes older saved outputs.",
    "Before running, set `SK_DATASET_PATH` to your dataset directory. The fallback `../../datasets` is relative to the notebook kernel’s working directory, so it may need adjustment after downloading or opening in Colab.",
    "This page displays saved notebook outputs; building the site does not run training.",
]
for ci, cell in enumerate(notebook["cells"]):
    text = "".join(cell.get("source", []))
    if cell["cell_type"] == "markdown":
        if ci == 0 and "View in Colab" in text:
            continue
        text = re.sub(r"^# Train Sleep Detection Model\s*", "", text)
        text = re.sub(r":[\w]+-[\w-]+:", "", text)
        text = re.sub(r"\{\s*\.[^}]+\}", "", text)
        text = text.replace(" markdown", "").replace("../features/fs_w_a_5.md", "/sleepkit/features/fs_w_a_5/")
        parts.append(re.sub(r"^# ", "## ", text, flags=re.MULTILINE))
    elif cell["cell_type"] == "code":
        parts.append("```python\n" + text.rstrip() + "\n```")
        for oi, output in enumerate(cell.get("outputs", [])):
            data = output.get("data", {})
            if "image/png" in data:
                name = f"{source.stem}-{ci}-{oi}.png"
                (assets / name).write_bytes(base64.b64decode("".join(data["image/png"])))
                parts.append(f"![Saved figure from notebook cell {ci + 1}](/sleepkit/notebooks/{name})")
            else:
                # Plain-text fallbacks retain tables without executing saved HTML scripts.
                raw = "".join(output.get("text", data.get("text/plain", [])))
                raw = re.sub(r"\x1b\][^\x07\x1b]*(?:\x07|\x1b\\)", "", raw)
                raw = re.sub(r"\x1b\[[0-?]*[ -/]*[@-~]", "", raw)
                if raw.strip():
                    block = "```text\n" + raw.rstrip() + "\n```"
                    parts.append(
                        "<details>\n<summary>Saved output</summary>\n\n" + block + "\n\n</details>"
                        if len(raw.splitlines()) > 12
                        else block
                    )
Path("src/content/docs/guides/train-detect-model.md").write_text("\n\n".join(parts) + "\n")
