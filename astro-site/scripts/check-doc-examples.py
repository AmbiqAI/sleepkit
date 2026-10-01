"""Check documented feature contracts and syntax without importing ML runtimes."""

import ast
import re
import textwrap
from pathlib import Path

root = Path("..")
for name in ["fs_w_a_5", "fs_c_ear_9", "fs_w_pa_14", "fs_w_p_5", "fs_h_e_10"]:
    tree = ast.parse((root / "sleepkit/features" / f"{name}.py").read_text())
    function = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == "feature_names")
    names = ast.literal_eval(next(n.value for n in ast.walk(function) if isinstance(n, ast.Return)))
    doc = (root / "astro-site/src/content/docs/features" / f"{name}.mdx").read_text()
    table = re.findall(r"^\| (\w+) \|", doc, re.MULTILINE)
    assert table == names, f"{name}: feature table does not match feature_names()"
    paths = {
        n.args[0].value
        for n in ast.walk(tree)
        if isinstance(n, ast.Call)
        and isinstance(n.func, ast.Attribute)
        and n.func.attr == "create_dataset"
        and n.args
        and isinstance(n.args[0], ast.Constant)
    }
    documented = set(re.findall(r"`(/\w+)`", doc))
    assert documented == paths, f"{name}: HDF5 paths differ: {documented ^ paths}"

for rel in [
    "datasets/byod.md",
    "models/byom.md",
    "features/byofs.md",
    "usage/python.md",
    "modes/demo.md",
    "datasets/cmidss.md",
]:
    text = (root / "astro-site/src/content/docs" / (rel + "x")).read_text()
    for match in re.finditer(r"^([ \t]*)```(?:py|python)(?:[^\n]*)\n(.*?)^\1```", text, re.MULTILINE | re.DOTALL):
        ast.parse(textwrap.dedent(match[2]))
print("Verified feature tables, HDF5 paths and updated Python example syntax.")
