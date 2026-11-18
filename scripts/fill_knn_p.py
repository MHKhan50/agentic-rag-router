"""Write the kNN (oracle label @ tau*) vs always-E McNemar p-value into Table 2 of the paper."""
import json
from pathlib import Path

root = Path(__file__).resolve().parents[1]
res = json.loads((root / "results/02_train_router/results.json").read_text())
key = "kNN (k=15) @ tau* vs always E_iterative_large"
if key not in res["mcnemar"]:
    raise SystemExit(f"'{key}' not in results.json - rerun notebook 02 and download its results first.")
p = res["mcnemar"][key]["p_value"]
cell = f"{p:.4f}" if p < 0.01 else f"{p:.2f}"
tex = root / "paper/main.tex"
old = r"$k$NN, oracle label @ $\tau^*$ & 46.0 & 91.5\% & $+1.0$ & -- \\"
s = tex.read_text()
if old not in s:
    raise SystemExit("kNN row already filled or changed.")
tex.write_text(s.replace(old, old.replace("& -- ", f"& {cell} ")))
print(f"kNN McNemar p = {cell} written to paper/main.tex")
