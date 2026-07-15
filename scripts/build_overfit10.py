# scripts/build_overfit10.py
"""Write the first N rows of a veRL grpo train parquet to an overfit subset."""
import argparse, pandas as pd

def build_overfit(src: str, out: str, n: int = 10) -> None:
    df = pd.read_parquet(src)
    req = ["data_source", "prompt", "reward_model", "extra_info"]
    missing = [c for c in req if c not in df.columns]
    if missing:
        raise ValueError(f"missing veRL columns: {missing}")
    df.head(n).to_parquet(out, index=False)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=10)
    a = ap.parse_args(); build_overfit(a.src, a.out, a.n)
    print(f"wrote {a.out}")
