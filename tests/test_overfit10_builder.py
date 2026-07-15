# tests/test_overfit10_builder.py
import pandas as pd
from scripts.build_overfit10 import build_overfit

def test_build_overfit_takes_first_n(tmp_path):
    src = tmp_path / "train.parquet"
    # NOTE: pyarrow 23.x cannot serialize empty-dict structs ([{}]) — it raises
    # "Cannot write struct type with no child field". Real veRL rows have
    # non-empty reward_model/extra_info dicts, so use non-empty dicts here.
    df = pd.DataFrame({"data_source": ["prism"]*20, "prompt": range(20),
                       "reward_model": [{"style": "rule"}]*20,
                       "extra_info": [{"index": 0}]*20})
    df.to_parquet(src)
    out = tmp_path / "train_overfit10.parquet"
    build_overfit(str(src), str(out), n=10)
    got = pd.read_parquet(out)
    assert len(got) == 10
    assert list(got.columns) == ["data_source", "prompt", "reward_model", "extra_info"]
    assert got["prompt"].tolist() == list(range(10))   # first 10, deterministic
