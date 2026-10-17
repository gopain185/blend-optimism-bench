"""python -m blendsim [outdir]: run every experiment, print the summaries and save the raw tables."""

import sys
import time
from pathlib import Path

from . import experiments as ex


def main(out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    for name, fn, by in [("exp1_pool_size", ex.pool_size, "k"), ("exp2_n_rows", ex.n_rows, "n"),
                         ("exp3_headroom", ex.headroom, "ceiling")]:
        t = time.time()
        df = fn()
        df.to_csv(out / f"{name}.csv", index=False)
        print(f"\n== {name} ({time.time() - t:.0f}s)\n{ex.summarise(df, by).to_string(index=False)}", flush=True)


if __name__ == "__main__":
    main(Path(sys.argv[1] if len(sys.argv) > 1 else "results"))
