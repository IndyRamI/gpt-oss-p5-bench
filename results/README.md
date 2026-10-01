# Results

One directory per run (`<YYYYmmdd-HHMM>-<host>/`), produced by `scripts/run_matrix.sh`:

```
<run>/experiment.yaml            config snapshot
<run>/env.txt                    driver, GPU topology, package versions
<run>/REPORT.md                  findings, capacity table, charts, tables  <- start here
<run>/summary.csv                one row per (topology, phase, concurrency)
<run>/<topology>/<phase>/        server.json, gpu.csv, launch/loadgen logs, per-level raw JSON
```

## Cross-run summary

_Add one line per run: date, hardware, what changed, and the headline result._
