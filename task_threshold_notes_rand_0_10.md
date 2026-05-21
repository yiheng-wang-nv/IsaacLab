# Task Progress Threshold Notes (rand_0_10)

Source data: timeout-only `rand_0_10` runs, using task 3/4 final progress as an offline filter for final episode success.

## Current Decisions

| open_loop_steps | task3 | task4 | status |
|---:|---:|---:|---|
| 4 | 0.9999 | 0.95 | Tested online: 52/100 success. Keep this for now. |
| 8 | 0.9999 | 0.98 | Tested online: 62/100 success. Keep this for now. |

## Chunk 2 Candidates

Timeout-only chunk2 baseline: 41/100 success.

Offline filter comparison:

| task3 | task4 | TP | FP | TN | FN | note |
|---:|---:|---:|---:|---:|---:|---|
| 0.9995 | 0.95 | 40 | 3 | 56 | 1 | Best recall; likely first choice to test. |
| 0.9995 | 0.90 | 40 | 3 | 56 | 1 | Same as 0.95 in this data. |
| 0.9999 | 0.95 | 35 | 2 | 57 | 6 | More conservative; blocks more successes. |
| 0.9999 | 0.98 | 34 | 2 | 57 | 7 | Too conservative for chunk2. |

Recommended next test for chunk2:

```bash
--auto_task_thresholds 0.999,0.999,0.9995,0.95,0.95
```

## Notes

- `--auto_task_thresholds` controls the direct auto benchmark. `--task3_complete_threshold` does not control `--auto_multistage_direct`.
- Task4 retry is not local to task4. A task4 threshold failure returns to task3 start and reruns task3.
