# Command Templates (NOT EXECUTED)

Commands 00-02 are the only ones this setup task actually ran, and only in their dry-run/self-test
form (environment validation, dataset structural check, synthetic unit tests). Commands 03-20 are
templates only — no video was decoded, no model was queried, and no real metric was computed by
running any of them.

Every command template below follows this contract:
- accepts `--config <path>` (explicit, no implicit default)
- accepts `--split {val_tune,val_confirm,official_test}` (explicit, no implicit default — never
  silently defaults to a test split)
- accepts `--out-dir <path>` and refuses to run if `<path>` already exists (no output-dir reuse)
- prints `git rev-parse HEAD` and a hash of the resolved config at the top of its log
- tees stdout/stderr to `<out-dir>/log_<UTC_TIMESTAMP>.txt`
- exits non-zero immediately (before any heavy work) if a required dependency/model/file is missing
- for tuning commands (04-08), calls `scripts/split_guard.py::guard_tuning_command` first
- for the official-test command (13), calls `scripts/split_guard.py::guard_official_test_command` first

See individual `NN_<name>.md` files for the exact CLI signature and current executability status.
