# 01_validate_dataset

**Purpose:** Structural validation of dataset_manifest.json entries: file existence, id uniqueness, span bounds, val/test id overlap.

**CLI signature:** `python commands/01_validate_dataset.py --split {val_tune,val_confirm,official_test} [--out-dir <path>]`
(refuses reuse of an existing --out-dir; prints git SHA + config hash; tees to a timestamped log)

**Status:** EXECUTED_DURING_SETUP for the placeholder subset only; official_test blocked (see dataset_manifest.json)
