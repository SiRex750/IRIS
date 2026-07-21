# 12_freeze_test_configuration

**Purpose:** Write TEST_CONFIGURATION_FROZEN.json from the val_confirm-selected config (only after 04-09 complete).

**CLI signature:** `python commands/12_freeze_test_configuration.py --config --out-dir [--out-dir <path>]`
(refuses reuse of an existing --out-dir; prints git SHA + config hash; tees to a timestamped log)

**Status:** NOT EXECUTED (also not applicable yet -- no tuning has run)
