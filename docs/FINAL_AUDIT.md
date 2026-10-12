# B-CARE anonymous release — packaging summary

Public companion to `SOURCE_REPRO_AUDIT.md`. No public push from this package.

## Offline PASS/FAIL

| Check | Result |
|---|---|
| `verify_release.py` | Must PASS before distribution |
| `pytest` (14 tests) | Must PASS before distribution |
| CLI `--help` smoke | Must PASS |
| Config / 12-setting parse | Must PASS |
| Anonymity / secret scan | Must PASS |
| GPU / OpenRouter / raw datasets | **NOT RUN** in packaging |

## Limitations

VaccineStance public mirror; OpenRouter revision unknowable. See `SOURCE_REPRO_AUDIT.md`.
