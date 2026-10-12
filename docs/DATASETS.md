# Datasets (code only — no data shipped)

B-CARE evaluates **four** TEST sets (12 settings with 3 bases).

| Dataset | Eval N | Labels | Official source |
|---|---:|---|---|
| SemEval-2016 Task 6A | 1249 | FAVOR/AGAINST/NONE | https://alt.qcri.org/semeval2016/task6/ (exclude Donald Trump) |
| VaccineStance | 1000 | FAVOR/AGAINST/NONE | Obtain from VaccineStance / MoPrO-SD authors under their license. **No verified public redistributable URL is packaged.** |
| COVID-19-Stance | 800 | FAVOR/AGAINST/NONE | Glandt et al. 2021 — https://github.com/kh4l3d/COVID-19-Stance |
| P-Stance | 2157 | FAVOR/AGAINST | https://github.com/chuchun8/PStance |

Place raw files under `data/raw/{semeval2016,vaccines,covid_19,PStance}/` then run `scripts/prepare_data.py`.

`prepare_data.py` and `scripts/validate_manifest.py` **fail hard** on wrong sample counts, missing columns, invalid labels/targets, duplicate IDs, empty fields, or bad `text_sha256`. Expected schema/counts: `configs/datasets.yaml`.
