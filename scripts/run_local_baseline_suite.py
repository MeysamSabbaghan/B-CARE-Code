#!/usr/bin/env python3
"""Local GPU suite: MSP, PE, DSV, LE, VR, Same-Model evidence, corrected p(True), SE, EVSE.

Computes real per-example risk scores (no placeholder notes).
Frozen decoding comes from configs/models.yaml.
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import yaml  # noqa: E402

from baselines.dsv import build_dsv_prompt, dsv_from_true_false_logprobs  # noqa: E402
from baselines.evse_wrapper import compute_evse  # noqa: E402
from baselines.label_entropy_vr import label_entropy, variation_ratio  # noqa: E402
from baselines.msp_pe import compute_msp_pe  # noqa: E402
from baselines.p_true_official import (  # noqa: E402
    build_few_shot_block,
    build_p_true_query_prompt,
    jlko_get_p_true_logprob,
    score_p_true_from_logprob_A,
    stance_question,
)
from baselines.semantic_entropy_official import semantic_entropy_from_samples  # noqa: E402
from bcare.data.validate import ManifestValidationError, validate_manifest  # noqa: E402
from bcare.models.load import load_base_model_and_tokenizer  # noqa: E402
from bcare.parse_evidence import parse_closed_label, parse_evidence_scores  # noqa: E402
from bcare.risk import evidencerisk_as  # noqa: E402


def _load_train_rows(path: Path, gold_map: dict | None = None) -> list[dict]:
    gold_map = gold_map or {"NEUTRAL": "NONE", "neutral": "NONE", "favor": "FAVOR", "against": "AGAINST"}
    rows = []
    with path.open(encoding="utf-8") as f:
        for i, r in enumerate(csv.DictReader(f)):
            target = r.get("Target") or r.get("target")
            text = r.get("Tweet") or r.get("tweet") or r.get("text") or r.get("Text")
            gold_raw = r.get("Stance") or r.get("stance") or r.get("gold") or r.get("label")
            g = str(gold_raw).strip()
            gold = gold_map.get(g, gold_map.get(g.upper(), g.upper()))
            rows.append(
                {
                    "train_id": r.get("ID") or r.get("id") or r.get("tweet_id") or i,
                    "target": target,
                    "text": text,
                    "gold": gold,
                }
            )
    if not rows:
        raise SystemExit(f"empty TRAIN csv: {path}")
    return rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--base", choices=["Phi", "Mistral", "Gemma"], required=True)
    ap.add_argument("--models-config", type=Path, default=ROOT / "configs/models.yaml")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--max-examples", type=int, default=None)
    ap.add_argument("--expected-n", type=int, default=None, help="Strict manifest n check")
    ap.add_argument("--enable-same-model-evidence", action="store_true")
    ap.add_argument("--enable-sampling", action="store_true", help="LE/VR (+ required for SE/EVSE/pTrue brainstorm)")
    ap.add_argument("--enable-ptrue", action="store_true", help="Requires --train-csv")
    ap.add_argument("--train-csv", type=Path, default=None)
    ap.add_argument("--enable-se-evse", action="store_true", help="Requires sampling + NLI model")
    ap.add_argument(
        "--model-local-path",
        type=Path,
        default=None,
        help="Optional local snapshot directory (overrides HF id / BCARE_MODEL_PATH_*).",
    )
    ap.add_argument(
        "--local-files-only",
        action="store_true",
        help="Refuse Hub downloads (same as BCARE_LOCAL_FILES_ONLY=1).",
    )
    args = ap.parse_args()

    if args.enable_ptrue and args.train_csv is None:
        raise SystemExit("--enable-ptrue requires --train-csv (TRAIN-only few-shot pool)")
    if args.enable_se_evse and not args.enable_sampling:
        raise SystemExit("--enable-se-evse requires --enable-sampling (shared M=5 draws)")
    if args.enable_ptrue and not args.enable_sampling:
        # Still need brainstorm samples; auto-enable sampling semantics for brainstorm
        args.enable_sampling = True
        print("NOTE: enabling sampling for p(True) brainstorm construction", flush=True)

    try:
        validate_manifest(args.manifest, expected_n=args.expected_n)
    except ManifestValidationError as e:
        raise SystemExit(f"manifest validation failed: {e}") from e

    try:
        import torch
    except ImportError as e:
        raise SystemExit("Install torch/transformers (requirements-infer.txt) to run this suite.") from e

    cfg = yaml.safe_load(args.models_config.read_text())
    mcfg = dict(cfg["bases"][args.base])
    if args.local_files_only:
        mcfg["local_files_only"] = True
    dec = cfg["decoding"]
    pt_cfg = dec["p_true"]
    samp = dec["uncertainty_sampling"]
    tern = (ROOT / "prompts/base_ternary.txt").read_text(encoding="utf-8")
    bina = (ROOT / "prompts/base_binary.txt").read_text(encoding="utf-8")
    ver_t = (ROOT / "prompts/bcare_pbce_ternary.txt").read_text(encoding="utf-8")
    ver_b = (ROOT / "prompts/bcare_pbce_binary.txt").read_text(encoding="utf-8")

    rows = list(csv.DictReader(args.manifest.open(encoding="utf-8")))
    if args.max_examples is not None:
        rows = rows[: args.max_examples]

    model_id = mcfg["hf_model_id"]
    revision = mcfg.get("revision")
    print(f"loading {model_id} revision={revision} class={mcfg.get('model_class')}", flush=True)
    try:
        model, tok = load_base_model_and_tokenizer(
            mcfg,
            device_map="auto",
            base_name=args.base,
            local_path=args.model_local_path,
        )
    except ImportError as e:
        raise SystemExit("Install torch/transformers (requirements-infer.txt) to run this suite.") from e
    device = next(model.parameters()).device

    def chat_ids(user: str) -> list[int]:
        ids = tok.apply_chat_template(
            [{"role": "user", "content": user}], tokenize=True, add_generation_prompt=True
        )
        if isinstance(ids, dict):
            ids = ids["input_ids"]
        if hasattr(ids, "tolist"):
            ids = ids.tolist()
        if ids and isinstance(ids[0], list):
            ids = ids[0]
        return [int(x) for x in ids]

    def generate_text(
        user: str,
        *,
        max_new_tokens: int,
        do_sample: bool,
        temperature: float,
        top_p: float,
        seed: int,
        return_token_logprobs: bool = False,
    ):
        random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        prefix = chat_ids(user)
        inp = torch.tensor([prefix], device=device)
        kw = {
            "max_new_tokens": int(max_new_tokens),
            "do_sample": bool(do_sample),
            "pad_token_id": tok.eos_token_id,
            "return_dict_in_generate": True,
            "output_scores": bool(return_token_logprobs),
        }
        if do_sample:
            kw["temperature"] = float(temperature)
            kw["top_p"] = float(top_p)
        with torch.no_grad():
            out = model.generate(inp, **kw)
        new_ids = out.sequences[0, len(prefix) :].tolist()
        text_out = tok.decode(new_ids, skip_special_tokens=True)
        if not return_token_logprobs:
            return text_out, prefix, None
        tok_lps = []
        for t, scores in enumerate(out.scores):
            logp = torch.log_softmax(scores[0].float(), dim=-1)
            tok_lps.append(float(logp[int(new_ids[t])].item()))
        # Append EOS logprob if not already ended (jlko/EVSE habit)
        eos_id = tok.eos_token_id
        if eos_id is not None and (not new_ids or new_ids[-1] != eos_id):
            full = torch.tensor([prefix + new_ids], device=device)
            with torch.no_grad():
                logits = model(full).logits[0, -1].float()
                tok_lps.append(float(torch.log_softmax(logits, dim=-1)[eos_id].item()))
        return text_out, prefix, tok_lps

    def teacher_forced_logprob(prefix_ids: list[int], cont_text: str) -> float:
        cont_ids = tok.encode(cont_text, add_special_tokens=False)
        full = prefix_ids + cont_ids
        inp = torch.tensor([full], device=device)
        with torch.no_grad():
            logits = model(inp).logits[0]
        start = len(prefix_ids) - 1
        s = 0.0
        for i, tid in enumerate(cont_ids):
            s += float(torch.log_softmax(logits[start + i].float(), dim=-1)[tid].item())
        return s

    # ---- p(True) TRAIN few-shot (gold only here) ----
    few_shot_prompt = ""
    if args.enable_ptrue:
        train_rows = _load_train_rows(args.train_csv)
        n_few = int(pt_cfg["num_fewshot"])
        n_brain = int(pt_cfg["num_brainstorm"])
        rng_fs = random.Random(int(pt_cfg["fewshot_seed"]))
        fs_indices = list(range(len(train_rows)))
        rng_fs.shuffle(fs_indices)
        fs_indices = fs_indices[:n_few]
        few_shot_parts: list[str] = []
        for it, ti in enumerate(fs_indices):
            tr = train_rows[ti]
            q = stance_question(tr["target"])
            # Few-shot demos omit tweet text in the question string (paper-disclosed adaptation)
            prompt_u = tern.replace("{target}", tr["target"]).replace("{text}", tr["text"])
            responses = []
            low_raw, _, _ = generate_text(
                prompt_u,
                max_new_tokens=int(dec["base"]["max_new_tokens"]),
                do_sample=True,
                temperature=float(pt_cfg["fewshot_low_t_temperature"]),
                top_p=1.0,
                seed=int(pt_cfg["fewshot_seed"]) + 1000 + ti,
            )
            low_pred, _ = parse_closed_label(low_raw, ("FAVOR", "AGAINST", "NONE"))
            most = low_pred if low_pred else (low_raw.strip() or "NONE")
            responses.append(most)
            for j in range(n_brain):
                raw_j, _, _ = generate_text(
                    prompt_u,
                    max_new_tokens=int(samp["max_new_tokens"]),
                    do_sample=True,
                    temperature=float(pt_cfg["brainstorm_temperature"]),
                    top_p=float(samp["top_p"]),
                    seed=int(pt_cfg["fewshot_seed"]) + 2000 + ti * 100 + j,
                )
                pred_j, _ = parse_closed_label(raw_j, ("FAVOR", "AGAINST", "NONE"))
                responses.append(pred_j if pred_j else (raw_j.strip() or "EMPTY"))
            is_correct = most == tr["gold"]
            block = build_few_shot_block(q, responses, most, is_correct)
            if it != 0:
                few_shot_parts.append("\n")
            few_shot_parts.append(block)
        few_shot_prompt = "".join(few_shot_parts)
        snap = args.out.parent / "ptrue_fewshot_snapshot.txt"
        snap.parent.mkdir(parents=True, exist_ok=True)
        snap.write_text(few_shot_prompt, encoding="utf-8")
        print(f"p(True) few-shot built: n={n_few} brainstorm={n_brain} → {snap}", flush=True)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    seed_base = int(dec["base"]["seed"])
    sample_seed_base = int(samp["seed"])
    M = int(samp["M"])
    n_brain = int(pt_cfg["num_brainstorm"])
    n_new = 0
    pending_nli = False

    with args.out.open("a", encoding="utf-8") as f:
        for i, r in enumerate(rows):
            labels = tuple(r.get("label_space", "FAVOR|AGAINST|NONE").split("|"))
            tmpl = bina if r.get("prompt_kind") == "binary" else tern
            user = tmpl.replace("{target}", r["target"]).replace("{text}", r["text"])
            raw, prefix, _ = generate_text(
                user,
                max_new_tokens=int(dec["base"]["max_new_tokens"]),
                do_sample=False,
                temperature=0.0,
                top_p=1.0,
                seed=seed_base + i,
            )
            pred, status = parse_closed_label(raw, labels)
            seq = {lab: teacher_forced_logprob(prefix, lab) for lab in labels}
            msp = compute_msp_pe(seq)
            cand = pred if pred in labels else labels[0]
            dsv_user = build_dsv_prompt(r["target"], r["text"], cand)
            dsv_prefix = chat_ids(dsv_user)
            dsv = dsv_from_true_false_logprobs(
                teacher_forced_logprob(dsv_prefix, "True"),
                teacher_forced_logprob(dsv_prefix, "False"),
            )
            rec = {
                "example_id": r["example_id"],
                "base": args.base,
                "base_pred": pred,
                "parse_status": status,
                "gold": r.get("gold"),
                "error": int(pred != r.get("gold")) if pred is not None else 1,
                "MSP_risk": msp["MSP_risk"],
                "PredictiveEntropy": msp["PredictiveEntropy"],
                "DSV_risk": dsv["risk"],
            }

            sample_details = []
            responses = []
            log_liks = []
            if args.enable_sampling:
                eid_seed = sample_seed_base + i
                for m in range(M):
                    seed_m = eid_seed + m
                    s_text, _, tok_lps = generate_text(
                        user,
                        max_new_tokens=int(samp["max_new_tokens"]),
                        do_sample=True,
                        temperature=float(samp["temperature"]),
                        top_p=float(samp["top_p"]),
                        seed=seed_m,
                        return_token_logprobs=True,
                    )
                    plab, _ = parse_closed_label(s_text, labels)
                    key = plab if plab in labels else (s_text.strip() or "EMPTY")
                    responses.append(key)
                    log_liks.append(tok_lps or [])
                    sample_details.append(
                        {
                            "m": m,
                            "raw_text": s_text,
                            "parsed": plab,
                            "cluster_key": key,
                            "token_logprobs": tok_lps,
                            "seq_logprob": float(sum(tok_lps or [0.0])),
                            "seed": seed_m,
                        }
                    )
                rec["LabelEntropy"] = label_entropy(
                    [s["parsed"] for s in sample_details], M=M
                )["LabelEntropy"]
                rec["VariationRatio"] = variation_ratio(
                    [s["parsed"] for s in sample_details], M=M
                )["VariationRatio"]
                rec["samples"] = sample_details

            if args.enable_ptrue:
                brainstorm = [s["cluster_key"] for s in sample_details]
                for j in range(max(0, n_brain - M)):
                    seed_b = sample_seed_base + 10_000 + i * 100 + j
                    s_text, _, _ = generate_text(
                        user,
                        max_new_tokens=int(samp["max_new_tokens"]),
                        do_sample=True,
                        temperature=float(pt_cfg["brainstorm_temperature"]),
                        top_p=float(samp["top_p"]),
                        seed=seed_b,
                    )
                    plab, _ = parse_closed_label(s_text, labels)
                    brainstorm.append(plab if plab in labels else (s_text.strip() or "EMPTY"))
                q_full = stance_question(r["target"]) + f" Text: {r['text']}"
                pt_prompt = build_p_true_query_prompt(
                    q_full,
                    cand,
                    brainstorm[:n_brain],
                    few_shot_prompt=few_shot_prompt,
                    hint=False,
                )
                log_p_A = jlko_get_p_true_logprob(model, tok, pt_prompt, device=device)
                log_p_B = jlko_get_p_true_logprob(
                    model, tok, pt_prompt, answer_suffix=" B", device=device
                )
                ptrue = score_p_true_from_logprob_A(log_p_A, log_p_B)
                rec["p_true_risk"] = ptrue["risk"]
                rec["p_true"] = ptrue
                rec["brainstorm_for_ptrue"] = brainstorm[:n_brain]

            if args.enable_same_model_evidence:
                vtmpl = ver_b if r.get("prompt_kind") == "binary" else ver_t
                vuser = vtmpl.replace("{target}", r["target"]).replace("{text}", r["text"])
                # Prefer example_id for seed (matches original suite when ids are 0..n-1)
                try:
                    eid_num = int(r["example_id"])
                except (TypeError, ValueError):
                    eid_num = i
                vraw, _, _ = generate_text(
                    vuser,
                    max_new_tokens=int(dec["same_model_evidence"]["max_new_tokens"]),
                    do_sample=False,
                    temperature=0.0,
                    top_p=1.0,
                    seed=int(dec["same_model_evidence"]["seed"]) + eid_num,
                )
                ev = parse_evidence_scores(vraw, labels)
                rec["same_model_evidence"] = ev
                if ev and pred in labels:
                    rec["SameModel_EvidenceRisk"] = evidencerisk_as(ev, pred, labels)

            if args.enable_se_evse:
                pending_nli = True
                q_ctx = f"Stance of the text toward '{r['target']}': {r['text']} Answer:"
                rec["_pending_nli"] = {
                    "responses": responses,
                    "log_liks": log_liks,
                    "seq_logprobs": [float(sum(x)) for x in log_liks],
                    "q_ctx": q_ctx,
                }
                rec["SemanticEntropy"] = None
                rec["EVSE"] = None

            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            f.flush()
            n_new += 1
            if n_new % 10 == 0:
                print(f"suite wrote {n_new}", flush=True)

    if pending_nli:
        print("Unloading generative model for official NLI pass (SE/EVSE)...", flush=True)
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        from nli.deberta_entailment import EntailmentDeberta

        nli = EntailmentDeberta(model_id=cfg["nli"]["hf_model_id"], dtype="float16")
        lines = args.out.read_text(encoding="utf-8").splitlines()
        new_lines = []
        for line in lines:
            rec = json.loads(line)
            pend = rec.pop("_pending_nli", None)
            if pend is None:
                new_lines.append(json.dumps(rec))
                continue
            responses = pend["responses"]
            q_ctx = pend["q_ctx"]
            nli_strings = [f"{q_ctx} {r}" for r in responses]

            def check(a, b, _nli=nli):
                return _nli.check_implication(a, b)

            se = semantic_entropy_from_samples(
                responses, pend["seq_logprobs"], check, nli_context_prefix=q_ctx
            )
            ev = compute_evse(responses, pend["log_liks"], check, nli_strings=nli_strings)
            rec["SemanticEntropy"] = se["SemanticEntropy"]
            rec["se_detail"] = {
                k: se[k]
                for k in (
                    "n_clusters",
                    "n_unique_responses",
                    "degeneracy_toward_label_entropy",
                    "clustering",
                    "nli_model",
                )
            }
            rec["EVSE"] = ev["EVSE"]
            rec["evse_detail"] = {
                "nli_model": ev["nli_model"],
                "frame_omega_size": ev["frame_omega_size"],
                "n_unique_responses": len(ev["unique_responses"]),
            }
            new_lines.append(json.dumps(rec))
        args.out.write_text("\n".join(new_lines) + "\n", encoding="utf-8")
        print("SE/EVSE NLI pass complete", flush=True)

    print(f"done suite base={args.base} n={n_new} out={args.out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
