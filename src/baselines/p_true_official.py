"""Official / source-faithful p(True).

Sources
-------
Paper: Kadavath et al., Language Models (Mostly) Know What They Know, arXiv:2207.05221
  - Sample answer; show brainstormed samples; ask A) True / B) False; read P(True).

Official open implementation used in Farquhar SE / EVSE experiments:
  github.com/jlko/semantic_uncertainty  (vendored in lucieK-J/EvidentialSemanticEntropy
  @ ba2aefee9cbc5aa015ce6642aa354ac4f02f2f88)
  Files:
    uncertainty/uncertainty_measures/p_true.py :: calculate_p_true, construct_few_shot_prompt
    uncertainty/models/huggingface_models.py :: get_p_true

Official scoring (jlko):
  Append ' A' to prompt ending in 'The possible answer is:'
  Return -NLL of last token = log P(token ' A')
  Uncertainty used in generate_answers.py as:
    p_false_fixed = 1 - exp(log_p_true)   # i.e. 1 - P(A)

Kadavath paper also discusses renormalized P(True) among True/False options.
Discrepancy: jlko primary stored score is log P(A) only (not softmax vs B).
We follow jlko for the published SE/EVSE baseline lineage, AND store both:
  - p_A = exp(log_p_A)           # official jlko probability of token A
  - risk_jlko = 1 - p_A          # p_false_fixed
  - p_true_renorm = softmax(A,B) # Kadavath-style diagnostic (not primary)

Stance adaptation (documented; required):
  question = "What is the stance of the text toward {target}?"
  context / text body = TEXT
  candidate answer = Base stance label (FAVOR|AGAINST|NONE)
  brainstormed answers = M high-T samples from the Base stance prompt
  few-shot pool = SemEval TRAIN only (gold allowed ONLY in few-shot construction,
                  never for the evaluated TEST example)

Scoring adaptation (CRITICAL — matches jlko even for *-chat models):
  Official get_p_true NEVER uses apply_chat_template. It does:
      input_data += ' A'
      tokenized = tokenizer(input_data, return_tensors='pt')
      labels with only last token supervised → return -NLL = log P(last_token)
  Default jlko model is Llama-2-7b-chat and still uses raw completion tokenize.
  Therefore Phi p(True) scoring MUST also use raw completion tokenize(prompt+' A'),
  NOT chat_template wrapping. (Base/DSV/MSP generation may still use chat_template.)
"""
from __future__ import annotations

import math
from typing import Any, Sequence

from utils.common import logsumexp

# Exact suffix appended by jlko huggingface_models.HuggingfaceModel.get_p_true
JLKO_P_TRUE_ANSWER_SUFFIX = " A"
JLKO_P_TRUE_FALSE_SUFFIX = " B"  # diagnostic only; not used in published p_false_fixed

# Defaults from jlko utils.py
DEFAULT_P_TRUE_NUM_FEWSHOT = 20
DEFAULT_NUM_BRAINSTORM = 5  # high-T samples shown; jlko default num_generations=10 —
# we document: use 5 to share M with SE/EVSE sampling in our frozen M=5 protocol,
# OR set 10 for closer jlko default. Config chooses; default config uses 5 shared M
# with explicit note. Actually user said don't simplify — use official 10 for brainstorm
# in p_true even if SE uses M=5. Config: p_true_num_brainstorm=10.

P_TRUE_METHOD = "p_true_official"
P_TRUE_DISPLAY = "p(True)"


def stance_question(target: str) -> str:
    return f"What is the stance of the text toward {target}?"


def build_p_true_query_prompt(
    question: str,
    most_probable_answer: str,
    brainstormed_answers: Sequence[str],
    few_shot_prompt: str = "",
    hint: bool = False,
) -> str:
    """Faithful to jlko calculate_p_true string construction."""
    if few_shot_prompt:
        prompt = few_shot_prompt + "\n"
    else:
        prompt = ""
    prompt += "Question: " + question
    prompt += "\nBrainstormed Answers: "
    for answer in list(brainstormed_answers) + [most_probable_answer]:
        prompt += str(answer).strip() + "\n"
    prompt += "Possible answer: " + str(most_probable_answer).strip() + "\n"
    if not hint:
        prompt += "Is the possible answer:\n"
        prompt += "A) True\n"
        prompt += "B) False\n"
        prompt += "The possible answer is:"
    else:
        prompt += (
            "Do the brainstormed answers match the possible answer? "
            "Respond with A if they do, if they do not respond with B. Answer:"
        )
    return prompt


def build_few_shot_block(
    question: str,
    brainstormed_including_low_t: Sequence[str],
    most_likely: str,
    is_correct: bool,
) -> str:
    """One few-shot example block matching jlko construct_few_shot_prompt."""
    parts: list[str] = []
    parts.append("Question: " + question)
    parts.append("\nBrainstormed Answers: ")
    for response in brainstormed_including_low_t:
        parts.append(f"{str(response).strip()} \n")
    parts.append("Possible answer: " + most_likely + "\n")
    parts.append("Is the possible answer:\n")
    parts.append("A) True\n")
    parts.append("B) False\n")
    parts.append("The possible answer is:")
    parts.append(" A" if is_correct else " B")
    return "".join(parts)


def score_p_true_from_logprob_A(log_p_A: float, log_p_B: float | None = None) -> dict:
    """Convert get_p_true log P(A) into confidence/risk.

    Primary (jlko): risk = 1 - exp(log_p_A)
    Absolute vocabulary probability — NOT A/(A+B) renormalization.
    """
    p_A = float(math.exp(log_p_A))
    out = {
        "method": P_TRUE_METHOD,
        "display_name": P_TRUE_DISPLAY,
        "log_p_A": float(log_p_A),
        "p_A": p_A,
        "risk": float(1.0 - p_A),  # jlko p_false_fixed
        "risk_definition": "1 - P(token ' A')  [jlko p_false_fixed]",
        "official_p_true": True,
        "scoring_mode": "jlko_raw_completion_last_token",
    }
    if log_p_B is not None:
        z = logsumexp([log_p_A, log_p_B])
        p_true_renorm = float(math.exp(log_p_A - z))
        out["log_p_B"] = float(log_p_B)
        out["p_B"] = float(math.exp(log_p_B))
        out["p_true_renorm_AB"] = p_true_renorm
        out["risk_renorm_diagnostic"] = float(1.0 - p_true_renorm)
    return out


def jlko_get_p_true_logprob(
    model: Any,
    tokenizer: Any,
    prompt: str,
    *,
    answer_suffix: str = JLKO_P_TRUE_ANSWER_SUFFIX,
    device: Any = None,
) -> float:
    """Exact jlko HuggingfaceModel.get_p_true scoring.

    Source: jlko/semantic_uncertainty uncertainty/models/huggingface_models.py
            (identical in lucieK-J/EvidentialSemanticEntropy @ ba2aefe)

    Steps:
      1. prompt += ' A'   (string concat; leading space is part of the scored string)
      2. tokenizer(prompt) — raw completion; NO apply_chat_template
      3. labels: only last token supervised (all prior = -100)
      4. return -loss = log P(last_token | context)

    Under Phi, encode(' A') is single token id 362 (ĠA). Multi-token answers are
    NOT sequence-scored in jlko — only the final token of tokenize(prompt+suffix)
    contributes to the loss.
    """
    import torch

    if answer_suffix != JLKO_P_TRUE_ANSWER_SUFFIX and answer_suffix != JLKO_P_TRUE_FALSE_SUFFIX:
        # Allow only the official spaced letter suffixes for this helper.
        raise ValueError(f"unexpected p(True) suffix {answer_suffix!r}")
    input_data = prompt + answer_suffix
    tokenized = tokenizer(input_data, return_tensors="pt")
    input_ids = tokenized["input_ids"]
    if device is not None:
        input_ids = input_ids.to(device)
    else:
        # Match jlko: .to('cuda') when available.
        if torch.cuda.is_available():
            input_ids = input_ids.to("cuda")
    target_ids = input_ids.clone()
    target_ids[0, :-1] = -100
    with torch.no_grad():
        model_output = model(input_ids, labels=target_ids)
    return -float(model_output.loss.item())
