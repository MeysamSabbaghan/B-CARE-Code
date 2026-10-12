"""Official NLI wrapper: microsoft/deberta-v2-xlarge-mnli (jlko EntailmentDeberta)."""
from __future__ import annotations

from typing import Any

OFFICIAL_NLI_ID = "microsoft/deberta-v2-xlarge-mnli"
# Labels: 0=contradiction, 1=neutral, 2=entailment (Deberta-MNLI)


class EntailmentDeberta:
    def __init__(self, model_id: str = OFFICIAL_NLI_ID, revision: str | None = None, dtype="float16"):
        import torch
        import torch.nn.functional as F
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self.torch = torch
        self.F = F
        kw = {"local_files_only": False}
        if revision:
            kw["revision"] = revision
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, **{k: v for k, v in kw.items() if k != "dtype"})
        # Prefer safetensors when present (same official weights; avoids torch<2.6 + .bin CVE gate)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_id, use_safetensors=True, **kw
        )
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        dt = torch.float16 if dtype == "float16" else torch.float32
        self.model = self.model.to(device=self.device, dtype=dt)
        self.model.eval()
        self.model_id = model_id

    def check_implication(self, text1: str, text2: str) -> tuple[int, float]:
        inputs = self.tokenizer(text1, text2, return_tensors="pt", truncation=True).to(self.device)
        with self.torch.no_grad():
            logits = self.model(**inputs).logits
            probs = self.F.softmax(logits, dim=-1)[0]
            pred = int(self.torch.argmax(probs).item())
            conf = float(probs[pred].item())
        return pred, conf
