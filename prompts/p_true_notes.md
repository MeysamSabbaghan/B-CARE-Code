# Official p(True) — Stance Adaptation (documented BEFORE evaluation)

## Mapping
| Official field | Stance field |
|----------------|--------------|
| Question | `What is the stance of the text toward {target}?` |
| Context / passage | `{text}` (also available in Base prompt samples) |
| Possible / candidate answer | Base greedy prediction ∈ {FAVOR, AGAINST, NONE} |
| Brainstormed Answers | High-T samples (T=1.0) from the **same Base no-hint stance prompt** |
| Few-shot examples | SemEval **TRAIN** only; gold used solely to label A/B for few-shot blocks |

## Prompt tail (jlko / Kadavath A/B format)
```
Is the possible answer:
A) True
B) False
The possible answer is:
```
Score via jlko `get_p_true`: raw `tokenize(prompt + ' A')`, last-token
absolute log-prob only. **No** `apply_chat_template` on the p(True) string
(even though Phi is a chat model — jlko also scores Llama-2-chat this way).

## Risk
`risk = 1 - P(A)`  (= jlko `p_false_fixed`)

## Non-TEST leakage
Evaluated TEST example never receives gold. Few-shot gold is TRAIN-only.
