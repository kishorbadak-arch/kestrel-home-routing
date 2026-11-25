# Evidence (all numbers from `train.py`; validation = Apr-Jun 2026, trained only on earlier requests)

| Question | Result |
|---|---|
| How often is the **bot** right about where a request ends up? | **77.2%** overall (76.7% in the validation months). 23% needed a transfer; mean 1.44 transfers per wrong one. |
| Model trained on bot labels (what the brief literally asks for) | matches the bot 95.6% - but is right about the real destination only **76.6%**. It is a copy of the bot, mistakes included. |
| Model trained on where requests ended up | **84.3%** right on the real destination (84.3% on texts never seen in training, so not memorisation). |
| Requests the model is confident about (>=80%) | **79.6%** of volume, **97.1% right** (bot on the same requests: 87.1%). |
| Requests it is not confident about | 20.4% of volume. Forced guess right **34%** (bot on the same: 36%). Nobody can route these from the text. |
| Confidence is honest | <50%: right 23% - 50-80%: 70% - 80-95%: 97% - >95%: 98%. |
| Per team recall | Repairs 91%, Billing 86%, Installs 84%, Filters 83%, Returns 83%, Warranty 80%, Advice 78%. |

## Where it still fails (how often, and why)
* **About 1 in 5 requests are too vague** ("complaint about purifier", "need help with my heater"). The final team for these is close to a coin toss (see `evidence_worst_errors.csv`, and the 34% above). More modelling does not fix this; a question at intake does.
* **~3% error even when confident**: the outcome data itself has a noise floor (about 1% of every bot-routed team lands on every other team at random).
* Water purifiers are where the bot is worst (66.5% right): breakdowns sent to Filters, which matches Meenal's note.
* Forced single-label ceiling on this data is about 84%; 90% against real outcomes is not reachable without more information per request.

## What was tried / discarded
1. Bot-label target (discarded: copies the bot's 23% error). 2. Word n-grams + char n-grams, logistic regression, C = 1, 5, 20 (C=1 best, 84.3 / 84.2 / 83.8). 3. Cleaning mojibake and dropping order numbers (kept). 4. Gradient boosting / LLM API (not tried: no signal left to gain on vague rows and an API bill would break Finance's requirement).
