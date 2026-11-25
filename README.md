# Kestrel Home - service-request routing (Variant B)

## In simple words
This tool reads a customer's message and says which Kestrel team should get it (Repairs, Billing, Installs & Demo, etc.).
When it is sure (about 8 in 10 requests, right 97% of the time) it routes automatically. When the message is too vague
("need help with my purifier") it says so and suggests asking the customer one question instead of guessing.
**Important:** it learns from where requests *actually ended up*, not from the old bot's labels, because the bot itself is only right 77% of the time.
It runs locally on any laptop: Python 3.10+, no API key, no internet after install, no per-request cost.
It is not hosted publicly on purpose (policy section 10: customer data must not be uploaded outside the engagement team).

## Run it (clean machine, no API key, no internet needed after install)
```
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python train.py        # ~1 min. Re-creates model/model.joblib, predictions.csv, evidence files
python app.py          # open http://localhost:8000
```
A trained model is already committed in `model/`, so `python app.py` alone also works.
If the model file is missing the service still starts and answers with a polite error + "send to manual triage".

Tests (13, about 1 second): `python -m unittest discover -s tests -v`

Endpoint: `POST /route` with JSON `{"request_text": "...", "product_family": "...", "warranty_status": "...", "channel": "..."}`
returns team, confidence, `auto_route` / `needs_clarification`, runner-up teams, and plain-English reasons.
`GET /health` reports whether the model loaded. The screen at `/` calls the endpoint.

## Key decisions (made because nobody was available to ask)
1. **Target = where the request ended up (`final_team`), not the bot's `team_label`.** The brief asks to "match the labels at 90%".
   But the labels are the *bot's* guesses, and the bot's guesses agree with the final team only 77% of the time
   (23% of requests were transferred). A model trained on them copies the bot: it matches the labels 95.6% but is right
   about the real destination only 76.6%. So "90% match" is easy and worthless. I report both numbers.
2. **Team names**: renamed teams merged (Installations = Installs & Demo, Consumables = Filters & Consumables; policy §5 says
   responsibilities did not change). `predictions.csv` uses the current names, as all test rows are after 15 Jan 2026.
   `predictions_legacy_names.csv` has the same predictions under the old names in case the scorer wants those.
3. **Validation is forward in time** (train to 31 Mar 2026, test Apr-Jun 2026), like the real test (Jul-Sep 2026). Final model trains on all rows.
4. **Mojibake** from legacy Zoho text (`Ã©`, `Ã¢â‚¬Â¦`, `Ã¢â‚¬â€œ`) is repaired; order/SR numbers stripped. `resolved_at` is not used
   (Zoho times are UTC, unconverted) and `transfers`/`first_team` are not features (not available at creation = leakage).
5. **Vague messages are not forced.** About 1 in 5 requests ("complaint about purifier") carry no routing information.
   The service says so and asks for one clarifying question instead of guessing (threshold 0.80).
6. **No AI API.** Plain TF-IDF + logistic regression. Run cost per request is zero (Finance asked for a written monthly cost).

## Files
`train.py` train/validate/predict - `app.py`, `static/index.html` service and screen - `prep.py` cleaning - `tests/` automated tests
`predictions.csv` submission - `EVIDENCE.md`, `evidence_metrics.json`, `evidence_worst_errors.csv` proof
`MEMO_Ritu.pdf` memo - `SUBMISSION_NOTES.md` expected score, AI-tool disclosure, what to do by hand
`data/` the pack's CSVs. **Policy §10: do not push this repo to a public GitHub; keep it private to the engagement team.**
