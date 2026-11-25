"""Kestrel routing service. stdlib HTTP server + scikit-learn. No API key, no network calls.
   python app.py  ->  http://localhost:8000   (POST /route, GET /health, GET / for the screen)"""
import json, os, re, sys
from http.server import BaseHTTPRequestHandler, HTTPServer
import joblib, numpy as np
from prep import features, clean

HERE = os.path.dirname(os.path.abspath(__file__))
try:
    BUNDLE = joblib.load(os.path.join(HERE, 'model', 'model.joblib'))
    MODEL, THR = BUNDLE['model'], BUNDLE['threshold']
    LOAD_ERR = None
except Exception as e:                                   # fail politely, never crash
    MODEL, THR, LOAD_ERR = None, 0.8, f'{type(e).__name__}: {e}. Run `python train.py` first.'

PAYMENT = re.compile(r'\b(paid|payment done|paid extra|upi|emi|by card|via card|already paid)\b')
CLARIFY = ('Not enough to route. Ask the customer ONE question: is the product (a) not working / '
           'faulty, (b) a new product needing installation or demo, (c) about a payment or invoice, '
           '(d) a filter/spare part, (e) a delivery or return, (f) warranty/Shield cover, (g) a how-to question?')

def reasons(text, row, team, probs, classes):
    feats = features(row["text"], row["product_family"], row["warranty_status"], row["channel"])
    vec_union = MODEL.steps[0][1]; lr = MODEL.steps[-1][1]
    word_vec = vec_union.transformer_list[0][1]
    X = word_vec.transform([feats]); names = word_vec.get_feature_names_out()
    ci = list(lr.classes_).index(team); nw = len(names)
    contrib = X.toarray()[0] * lr.coef_[ci][:nw]
    top = [names[i] for i in np.argsort(-contrib)[:6] if contrib[i] > 0]
    out = []
    if top: out.append('Wording that pointed here: ' + ', '.join(f'"{t}"' for t in top[:4]))
    t = clean(text)
    if PAYMENT.search(t) and team != 'Billing':
        out.append('Customer mentions payment, but per Ops policy §3 a mention of having paid does not make it Billing; '
                   'Billing only when the payment itself is the problem.')
    if len(t.split()) <= 6 and probs.max() < THR:
        out.append('Message is very short / generic (no fault, order, payment or product-need stated).')
    return out

def route(rec):
    if MODEL is None:
        return {'ok': False, 'error': 'Model not loaded. ' + LOAD_ERR, 'fallback': 'Send to manual triage.'}
    text = str(rec.get('request_text', '')).strip()
    if not text:
        return {'ok': False, 'error': 'request_text is required.'}
    row = dict(text=text, product_family=rec.get('product_family', ''),
               warranty_status=rec.get('warranty_status', ''), channel=rec.get('channel', ''))
    probs = MODEL.predict_proba([features(text, row['product_family'], row['warranty_status'], row['channel'])])[0]; classes = list(MODEL.classes_)
    order = np.argsort(-probs); team = classes[order[0]]; conf = float(probs[order[0]])
    auto = conf >= THR
    return {'ok': True, 'team': team, 'confidence': round(conf, 3),
            'decision': 'auto_route' if auto else 'needs_clarification',
            'alternatives': [{'team': classes[i], 'p': round(float(probs[i]), 3)} for i in order[1:3]],
            'reasons': reasons(text, row, team, probs, classes),
            'next_step': (f'Route to {team}.' if auto else
                          f'Best guess {team} but only {conf:.0%} sure (such guesses are right ~1 in 3 times in testing). ' + CLARIFY),
            'cost_note': 'Runs locally; no per-request fee.', 'model': 'tfidf+logreg, trained on final_team'}

class H(BaseHTTPRequestHandler):
    def _send(self, code, body, ctype='application/json'):
        b = body if isinstance(body, bytes) else json.dumps(body).encode()
        self.send_response(code); self.send_header('Content-Type', ctype); self.send_header('Content-Length', str(len(b)))
        self.end_headers(); self.wfile.write(b)
    def do_GET(self):
        if self.path == '/health': return self._send(200, {'status': 'ok' if MODEL else 'model_missing', 'detail': LOAD_ERR})
        if self.path in ('/', '/index.html'):
            with open(os.path.join(HERE, 'static', 'index.html'), 'rb') as f:
                return self._send(200, f.read(), 'text/html; charset=utf-8')
        self._send(404, {'error': 'not found'})
    def do_POST(self):
        if self.path != '/route': return self._send(404, {'error': 'not found'})
        try:
            rec = json.loads(self.rfile.read(int(self.headers.get('Content-Length', 0))) or b'{}')
            if not isinstance(rec, dict): raise ValueError('body must be a JSON object')
        except Exception as e:
            return self._send(400, {'ok': False, 'error': f'Bad JSON: {e}'})
        r = route(rec); self._send(200 if r.get('ok') else 422, r)
    def log_message(self, *a): pass

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8000))
    print(f'Kestrel routing on http://localhost:{port}' + ('' if MODEL else '  (MODEL MISSING: ' + LOAD_ERR + ')'))
    HTTPServer(('0.0.0.0', port), H).serve_forever()
