"""Run:  python -m unittest discover -s tests -v   (from the repo root, after `python train.py`)"""
import json, os, sys, threading, unittest, urllib.request
from http.server import HTTPServer
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import app
from prep import clean, TEAMS

class TestCleaning(unittest.TestCase):
    def test_mojibake_repaired(self):
        self.assertEqual(clean('namastÃ©, complaint Ã¢â‚¬Â¦'), 'namaste, complaint')
    def test_ids_removed(self):
        self.assertNotIn('sr75294', clean('help reg no SR75294 order KO2608878'))

class TestRouting(unittest.TestCase):
    def rec(self, text, product='Water Purifier'):
        return app.route({'request_text': text, 'product_family': product,
                          'warranty_status': 'in_warranty', 'channel': 'chat'})
    def test_fault_goes_to_repairs_even_if_paid(self):
        r = self.rec('water purifier not turning on, paid by card')
        self.assertEqual(r['team'], 'Repairs'); self.assertEqual(r['decision'], 'auto_route')
        self.assertTrue(any('§3' in x for x in r['reasons']))      # explains the "I paid" rule
    def test_installation_paid_is_not_billing(self):
        self.assertEqual(self.rec('need installation for my new air fryer, payment done', 'Air Fryer')['team'], 'Installs & Demo')
    def test_vague_message_asks_a_question(self):
        r = self.rec('namaste, need help with my purifier')
        self.assertEqual(r['decision'], 'needs_clarification'); self.assertLess(r['confidence'], 0.8)
    def test_output_shape(self):
        r = self.rec('refund for double charge on my order', 'Air Fryer')
        for k in ('ok', 'team', 'confidence', 'decision', 'alternatives', 'reasons', 'next_step'):
            self.assertIn(k, r)
        self.assertIn(r['team'], TEAMS)
    def test_empty_text_rejected(self):
        self.assertFalse(app.route({'request_text': '  '})['ok'])
    def test_missing_optional_fields_ok(self):
        self.assertTrue(app.route({'request_text': 'filter replacement for purifier'})['ok'])
    def test_no_model_fails_politely(self):
        saved = app.MODEL; app.MODEL = None; app.LOAD_ERR = 'test'
        try:
            r = app.route({'request_text': 'x'})
            self.assertFalse(r['ok']); self.assertIn('manual triage', r['fallback'])
        finally:
            app.MODEL = saved

class TestHTTP(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.srv = HTTPServer(('127.0.0.1', 0), app.H); cls.port = cls.srv.server_port
        threading.Thread(target=cls.srv.serve_forever, daemon=True).start()
    @classmethod
    def tearDownClass(cls): cls.srv.shutdown()
    def post(self, body):
        req = urllib.request.Request(f'http://127.0.0.1:{self.port}/route', data=body, method='POST')
        try:
            with urllib.request.urlopen(req) as r: return r.status, json.load(r)
        except urllib.error.HTTPError as e: return e.code, json.load(e)
    def test_route_ok(self):
        s, j = self.post(json.dumps({'request_text': 'mixer grinder stopped working'}).encode())
        self.assertEqual(s, 200); self.assertEqual(j['team'], 'Repairs')
    def test_bad_json_is_400(self):
        self.assertEqual(self.post(b'not json')[0], 400)
    def test_screen_and_health(self):
        for p in ('/', '/health'):
            with urllib.request.urlopen(f'http://127.0.0.1:{self.port}{p}') as r: self.assertEqual(r.status, 200)

class TestSubmissionFile(unittest.TestCase):
    def test_predictions_match_sample_submission(self):
        p = pd.read_csv(os.path.join(ROOT, 'predictions.csv')); s = pd.read_csv(os.path.join(ROOT, 'data', 'sample_submission.csv'))
        self.assertEqual(list(p.columns), ['request_id', 'team'])
        self.assertEqual(len(p), len(s)); self.assertTrue(p.request_id.is_unique)
        self.assertEqual(set(p.request_id), set(s.request_id))
        self.assertTrue(set(p.team) <= set(TEAMS))

if __name__ == '__main__':
    unittest.main()
