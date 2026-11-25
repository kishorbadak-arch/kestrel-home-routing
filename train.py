"""Train, validate, write predictions.csv + model + evidence.  python train.py"""
import json, joblib, numpy as np, pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline, make_union
from prep import *

def build():
    vec = make_union(TfidfVectorizer(ngram_range=(1, 3), sublinear_tf=True, min_df=2),
                     TfidfVectorizer(analyzer='char_wb', ngram_range=(3, 5), sublinear_tf=True, min_df=3))
    return make_pipeline(vec, LogisticRegression(C=1, max_iter=3000))

d = load_train()
CUT = '2026-04-01'                       # forward-in-time split, like the real test (Jul-Sep 2026)
tr, va = d[d.created_at_ist < CUT], d[d.created_at_ist >= CUT]
m = build().fit(tr.x, tr.y)
p = m.predict(va.x); conf = m.predict_proba(va.x).max(1)
va = va.assign(p=p, conf=conf, ok=p == va.y)
seen = va.request_text.isin(tr.request_text)
THR = 0.80
auto = va.conf >= THR
res = {
 'n_train': len(tr), 'n_val': len(va),
 'bot_vs_outcome_val': float((va.bot == va.y).mean()),
 'bot_vs_outcome_all': float((d.bot == d.y).mean()),
 'model_vs_outcome_val': float(va.ok.mean()),
 'model_vs_outcome_val_unseen_text': float(va[~seen].ok.mean()), 'share_unseen_text': float((~seen).mean()),
 'threshold': THR, 'auto_share': float(auto.mean()), 'auto_acc': float(va[auto].ok.mean()),
 'review_share': float((~auto).mean()), 'review_acc_if_forced': float(va[~auto].ok.mean()),
 'bot_acc_on_auto_rows': float((va[auto].bot == va[auto].y).mean()),
 'bot_acc_on_review_rows': float((va[~auto].bot == va[~auto].y).mean()),
}
# model trained on the BOT labels (what the brief literally asks for)
mb = build().fit(tr.x, tr.bot); pb = mb.predict(va.x)
res['botclone_match_bot_val'] = float((pb == va.bot).mean()); res['botclone_vs_outcome_val'] = float((pb == va.y).mean())
res['per_team_recall'] = va.groupby('y').ok.mean().round(3).to_dict()
cal = va.groupby(pd.cut(va.conf, [0, .5, .8, .95, 1]), observed=True).ok.agg(['mean', 'size']).round(3)
res['calibration'] = {str(k): v for k, v in cal.to_dict('index').items()}
res['errors_by_true_team'] = pd.crosstab(va.y, va.p).to_dict()
json.dump(res, open('evidence_metrics.json', 'w'), indent=1, default=str)
va[['request_id', 'request_text', 'y', 'p', 'conf']].query('p != y').sort_values('conf', ascending=False).head(60).to_csv('evidence_worst_errors.csv', index=False)

# final model on ALL labelled rows -> predict test
final = build().fit(d.x, d.y)
te = pd.read_csv('data/test_unlabelled.csv')
Xt = [features(*a) for a in zip(te.request_text, te.product_family, te.warranty_status, te.channel)]
te['team'] = final.predict(Xt); te['conf'] = final.predict_proba(Xt).max(1)
te[['request_id', 'team']].to_csv('predictions.csv', index=False)
te[['request_id', 'team', 'conf']].to_csv('predictions_with_confidence.csv', index=False)
joblib.dump({'model': final, 'threshold': THR}, 'model/model.joblib')
print(json.dumps(res, indent=1, default=str))
print('test auto share', (te.conf >= THR).mean(), te.team.value_counts().to_dict())
