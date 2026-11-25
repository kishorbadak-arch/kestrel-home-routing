import re
import pandas as pd
CANON = {'Installations': 'Installs & Demo', 'Consumables': 'Filters & Consumables'}
FIX = {'Ã©': 'é', 'Ã¢â‚¬Â¦': '…', 'Ã¢â‚¬â€œ': '–'}   # legacy Zoho mojibake
TEAMS = ['Billing', 'Filters & Consumables', 'Installs & Demo', 'Product Advice',
         'Repairs', 'Returns & Replacement', 'Warranty Claims']

def clean(s):
    s = str(s)
    for k, v in FIX.items():
        s = s.replace(k, v)
    s = s.lower().replace('é', 'e').replace('…', ' ').replace('–', ' ')
    s = re.sub(r'\b(sr|ko)\d+\b', ' ', s)      # order / SR numbers carry no routing signal
    return re.sub(r'\s+', ' ', s).strip()

def features(text, product_family, warranty_status, channel):
    return (clean(text) + ' || ' + str(product_family).replace(' ', '_') + ' '
            + str(warranty_status) + ' ' + str(channel))

def load_train(path='data'):
    t = pd.read_csv(f'{path}/train.csv'); r = pd.read_csv(f'{path}/resolution_log.csv')
    d = t.merge(r, on='request_id')
    d['bot'] = d.team_label.replace(CANON)
    d['y'] = d.final_team.replace(CANON)          # TARGET = where the request really ended up
    d['x'] = [features(*a) for a in zip(d.request_text, d.product_family, d.warranty_status, d.channel)]
    return d
