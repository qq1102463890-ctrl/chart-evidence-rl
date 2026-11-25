import ast
import collections
import json
from decimal import Decimal as D
from .units import NUM, key, unit_keys, numeric, in_target_units

def extract(raw):
    s = raw.strip()
    start = s.find('{')
    end = s.rfind('}')
    if 0 <= start < end:
        try:
            r = json.loads(s[start:end + 1])
        except ValueError:
            try:
                r = ast.literal_eval(s[start:end + 1])
            except (ValueError, SyntaxError):
                r = {}
        if isinstance(r, dict) and {'readings', 'ranking'} & r.keys():
            return (r.get('readings'), r.get('ranking'))
    return (None, None)

def score(raw, g):
    (reads, ranking) = extract(raw)
    if isinstance(reads, dict):
        reads = list(reads.items())
    vals = collections.defaultdict(list)
    for t in reads if isinstance(reads, list) else []:
        if isinstance(t, dict):
            e = t.get('entity', t.get('category', t.get('label', t.get('name'))))
            v = t.get('value')
            u = t.get('unit')
            if e is None and len(t) == 1:
                (e, v) = next(iter(t.items()))
        elif isinstance(t, (list, tuple)) and len(t) in (2, 3):
            (e, v) = t[:2]
            u = t[2] if len(t) == 3 else None
        else:
            continue
        if e is not None:
            vals[key(e)].append((v, u))
    labels = [key(e) for e in g['ordering']]
    rr = [key(x) for x in ranking] if isinstance(ranking, list) else []
    correct = []
    present = []
    unknown = []
    for e in g['evidence']:
        k = key(e['entity'])
        v = vals[k]
        present.append(bool(v))
        matches = []
        for (x, u) in v:
            s = str(x).strip()
            m = NUM.match(s)
            if not m:
                matches.append(False)
                unknown.append(k)
                continue
            number = numeric(m[0])
            units = [z for z in (s[m.end():].strip(), u) if z]
            ns = {in_target_units(number, z, e['unit'], unit_keys(g, e['unit'])) for z in units} if units else {number}
            if None in ns or len(ns) != 1:
                unknown.append(k)
                matches.append(False)
                continue
            matches.append(D(e['reading_min']) <= ns.pop() <= D(e['reading_max']))
        correct.append(bool(matches) and all(matches))
    return dict(count=sum(correct), present=sum(present), all_read=all(correct), rank_ok=rr == labels, rank_present=bool(rr), both=all(correct) and rr == labels, layout_pending=not isinstance(reads, list) or not isinstance(ranking, list), units_pending=unknown, per_target=correct)
