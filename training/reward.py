import json
import re
from collections import Counter, defaultdict
from decimal import Decimal as D
from fractions import Fraction
from itertools import combinations
from chart_eval import content as ev
from chart_eval.parsing import extract

def quantity(number, supplied, evidence, g):
    target = evidence['unit']
    axis = g['axis']

    def simplify(name):
        name = str(name).strip()
        m = re.fullmatch('.+?\\s*\\((?:in\\s+)?([^()]+)\\)', name, flags=re.I)
        return (m[1] if m else name).strip()
    equivalents = {'万人': 'ten thousand persons', '人': 'persons', '万': 'ten thousand', 'metric tons of co2 equivalent': 'tonnes co2e', 'metric tons of co2': 'tonnes co2', 'us$': 'usd', 'us dollars': 'usd', 'us dollar': 'usd'}

    def canonical(name):
        text = simplify(name)
        for (a, b) in equivalents.items():
            text = re.sub(re.escape(a), lambda _: b, text, flags=re.I)
        m = re.match('^(ten thousand|1000|100,000|10,000|1,000)\\s*', text, re.I)
        if m:
            factor = 10000 if m[1].lower() == 'ten thousand' else int(m[1].replace(',', ''))
            (unit, prefix) = unit_pair(text[m.end():])
            return (unit, D(factor) * prefix)
        return unit_pair(text)
    from chart_eval.units import canonical_unit as unit_pair
    if not supplied:
        return number
    axis_name = simplify(axis)
    scaled = bool(re.search('\\b(thousand|million|billion|trillion)s?\\b|(?:^|\\s)(?:1,000|1000)\\s', axis_name, re.I))
    main = axis_name if scaled else target
    (a, fa) = canonical(supplied)
    (b, fb) = canonical(main)
    if a == b:
        return number * fa / fb
    if a in ('grams', 'kilograms', 'tonnes', 'kilotonnes') and b in ('grams', 'kilograms', 'tonnes', 'kilotonnes'):
        mass = {'grams': D('.001'), 'kilograms': D(1), 'tonnes': D(1000), 'kilotonnes': D(1000000)}
        return number * fa * mass[a] / (fb * mass[b])
    if ev.key(supplied) == ev.key(axis):
        return number
    aliases = {ev.key(main)} if scaled else ev.unit_keys(g, target)
    return ev.in_target_units(number, supplied, main, aliases)

def score(raw, g):
    (reads, rank, parse) = extract(raw)
    if isinstance(reads, dict):
        reads = list(reads.items())
    by = defaultdict(list)
    for item in reads if isinstance(reads, list) else []:
        if isinstance(item, dict):
            name = next((item[k] for k in ('entity', 'category', 'label', 'name') if k in item), None)
            val = item.get('value')
            u = item.get('unit')
            if name is None and len(item) == 1:
                (name, val) = next(iter(item.items()))
        elif isinstance(item, (tuple, list)) and len(item) in (2, 3):
            (name, val) = item[:2]
            u = item[2] if len(item) == 3 else None
        else:
            continue
        by[ev.key(name)].append((val, u))
    correct = Counter()
    bad_units = explicit = 0
    missing = 0
    for e in g['evidence']:
        label = ev.key(e['entity'])
        vals = by.get(label, [])
        if not vals:
            missing += 1
            continue
        parsed = []
        has_unit = False
        unit_error = False
        for (v, u) in vals:
            token = str(v).strip().replace('−', '-')
            m = ev.NUM.match(token)
            if not m:
                parsed.append(None)
                continue
            num = ev.numeric(m[0])
            units = [z for z in [token[m.end():].strip(), u] if z]
            has_unit |= bool(units)
            measures = {quantity(num, z, e, g) for z in units} if units else {num}
            if None in measures or len(measures) != 1:
                unit_error = True
                parsed.append(None)
            else:
                parsed.append(measures.pop())
        explicit += has_unit
        bad_units += unit_error
        if len(set(parsed)) == 1 and parsed[0] is not None and (D(e['reading_min']) <= parsed[0] <= D(e['reading_max'])):
            correct[label] = 1
    order = [ev.key(k) for k in g['ordering']]
    n = len(order)
    chosen = [ev.key(k) for k in rank] if isinstance(rank, list) else []
    positions = {k: index for (index, k) in enumerate(chosen) if chosen.count(k) == 1}
    pairs = sum((a in positions and b in positions and (positions[a] < positions[b]) for (a, b) in combinations(order, 2)))
    num = sum(correct.values())
    joint = int(num == n and chosen == order)
    pts = Fraction(80 * num, n) + Fraction(10 * pairs, n * (n - 1) // 2) + 10 * joint
    return dict(correct=num, rank=int(chosen == order), both=joint, pairs=pairs, score=float(pts), exact=str(pts), missing=missing, explicit=explicit, invalid_units=bad_units, parse=parse)

def engineered():
    evidence = [dict(entity=c, value=str(x), reading_min=str(x - D('.5')), reading_max=str(x + D('.5')), unit='million persons') for (c, x) in zip('ABCD', map(D, ['40', '30', '20', '10']))]
    g = dict(ordering=list('ABCD'), evidence=evidence, axis='Population (in million persons)', unit_aliases={})

    def answer(values, rank=list('ABCD')):
        return json.dumps(dict(readings=values, ranking=rank))
    base = [[e['entity'], int(e['value'])] for e in evidence]
    cases = [('all_correct', answer(base), 100), ('wrong_unit', answer(base[:3] + [['D', '10 kg']]), 70), ('equivalent_unit', answer(base[:3] + [['D', '10000000 persons']]), 100), ('wrong_magnitude', answer(base[:3] + [['D', '10 persons']]), 70), ('missing', answer(base[:3]), 70), ('conflict', answer(base + [['D', 15]]), 70), ('swapped_values', answer([['A', 30], ['B', 40], *base[2:]]), 50), ('rank_reversed', answer(base, list('DCBA')), 80), ('rank_missing', answer(base, list('ABC')), 85), ('all_wrong_rank_right', answer([[k, -99] for (k, v) in base]), 10), ('boundary', answer(base[:3] + [['D', 10.5]]), 100), ('outside', answer(base[:3] + [['D', 10.51]]), 70), ('format', repr(json.loads(answer(base))), 100)]
    for (name, raw, expected) in cases:
        actual = score(raw, g)['score']
        assert actual == expected, (name, actual, expected)
    return len(cases)
