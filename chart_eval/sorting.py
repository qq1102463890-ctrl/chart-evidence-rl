from . import content as ev

def sort_readings(reads, ranking, targets):
    if not isinstance(reads, list):
        return (ranking, 'incomplete')
    expected = {ev.key(t) for t in targets}
    values = {}
    labels = {}
    for pair in reads:
        if not isinstance(pair, list) or len(pair) != 2:
            return (ranking, 'incomplete')
        (label, number) = pair
        k = ev.key(label)
        if k not in expected or k in values:
            return (ranking, 'ambiguous_labels')
        text = str(number).strip()
        m = ev.NUM.fullmatch(text)
        if m is None:
            return (ranking, 'non_numeric')
        v = ev.numeric(text)
        if not v.is_finite():
            return (ranking, 'non_numeric')
        values[k] = v
        labels[k] = label
    if set(values) != expected:
        return (ranking, 'incomplete')
    old = [ev.key(t) for t in ranking] if isinstance(ranking, list) else []
    priority = old if len(old) == len(expected) and set(old) == expected else list(values)
    keys = sorted(priority, key=values.__getitem__, reverse=True)
    return ([labels[k] for k in keys], 'ties' if len(set(values.values())) != len(values) else 'sorted')
def sort_targets(reads, ranking, targets):
    if not isinstance(reads, list):
        return (ranking, 'incomplete')
    expected = {ev.key(t) for t in targets}
    unique = {}
    numbers = {}
    for item in reads:
        if not isinstance(item, list) or len(item) != 2:
            return (ranking, 'incomplete')
        (label, raw) = item
        k = ev.key(label)
        if k not in expected:
            continue
        s = str(raw).strip()
        if not ev.NUM.fullmatch(s):
            return (ranking, 'non_numeric')
        value = ev.numeric(s)
        if k in numbers and value != numbers[k]:
            return (ranking, 'conflicting_readings')
        numbers[k] = value
        unique[k] = item
    return sort_readings(list(unique.values()), ranking, targets)
