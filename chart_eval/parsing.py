import json
import re
from . import content as ev

def extract(raw):
    (reads, order) = ev.extract(raw)
    if reads is not None and order is not None:
        return (reads, order, 'parsed')
    (repaired, n) = re.subn('("[+-]?[0-9]+(?:\\.[0-9]+)?)(?=\\])', '\\1"', raw)
    if n:
        (reads, order) = ev.extract(repaired)
        if reads is not None and order is not None:
            return (reads, order, 'quote_repaired')
    decoder = json.JSONDecoder()

    def field(name):
        m = re.search('["\\\']' + name + '["\\\']\\s*:\\s*', raw)
        if not m:
            return None
        s = raw[m.end():].lstrip()
        try:
            return decoder.raw_decode(s)[0]
        except ValueError:
            pass
        if not s.startswith('['):
            return None
        items = []
        s = s[1:].lstrip()
        while s:
            try:
                (v, end) = decoder.raw_decode(s)
            except ValueError:
                break
            items.append(v)
            s = s[end:].lstrip()
            if not s.startswith(','):
                break
            s = s[1:].lstrip()
        return items
    return (field('readings') if reads is None else reads, field('ranking') if order is None else order, 'inspect')
