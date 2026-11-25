from decimal import Decimal
import re
import unicodedata

NUM = re.compile('^[+-]?(?:\\d[\\d,]*\\.?\\d*|\\.\\d+)(?:[eE][+-]?\\d+)?')

def key(s):
    return ' '.join(unicodedata.normalize('NFKC', str(s)).strip(' \t\r\n*.,;!').casefold().split())

def unit_keys(r, unit):
    return {key(unit), *(key(u) for u in r.get('unit_aliases', {}).get(unit, []))}

EQUIVALENTS = {'%': 'percent', 'percentage': 'percent', 'pct': 'percent', 'per cent': 'percent', 'h': 'hours', 'hr': 'hours', 'hrs': 'hours', 'hour': 'hours', 'hours/week': 'hours per week', 'hr/week': 'hours per week', 'kg': 'kilograms', 'kilogram': 'kilograms', 'g': 'grams', 'gram': 'grams', 't': 'tonnes', 'tonne': 'tonnes', 'metric tons': 'tonnes', 'metric ton': 'tonnes', 'people': 'persons', 'person': 'persons', 'day': 'days', 'year': 'years', 'u.s. dollars': 'usd', 'us dollars': 'usd', 'us dollar': 'usd', 'u.s. dollar': 'usd', 'euro': 'euros', '€': 'euros', 'sq ft': 'square feet', 'm3': 'm³', 'cubic metres': 'm³', 'cubic meters': 'm³', 'percentage points': 'percentage points', 'percentage point': 'percentage points'}

PREFIXES = {'thousand': 1000, 'million': 1000000, 'billion': 1000000000, 'trillion': 1000000000000}

SYMBOLS = {'mW': ('watts', '0.001'), 'W': ('watts', 1), 'kW': ('watts', 1000), 'MW': ('watts', 10 ** 6), 'GW': ('watts', 10 ** 9), 'TW': ('watts', 10 ** 12), 'mWh': ('watt-hours', '0.001'), 'Wh': ('watt-hours', 1), 'kWh': ('watt-hours', 1000), 'MWh': ('watt-hours', 10 ** 6), 'GWh': ('watt-hours', 10 ** 9), 'TWh': ('watt-hours', 10 ** 12)}

def numeric(value):
    return Decimal(str(value).replace(',', '').strip())

def canonical_unit(v):
    text = str(v).strip()
    if text in SYMBOLS:
        (unit, multiplier) = SYMBOLS[text]
        return (unit, Decimal(str(multiplier)))
    scaled = key(v)
    amount = 1
    for (prefix, factor) in PREFIXES.items():
        m = re.match('^' + prefix + 's?(?:\\s+of)?(?:\\s+|$)', scaled)
        if m:
            scaled = scaled[m.end():].strip()
            amount = factor
            break
    return (EQUIVALENTS.get(scaled, scaled), Decimal(amount))

def in_target_units(number, supplied, target, aliases):
    (su, sf) = canonical_unit(supplied)
    if str(supplied).strip() not in SYMBOLS and key(supplied) in aliases:
        return number
    for allowed in [target, *aliases]:
        (tu, tf) = canonical_unit(allowed)
        if su == tu or (not su and sf == tf and (sf != 1)):
            return number * sf / tf
        if tu == 'percent' and su in {'fraction', 'ratio', 'proportion'}:
            return number * 100
        if su == 'percent' and tu in {'fraction', 'ratio', 'proportion'}:
            return number / 100
    return None
