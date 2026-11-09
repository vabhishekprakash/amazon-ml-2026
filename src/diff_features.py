"""Pair difference features: one-to-one name-token alignment, address-number roles, directional address agreement.

Every value comes from the two records' text and token rarities among Source 1 records of the same country.
"""
import collections, math, os, re
import numpy as np
from rapidfuzz.distance import Levenshtein
import config as C
from src.er_norm import FILLER, normalize_name, postcodes, strip_accents

NAME_COLS = ["nm_exact", "nm_typo", "nm_abbrev", "nm_reordered", "nm_unm_w_s1", "nm_unm_w_t", "nm_max_unm_w_s1",
             "nm_max_unm_w_t", "nm_replaced", "nm_added", "nm_dropped", "nm_many_to_one", "nm_typo_w", "nm_kept_w_frac"]
ADDR_COLS = ["pc_state", "house_state", "unit_state", "house_missing_one", "house_suffix_conflict", "compound_either",
             "house_gap", "ad_shared_over_s1", "ad_shared_over_t", "ad_unm_w_s1", "ad_unm_w_t", "ad_max_unm_w_t",
             "ad_fuzzy_rare", "ad_street_run"]
_NONALNUM = re.compile(r"[^a-z0-9]+")
_UNIT = re.compile(r"\b(?:apt|apartment|suite|ste|unit|fl|floor|flat|room|rm|bureau|etage|bat|batiment|door)\s*(?:no\s*)?(\d+[a-z]?)\b")
_HOUSE = re.compile(r"\b(\d{1,5})(?:\s?(bis|ter|[a-z])\b)?")
_COMPOUND = re.compile(r"\b\d+\s*[-/]\s*\d+\b")


def _addr_tokens(raw):
    """Lowercase ASCII alphanumeric tokens of an address."""
    return _NONALNUM.sub(" ", strip_accents(raw.lower())).split()


def rarity_tables(split):
    """Per-country document counts of Source 1 name tokens (fillers removed) and address tokens."""
    name_df, addr_df, n = collections.Counter(), collections.Counter(), collections.Counter()
    with open(C.source_file(split, 1), encoding="utf-8") as f:
        next(f)
        for line in f:
            q = line.rstrip("\n").split("\t")
            if len(q) != 4:
                continue
            c = q[3]; n[c] += 1
            for t in set(normalize_name(q[1]).split()) - FILLER:
                name_df[(c, t)] += 1
            for t in set(_addr_tokens(q[2])):
                addr_df[(c, t)] += 1
    return name_df, addr_df, n


def _w(df, n, c, t):
    """Inverse document frequency of a token within its country."""
    return math.log((n[c] + 1) / (df[(c, t)] + 1))


def _close(a, b):
    """Spelling variant: one or two edits on long enough words, or a shared stem with a short differing tail."""
    if a == b:
        return False
    m = min(len(a), len(b))
    d = Levenshtein.distance(a, b)
    if (m >= 4 and d <= 1) or (m >= 8 and d <= 2):
        return True
    k = len(os.path.commonprefix([a, b]))
    return k >= 4 and abs(len(a) - len(b)) <= 2 and max(len(a), len(b)) - k <= 2


def _abbrev(a, b):
    """Initial or in-order contraction of the longer word, same first letter."""
    s, l = (a, b) if len(a) <= len(b) else (b, a)
    if s[0] != l[0] or len(s) == len(l):
        return False
    if len(s) == 1:
        return True
    it = iter(l)
    return len(s) >= 2 and all(ch in it for ch in s)


def name_align(sa, tb, wa, wb):
    """Greedy one-to-one alignment of Source 1 tokens sa with target tokens tb (exact, then spelling, then abbreviation)."""
    used_a, match = set(), {}
    kind = {}
    for step, test in (("exact", lambda x, y: x == y), ("typo", _close), ("abbrev", _abbrev)):
        for j, b in enumerate(tb):
            if j in match:
                continue
            for i, a in enumerate(sa):
                if i not in used_a and test(a, b):
                    used_a.add(i); match[j] = i; kind[j] = step
                    break
    many = sum(1 for j, b in enumerate(tb) if j not in match and any(a == b or _close(a, b) for a in sa))
    order = [match[j] for j in sorted(match)]
    reordered = any(x > y for x, y in zip(order, order[1:]))
    ua = [i for i in range(len(sa)) if i not in used_a]; ub = [j for j in range(len(tb)) if j not in match]
    rep = min(len(ua), len(ub))
    tot_a = sum(wa) or 1.0
    return [sum(1 for k in kind.values() if k == "exact"), sum(1 for k in kind.values() if k == "typo"),
            sum(1 for k in kind.values() if k == "abbrev"), float(reordered),
            sum(wa[i] for i in ua), sum(wb[j] for j in ub), max([wa[i] for i in ua], default=0.0),
            max([wb[j] for j in ub], default=0.0), rep, len(ub) - rep, len(ua) - rep, many,
            sum(wb[j] for j, k in kind.items() if k == "typo"), sum(wa[i] for i in used_a) / tot_a]


def _numbers(raw):
    """Probable postcode set, house number with suffix, unit number and compound flag of a raw address."""
    low = strip_accents(raw.lower())
    pcs = postcodes(raw)
    unit = _UNIT.search(low)
    house, suf = None, ""
    for m in _HOUSE.finditer(low):
        if m.group(1) in pcs or (unit and m.start(1) == unit.start(1)):
            continue
        house, suf = int(m.group(1)), m.group(2) or ""
        break
    return pcs, house, suf, unit.group(1) if unit else None, bool(_COMPOUND.search(low))


def _state(x, y):
    """1 agree, -1 conflict, 0 when either side lacks the value."""
    if not x or not y:
        return 0
    return 1 if (x & y if isinstance(x, (set, frozenset)) else x == y) else -1


def addr_diff(s1_addr, t_addr, c, df, n, wf=None, rare=5.0):
    """Number-role and directional token agreement between two raw addresses (token weights from wf when given)."""
    p1, h1, x1, u1, c1 = _numbers(s1_addr); p2, h2, x2, u2, c2 = _numbers(t_addr)
    a, b = [t for t in _addr_tokens(s1_addr) if not t.isdigit()], [t for t in _addr_tokens(t_addr) if not t.isdigit()]
    sa, sb = set(a), set(b)
    w = {t: (wf(t) if wf else _w(df, n, c, t)) for t in sa | sb}
    shared = sum(w[t] for t in sa & sb); wa = sum(w[t] for t in sa); wb = sum(w[t] for t in sb)
    ua, ub = sa - sb, sb - sa
    rare = [t for t in ua if w[t] >= rare]
    fuzzy = sum(1 for t in rare if any(_close(t, u) for u in ub))
    run = best = 0
    pos_b = {}
    for j, t in enumerate(b):
        pos_b.setdefault(t, []).append(j)
    prev = {}
    for t in a:  # longest common contiguous token run
        cur = {}
        for j in pos_b.get(t, ()):
            cur[j] = prev.get(j - 1, 0) + 1
            best = max(best, cur[j])
        prev = cur
    run = best / max(1, min(len(a), len(b)))
    return [_state(p1, p2), _state(h1, h2), _state(u1, u2), float((h1 is None) != (h2 is None)),
            float(h1 is not None and h1 == h2 and x1 != x2), float(c1 or c2),
            math.log1p(abs(h1 - h2)) if h1 is not None and h2 is not None else 0.0,
            shared / wa if wa else 0.0, shared / wb if wb else 0.0, sum(w[t] for t in ua), sum(w[t] for t in ub),
            max([w[t] for t in ub], default=0.0), fuzzy, run]


def diff_features(split, s1_row, texts, cty, tables, bundle):
    """Matrix of NAME_COLS ("name") or ADDR_COLS ("address") for pairs with (t_name, t_addr, s1 text) texts."""
    name_df, addr_df, n = tables
    t_name, t_addr, s1t = texts
    cols = NAME_COLS if bundle == "name" else ADDR_COLS
    X = np.zeros((len(s1_row), len(cols)), dtype=np.float32)
    for i in range(len(s1_row)):
        r = int(s1_row[i]); c = cty[r]; sn, sa = s1t.get(r, ("", ""))
        if bundle == "name":
            a = [t for t in normalize_name(sn).split() if t not in FILLER]
            b = [t for t in normalize_name(t_name[i]).split() if t not in FILLER]
            X[i] = name_align(a, b, [_w(name_df, n, c, t) for t in a], [_w(name_df, n, c, t) for t in b])
        else:
            X[i] = addr_diff(sa, t_addr[i], c, addr_df, n)
    return X


if __name__ == "__main__":
    df = collections.Counter({("US", "meridian"): 3, ("US", "foods"): 500, ("US", "logistics"): 400, ("US", "premium"): 300})
    n = collections.Counter({"US": 10000})
    W = lambda ts: [_w(df, n, "US", t) for t in ts]
    f = lambda a, b: dict(zip(NAME_COLS, name_align(a.split(), b.split(), W(a.split()), W(b.split()))))
    assert f("meridian foods", "meridan foods")["nm_typo"] == 1 and f("meridian foods", "meridan foods")["nm_replaced"] == 0
    assert f("meridian foods", "meridian food")["nm_typo"] == 1
    assert f("meridian foods", "meridian logistics")["nm_replaced"] == 1
    assert f("meridian foods", "meridian premium foods")["nm_added"] == 1 and f("meridian foods", "meridian premium foods")["nm_exact"] == 2
    assert f("foods meridian", "meridian foods")["nm_reordered"] == 1
    assert f("international business machines", "intl business machines")["nm_abbrev"] == 1
    g = dict(zip(ADDR_COLS, addr_diff("12 Rue Victor Hugo, 75000", "18 Rue Victor Hugo, 75000", "US", collections.Counter(), n)))
    assert g["pc_state"] == 1 and g["house_state"] == -1 and g["ad_street_run"] == 1.0
    g = dict(zip(ADDR_COLS, addr_diff("12 bis Rue Hugo 75000", "12 Rue Hugo 75000", "US", collections.Counter(), n)))
    assert g["house_state"] == 1 and g["house_suffix_conflict"] == 1
    g = dict(zip(ADDR_COLS, addr_diff("500 Main St Suite 210, Austin TX 78701", "500 Main St Suite 300, Austin TX 78701", "US", collections.Counter(), n)))
    assert g["unit_state"] == -1 and g["house_state"] == 1 and g["pc_state"] == 1, g
    print("ok")
