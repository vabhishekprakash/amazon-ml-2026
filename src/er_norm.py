"""Text normalization and address feature extraction for entity resolution.

Pure functions apart from one lazily loaded lookup table (the word-level
transliteration table mined by src/translit.py). Everything here must behave
the same for every country label: the rules key on shapes that occur in all of
them (digit runs, latin punctuation, common legal and street abbreviations),
never on the country field.
"""
import hashlib
import itertools
import json
import re
import unicodedata

import config as C

TRANSLIT_PATH = C.OUT_DIR / "cache" / "translit.json"
_TRANSLIT = None


def _translit_table() -> dict:
    """The non-Latin -> Latin word table, or {} when it has not been built."""
    global _TRANSLIT
    if _TRANSLIT is None:
        _TRANSLIT = (json.loads(TRANSLIT_PATH.read_text(encoding="utf-8"))
                     if TRANSLIT_PATH.exists() else {})
    return _TRANSLIT

# Variant -> canonical token. Legal suffixes first, then street words. Kept
# generic so it transfers to a country that never appears in training.
ABBREVIATIONS = {
    # legal form
    "corporation": "corp", "incorporated": "inc", "limited": "ltd",
    "private": "pvt", "company": "co", "societe": "ste", "société": "ste",
    "anonyme": "sa",
    # "&" is stripped as punctuation, so "and" (and French "et") is dropped to match it.
    "and": "", "et": "",
    # street and address words
    "road": "rd", "street": "st", "avenue": "ave", "boulevard": "blvd",
    "lane": "ln", "drive": "dr", "floor": "fl", "apartment": "apt",
    "building": "bldg", "opposite": "opp", "near": "nr", "saint": "st",
    "sainte": "st", "number": "no", "house": "h", "block": "blk",
    "sector": "sec", "phase": "ph", "nagar": "ngr", "colony": "col",
    "place": "pl", "court": "ct", "circle": "cir", "highway": "hwy", "parkway": "pkwy",
    "square": "sq", "route": "rte",
    # French street types and their usual abbreviations ("34 B Rue Lachassaigne"
    # vs "34 B R LACHASSAIGNE"); accents are already stripped at this point.
    "rue": "r", "bd": "blvd", "bld": "blvd", "av": "ave", "impasse": "imp",
    "allee": "all", "chemin": "ch", "chem": "ch", "faubourg": "fg", "fbg": "fg",
    "numero": "no",
}

# Characters that carry no identity: punctuation injected as noise
# ("[LLC]", "-- Name", "<< Name", "B+ Retail"), quotes, brackets, pipes.
_URL = re.compile(r"\b(?:https?://|www\.)\S+|\b\S+\.(?:com|in|net|org|co)\b", re.I)
_SCHEME = re.compile(r"\b(?:https?://|www\.)", re.I)
_TLD = re.compile(r"\.(?:com|in|net|org|co)\b", re.I)
# Indic scripts (U+0900-U+0DFF: Devanagari through Malayalam) write vowel signs
# and viramas as combining marks, which \w does not match; without the range
# every Hindi, Bengali, Tamil, Telugu or Kannada word would be shattered into
# single consonants ("लिमिटेड" -> "ल म ट ड").
_PUNCT = re.compile(r"[^\w\s\u0900-\u0DFF]+", re.UNICODE)
_SPACE = re.compile(r"\s+")
_DIGIT_RUN = re.compile(r"\d+")


def strip_accents(text: str) -> str:
    """Remove combining marks so accented and plain spellings collapse together."""
    return "".join(ch for ch in unicodedata.normalize("NFKD", text)
                   if not unicodedata.combining(ch))


def normalize_name(text: str, translit: bool = True) -> str:
    """Canonical form of a business name.

    Lowercases, drops a trailing URL, removes punctuation, strips accents, and
    maps known abbreviation variants onto one spelling.

    A name that is nothing but a domain ("heassociates.com" — about 3.5% of
    Source 2) would be erased by the URL rule, so when URL removal leaves
    nothing the domain body is kept instead. Returns "" only when the name
    carries no word characters at all.

    With translit, words in Indic scripts are replaced by their Latin
    equivalent from the mined table, so "शिव फुडस" and "Shiva Foods" agree.
    """
    if not text:
        return ""
    lowered = text.lower()
    out = _URL.sub(" ", lowered)
    if not out.strip():  # the whole name was a URL: keep the domain body
        out = _TLD.sub(" ", _SCHEME.sub(" ", lowered))
    out = _PUNCT.sub(" ", strip_accents(out))
    tokens = [ABBREVIATIONS.get(tok, tok) for tok in _SPACE.sub(" ", out).split()]
    if translit:
        table = _translit_table()
        if table:
            tokens = [table.get(tok, tok) for tok in tokens]
    return " ".join(tok for tok in tokens if tok)


# Words that say what kind of legal entity a business is, not which one it is.
# Dropping them lets "Corner Ice Cream" meet "CORNER ICE CREAM CORPORATION" and
# "Nextgen Agrotech Limited" meet "M/s Nextgen Agrotech Limited".
FILLER = frozenset("""
    ltd pvt llc llp lp inc corp co pc plc sa sas sarl ste gmbh eurl sci snc cie
    groupe distribution fils freres
    group services service enterprises holdings partners associates ventures
    solutions global trading brothers industries center company the of
    m s shri sri mr dr
""".split())


def core_name(normalized: str) -> str:
    """Sorted name tokens without legal forms, honorifics and corporate filler."""
    return " ".join(sorted(t for t in normalized.split() if t not in FILLER))


def token_sort(normalized: str) -> str:
    """Reorder tokens alphabetically so word-order swaps collapse to one key."""
    return " ".join(sorted(normalized.split()))


def house_numbers(address: str) -> frozenset:
    """Integer values of every digit run of 1-6 digits in the address.

    Leading zeros are dropped so "008138 MARYDEAN AVE" and "8138 Marydean Ave"
    agree. Runs longer than 6 digits are phone-like and are ignored.
    """
    return frozenset(int(run) for run in _DIGIT_RUN.findall(address or "")
                     if 1 <= len(run) <= 6)


def postcodes(address: str) -> frozenset:
    """Digit runs of exactly 5 or 6 characters, kept as strings.

    Covers US ZIP (5), India PIN (6) and France code postal (5). Leading zeros
    are significant here, so these stay strings.
    """
    return frozenset(run for run in _DIGIT_RUN.findall(address or "")
                     if len(run) in (5, 6))


def address_tokens(address: str) -> frozenset:
    """Normalized non-numeric word tokens of an address, for overlap features."""
    norm = normalize_name(address)
    return frozenset(tok for tok in norm.split() if not tok.isdigit())


def number_word_keys(address: str, span: int = 2, min_len: int = 1) -> frozenset:
    """'number word' keys pairing each address number with the words after it.

    "319 High Street, Roxboro" gives {"319 high", "319 st"}. Each word of the
    address is common on its own ("high", "st", a town name), but a number
    next to its street word picks out one building, whatever order the other
    components appear in and whatever script the business name is written in.
    Numbers lose leading zeros ("0710" -> "710"); runs over 6 digits are
    phone-like and skipped.
    """
    return number_word_keys_from_tokens(normalize_name(address).split(), span, min_len)


def number_word_keys_from_tokens(toks, span: int = 2, min_len: int = 1) -> frozenset:
    """number_word_keys on an already normalized, split address.

    Words shorter than min_len are skipped when pairing, so street-type and
    function words ("rue", "de", "st") give way to the street name itself.
    """
    out = set()
    for i, tok in enumerate(toks):
        if not (tok.isdigit() and len(tok) <= 6):
            continue
        num, taken = str(int(tok)), 0
        for nxt in toks[i + 1:]:
            if nxt.isdigit() or len(nxt) < min_len:
                continue
            out.add(f"{num} {nxt}")
            taken += 1
            if taken == span:
                break
    return frozenset(out)


def name_pair_keys(core: str, max_tokens: int = 8) -> frozenset:
    """Unordered pairs of the distinctive words of a name, from its core form.

    "Womens Health Specialists LLC" and "Womens Health Specialists LLC Society"
    share no whole-name key, and every word they have in common is far too
    frequent to be a key on its own, but both contain the pair "specialists
    womens". A word added, dropped, reordered or misspelled leaves the pairs
    among the untouched words intact, and a pair of words is rare even when
    each word alone is common. Pairing is limited to the first max_tokens words
    in sorted order, which bounds the work on the few very long names.
    """
    toks = sorted(set(core.split()))[:max_tokens]
    return frozenset(f"{a} {b}" for a, b in itertools.combinations(toks, 2))


def concat_keys(normalized: str, max_tokens: int = 8) -> frozenset:
    """Every ordered pair of name words joined without a space, plus the whole name joined.

    A name rewritten as a domain ("Suprita Industries Pvt Ltd" ->
    "supritaindustries.com", "Stalwart & Brothers" -> "brothersstalwart.com")
    normalizes to one word that shares no token with the original; joining
    the original's words reproduces it.
    """
    toks = normalized.split()[:max_tokens]
    keys = {a + b for a, b in itertools.permutations(toks, 2)}
    if len(toks) > 2:
        keys.add("".join(toks))
    return frozenset(keys)


def word_pair_keys(address: str, min_len: int = 4) -> frozenset:
    """Unordered pairs of adjacent address words of min_len+ characters.

    "414 Sunnyside Drive, Tyler, TX" gives {"sunnyside tyler"}. Numbers are
    skipped, so a truncated house number ("14 Sunnyside Drive") still agrees,
    and sorting each pair survives reordered components.
    """
    words = [t for t in normalize_name(address).split() if not t.isdigit() and len(t) >= min_len]
    return frozenset(" ".join(sorted(p)) for p in zip(words, words[1:]) if p[0] != p[1])


def hash64(text: str) -> int:
    """Stable 64-bit hash of a string, identical across processes and runs.

    Python's built-in hash() is randomized per process, so it cannot be cached
    to disk; blake2b gives a reproducible digest instead. Returns 0 for the
    empty string so callers can treat "no usable key" as a single sentinel.
    """
    if not text:
        return 0
    return int.from_bytes(hashlib.blake2b(text.encode("utf-8"), digest_size=8).digest(),
                          "little", signed=True)


# Digits that stand in for letters in corrupted copies ("N0LAN", "keshavbr0s").
# Only 2, 6 and 9 have no common letter shape, so they are left alone.
_DIGIT_LETTER = str.maketrans("0134578", "oleastb")
# A number token carrying a short letter prefix or suffix: "1209c", "apt4", "no128".
_NUM_AFFIX = re.compile(r"^[a-z]{0,3}(\d{1,6})[a-z]{0,3}$")


def repair_digits(token: str) -> str:
    """Look-alike digits turned back into letters, for tokens holding both.

    "n0lan" -> "nolan". A token of digits alone is a house number or a year and
    is left as it is; a token of letters alone has nothing to repair. Applied
    to both sides of a comparison, so a token the rule mangles ("3m" -> "em")
    still agrees with its counterpart.
    """
    if not token or token.isdigit() or token.isalpha():
        return token
    if not (any(c.isdigit() for c in token) and any(c.isalpha() for c in token)):
        return token
    return token.translate(_DIGIT_LETTER)


def repair_name(normalized: str) -> str:
    """repair_digits over every token of an already normalized name."""
    return " ".join(repair_digits(tok) for tok in normalized.split())


def domain_prefix_key(normalized: str, n: int = 8) -> str:
    """First n characters of the name with its spaces removed.

    A name rewritten as a domain loses its word boundaries and usually its tail
    ("Muire Jackson Interstate Ridge" -> "muirejacksoninterstate.com"), so no
    token or pair of tokens survives, but the opening characters do. Returns ""
    for a name shorter than n characters joined, where a prefix identifies
    nothing.
    """
    joined = "".join(normalized.split())
    return joined[:n] if len(joined) >= n else ""


def token_number(token: str) -> str:
    """The digits of a number token, with any short letter prefix or suffix dropped.

    "1209c" -> "1209", "apt4" -> "4". Returns "" when the token is not a number
    in that sense, or when its digit run is over 6 characters and so phone-like.
    """
    if token.isdigit():
        return str(int(token)) if len(token) <= 6 else ""
    m = _NUM_AFFIX.match(token)
    return str(int(m.group(1))) if m else ""


def number_word_keys_loose(address: str, span: int = 2, min_len: int = 1) -> frozenset:
    """number_word_keys with letter-affixed numbers reduced to their digits.

    "1209C STONEY RUN TRL" gives {"1209 stoney", "1209 run"}, so a unit letter
    stuck to the house number no longer hides the building. The words are read
    the same way as in number_word_keys_from_tokens.
    """
    toks = normalize_name(address).split()
    out = set()
    for i, tok in enumerate(toks):
        num = token_number(tok)
        if not num:
            continue
        taken = 0
        for nxt in toks[i + 1:]:
            if token_number(nxt) or len(nxt) < min_len:
                continue
            out.add(f"{num} {nxt}")
            taken += 1
            if taken == span:
                break
    return frozenset(out)


def cross_keys(name: str, address: str, max_name: int = 4, max_addr: int = 5) -> frozenset:
    """Pairs of one distinctive name word with one address word.

    When a copy is corrupted in the name and in the address at once, every key
    drawn from one field alone either breaks or is too common to keep. A rare
    name word next to a shared street or city word survives, because the two
    corruptions rarely land on the same pair. Name words come from the core
    form, so legal suffixes never pair; address words skip pure numbers, which
    the number-word keys already cover. Both sides are capped in sorted order,
    which bounds the work on long records.
    """
    nt = sorted({repair_digits(t) for t in core_name(normalize_name(name)).split()
                 if len(t) >= 3})[:max_name]
    if not nt:
        return frozenset()
    at = sorted({t for t in normalize_name(address).split()
                 if len(t) >= 4 and not t.isdigit()})[:max_addr]
    return frozenset(f"{n} {a}" for n in nt for a in at)
