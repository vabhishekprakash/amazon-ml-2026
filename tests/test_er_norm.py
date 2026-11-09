"""Checks for entity resolution normalization and address feature extraction.

Run either way:
    python tests/test_er_norm.py
    pytest tests/test_er_norm.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.er_norm import (address_tokens, hash64, house_numbers, normalize_name,
                         number_word_keys, postcodes, strip_accents, token_sort)


def test_legal_suffix_variants_collapse():
    """Corp/Corporation and Pvt Ltd/Private Limited are the stated noise pattern."""
    assert normalize_name("Acme Corporation") == normalize_name("Acme Corp")
    assert normalize_name("Build Cae Private Limited") == normalize_name("Build Cae Pvt Ltd")


def test_injected_punctuation_is_dropped():
    """Observed noise: bracketed suffixes and leading junk characters."""
    assert normalize_name("Jo-Ann Millette Token [LLC]") == "jo ann millette token llc"
    assert normalize_name("-- Holloway Peak Inc Seafood") == "holloway peak inc seafood"
    assert normalize_name("<< Team Ecole") == "team ecole"
    assert normalize_name("B+ Retail Inc") == "b retail inc"


def test_number_word_keys_survive_reordering_and_zeros():
    """Observed missed matches: same building, components reordered, zero-padded."""
    a = number_word_keys("710 Page Cemetery Road, Melbourne, AR")
    assert a == {"710 page", "710 cemetery"}
    assert a & number_word_keys("AR, MELBOURNE, 0710 PAGE CEMETERY RD")
    assert number_word_keys("386/1 Gariahat Road South") == {"386 gariahat", "386 rd",
                                                             "1 gariahat", "1 rd"}
    assert number_word_keys("Kolkata 700019") == frozenset()
    assert number_word_keys("") == frozenset()


def test_indic_words_stay_whole():
    """Vowel signs and viramas are combining marks; they must not split words."""
    raw = normalize_name("शिव फुड्स प्राइवेट लिमिटेड", translit=False)
    assert raw.split() == ["शिव", "फुडस", "पराइवेट", "लिमिटेड"]
    assert len(normalize_name("శివ ఫుడ్స్ ప్రైవేట్ లిమిటెడ్", translit=False).split()) == 4
    assert normalize_name("Café Société & Co.") == "cafe ste co"


def test_ampersand_and_and_agree():
    assert normalize_name("B & G Exchange") == normalize_name("B and G Exchange")


def test_accents_stripped():
    assert strip_accents("Lycée du Dame") == "Lycee du Dame"
    assert normalize_name("Équilibre & Frères SAS") == normalize_name("Equilibre & Freres SAS")


def test_url_suffix_removed():
    assert normalize_name("Shiva International Limited | www.shivainte.com") == \
           normalize_name("Shiva International Limited")


def test_bare_domain_name_keeps_its_body():
    """About 3.5% of Source 2 names are only a domain; erasing them loses the key."""
    assert normalize_name("heassociates.com") == "heassociates"
    assert normalize_name("KASEYHEREDIATAILWIND.COM") == "kaseyherediatailwind"
    assert normalize_name("www.shivainte.com") == "shivainte"


def test_empty_and_symbol_only_names():
    assert normalize_name("") == ""
    assert normalize_name("   ") == ""
    assert normalize_name("!!! ---") == ""


def test_non_latin_names_survive():
    """Devanagari names must not normalize to empty; they are real keys."""
    assert normalize_name("डिजिटल आईटी प्राइवेट लिमिटेड") != ""


def test_token_sort_fixes_word_order():
    """Word-order transposition is a stated noise pattern."""
    a = token_sort(normalize_name("CAMPBELL HEARTLAND INC SERVICES"))
    b = token_sort(normalize_name("Heartland Campbell Services Inc"))
    assert a == b


def test_house_numbers_ignore_leading_zeros():
    assert 8138 in house_numbers("TX, FORT WORTH, 008138 MARYDEAN AVE")
    assert house_numbers("008138 MARYDEAN AVE") & house_numbers("8138 Marydean Avenue")


def test_house_numbers_found_in_any_position():
    """Component reordering means the number is not always first."""
    assert house_numbers("TX, FORT WORTH, 008138 MARYDEAN AVE") == \
           house_numbers("008138 Marydean Ave, Fort Worth, TX")


def test_house_numbers_skip_long_runs():
    assert house_numbers("call 9876543210") == frozenset()


def test_house_numbers_split_compound():
    """Indian forms like 112/D and A-408A still yield their numeric part."""
    assert 112 in house_numbers("112/D Picnic Garden Road, Kolkata")
    assert 408 in house_numbers("House No. A-408A New Ashok Nagar")


def test_postcodes_keep_leading_zeros():
    assert postcodes("Boston MA 02134") == {"02134"}
    assert postcodes("Mumbai 400001") == {"400001"}
    assert postcodes("75008 Paris") == {"75008"}


def test_postcodes_reject_wrong_length():
    assert postcodes("193 Maple Avenue") == frozenset()
    assert postcodes("1234 Main St") == frozenset()


def test_address_tokens_drop_numbers():
    toks = address_tokens("193 Maple Avenue, White Sulphur Springs, WV")
    assert "maple" in toks and "193" not in toks


def test_hash64_is_stable_and_signed():
    """Cached to .npy as int64, so it must be reproducible and in range."""
    assert hash64("acme corp") == hash64("acme corp")
    assert hash64("acme corp") != hash64("acme co")
    assert hash64("") == 0
    assert -2**63 <= hash64("anything") < 2**63


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failures = 0
    for fn in tests:
        try:
            fn()
            print(f"PASS  {fn.__name__}")
        except AssertionError as exc:
            failures += 1
            print(f"FAIL  {fn.__name__}: {exc}")
    print(f"\n{len(tests) - failures}/{len(tests)} passed")
    sys.exit(1 if failures else 0)
