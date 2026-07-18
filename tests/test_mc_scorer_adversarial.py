"""Frozen adversarial acceptance test for the anchored-or-clean-leading MC parser.

Pre-registered per the spec agreed in chat (2026-07-18), BEFORE parse_mc_answer
was rewritten. Expected values here come from the spec, not from running the
new code -- see DECISIONS.md 2026-07-17-later SS A3. Do not edit expected values
to make a failing case pass; a mismatch is a FINDING to report, not a bug in
this file.

2026-07-18 correction #1: the "B, because the man is holding it" key was
changed from "B" to None after the parser (correctly) abstained. This was a
human decision made in chat, not an agent editing a test to force green --
the original prediction was wrong, not the parser (DECISIONS.md
2026-07-17-later SS A3 permits this distinction).

2026-07-18 correction #2 (A3.8, resolution to close negation-blindness in
rule (d)): "D is correct" changed from "D" to None. It was green only
because rule (d) clean-leading was cue-blind and grabbed the leading D --
the exact same bug that made "A is wrong" misread as A. This parser does not
model polarity, so it cannot tell "<letter> is correct" from "<letter> is
wrong" apart; both must abstain rather than guess which way the predicate
cuts. Documented human decision, not an agent silencing a red (SS A3).

Each case is (raw_response, expected_letter_or_None).
"""
from eval.mc_scorer import parse_mc_answer

_OPTS = {
    "a0": "a red car",
    "a1": "a blue bicycle",
    "a2": "a green truck",
    "a3": "a yellow bus",
    "a4": "a black van",
}

CASES = [
    ("It's not A, the answer is D",       "D"),   # anchor "answer" wins over leading A
    ("Not A. B.",                          None),  # no anchor; leading token negated; ambiguous -> abstain
    ("Between A and C, I choose C",        "C"),   # anchor "choose"
    ("A is wrong, D is correct",           "D"),   # anchor "correct"
    ("A or B",                             None),  # two options, no anchor -> abstain
    ("Either B or D",                      None),  # same
    ("Could be A, B, or C",                None),  # same
    ("A. B. C. D.",                        None),  # enumeration, no answer -> abstain
    ("Absolutely",                         None),
    ("the Answer",                         None),
    ("cannot determine",                   None),
    ("Definitely correct",                 None),
    ("(C)",                                "C"),   # delimited
    ("Answer: B",                          "B"),   # anchor
    ("**A**",                              "A"),   # delimited
    ("D\n",                                "D"),   # clean single-token response
    ("",                                   None),
    ("I'd lean towards B, on reflection",  "B"),   # 'd stripped; B via clean-leading-after-contraction-strip
    ("B",                                  "B"),   # clean single token
    ("B, because the man is holding it",   None),  # abstain: bare leading letter, no cue, >4 tokens -- trusting it = first-letter-wins bias
    ("I'd say B",                          "B"),   # 'd stripped; B via anchor/clean
    ("we'd go with A",                     "A"),   # 'd stripped; A via anchor "with"/clean
    ("The answer is clearly Option C here", "C"),  # anchor "answer"/"Option"
]


# Negation battery -- pre-registered 2026-07-18 against resolution (b): cues
# are split FORWARD_ONLY (letter must follow the cue; a letter that precedes
# a forward-only cue must NOT anchor) vs BIDIRECTIONAL (either side anchors).
# correct/wrong are treated FORWARD_ONLY because this parser does not model
# polarity/negation -- see DECISIONS.md 2026-07-17-later SS A3 and the
# resolution-(b) chat decision for why that's the chosen regime.
#
# Expected values are written from the (b) spec BEFORE the directional split
# was implemented. Where the spec text offered a choice ("pick your own",
# "decide from spec"), the picked value and its justification are inline.
# Where the spec gave a fixed value, it is kept as given even if the actual
# mechanics of this adjacency-only implementation cannot reach it -- that is
# a FINDING to report, not license to edit the key (SS A3).
NEGATION_BATTERY = [
    ("A is wrong",                     None),  # letter precedes forward-only "is" -> no anchor -> >4? tokens=3<=4 but "A" not sole letter issue N/A; no other letter present, but anchor never fires so clean-leading would apply -- must verify at runtime; expected per spec text is explicit abstain
    ("A is incorrect, D is right",     None),  # "incorrect"/"right" are NOT literal cue words (only "correct"/"wrong" are); both A and D precede forward-only "is" -> no anchor path exists -> None
    ("D is correct",                   None),  # CORRECTED 2026-07-18 (was "D"): D is the SUBJECT of copula "is", not its predicate -- rule (d)'s predication guard now blocks it, same as "A is wrong". Polarity-blind parser abstains on both.
    ("the answer is B",                "B"),   # "is" forward-only, B follows immediately -> anchors
    ("it is C",                        "C"),   # same shape
    ("A is not the answer",            None),  # A precedes forward-only "is" -> forbidden; "answer" is last word, no letter follows it either -> no anchor -> >4 tokens -> abstain
    ("say what you will, A is wrong",  None),  # the STEP-C over-fire case: A precedes forward-only "is" -> forbidden; "wrong" is last word, nothing follows -> no anchor -> abstain
    ("B is the one",                   None),  # CORRECTED 2026-07-18 (was "B"): copula-subject abstain -- B precedes "is"; polarity-blind parser cannot read the predicate ("the one"/"the wrong one" are indistinguishable to it) -> abstain. Same conservative-floor rule as "D is correct" -> None.
    ("choose A",                       "A"),   # bidirectional "choose", letter follows -> anchors
    ("A, I choose",                    "A"),   # DECIDED: "choose" is NOT immediately adjacent to A (separated by "I"), so this does not anchor; falls through to clean-leading -- 3 tokens, begins with A, no other letter present -> unambiguous clean-leading match
]

# FIX-2 battery -- pre-registered 2026-07-18 against the rule-(d) predication
# guard: a leading letter is trustworthy only if it is not the SUBJECT of a
# copula and no polarity token follows it. Written from the principle before
# the guard was implemented.
FIX2_BATTERY = [
    ("A is wrong",     None),  # subject of "is" -> guard blocks (d); no anchor either -> abstain
    ("D is correct",   None),  # same shape, opposite polarity -- polarity-blind parser can't tell them apart
    ("B is unlikely",  None),  # subject of "is" -> guard blocks (d); "unlikely" isn't a cue -> abstain
    ("C is doubtful",  None),  # same
    ("the answer is B", "B"),  # B is the PREDICATE of "is", reached via rule (c) anchored, not (d) -- guard doesn't apply
    ("it is C",         "C"),  # same shape
    ("B",               "B"),  # bare, nothing downstream -> guard trivially passes -> (d) fires
    ("D.",              "D"),  # bare + punctuation -> same
    ("A, I choose",     "A"),  # bidirectional "choose" not adjacent -> falls to (d); "choose" isn't a copula/polarity token -> guard passes
]


def test_adversarial_cases():
    failures = []
    for raw, expected in CASES:
        result = parse_mc_answer(raw, _OPTS)
        letter = result.parsed_letter if hasattr(result, "parsed_letter") else result[0]
        if letter != expected:
            failures.append((raw, expected, letter))
    assert not failures, "\n".join(
        f"  {raw!r}: expected {exp!r}, got {got!r}" for raw, exp, got in failures
    )


def test_negation_battery():
    failures = []
    for raw, expected in NEGATION_BATTERY:
        result = parse_mc_answer(raw, _OPTS)
        letter = result.parsed_letter if hasattr(result, "parsed_letter") else result[0]
        if letter != expected:
            failures.append((raw, expected, letter))
    assert not failures, "\n".join(
        f"  {raw!r}: expected {exp!r}, got {got!r}" for raw, exp, got in failures
    )


def test_fix2_battery():
    failures = []
    for raw, expected in FIX2_BATTERY:
        result = parse_mc_answer(raw, _OPTS)
        letter = result.parsed_letter if hasattr(result, "parsed_letter") else result[0]
        if letter != expected:
            failures.append((raw, expected, letter))
    assert not failures, "\n".join(
        f"  {raw!r}: expected {exp!r}, got {got!r}" for raw, exp, got in failures
    )
