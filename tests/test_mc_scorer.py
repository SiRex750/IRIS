from eval.mc_scorer import parse_mc_answer

_OPTS = {
    "a0": "a red car",
    "a1": "a blue bicycle",
    "a2": "a green truck",
    "a3": "a yellow bus",
    "a4": "a black van",
}


def test_bare_letter():
    assert parse_mc_answer("A", _OPTS) == ("A", None)


def test_trailing_period():
    assert parse_mc_answer("A.", _OPTS) == ("A", None)


def test_trailing_paren():
    assert parse_mc_answer("A)", _OPTS) == ("A", None)


def test_lowercase_letter():
    assert parse_mc_answer("a", _OPTS) == ("A", None)


def test_letter_embedded_in_sentence():
    assert parse_mc_answer("The correct answer is D.", _OPTS) == ("D", None)


def test_unparseable_response():
    choice, reason = parse_mc_answer("I am not sure about this", _OPTS)
    assert choice is None
    assert reason == "no match in: 'I am not sure about this'"
