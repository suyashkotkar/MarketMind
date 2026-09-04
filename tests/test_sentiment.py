from stockseer.sentiment.scorer import label_for, score_text, score_texts


def test_polarity_direction():
    assert score_text("Revenue beats estimates; company raises guidance").score > 0.2
    assert score_text("Shares plunge after the company slashes guidance").score < -0.2
    assert abs(score_text("The company will hold its annual meeting").score) < 0.1


def test_negation_flips_polarity():
    plain = score_text("The company beat estimates").score
    negated = score_text("The company did not beat estimates").score
    assert plain > 0 and negated < plain


def test_intensifier_amplifies():
    mild = score_text("Shares fell").score
    strong = score_text("Shares fell sharply").score
    assert strong <= mild


def test_scores_are_bounded_and_labelled():
    texts = ["fraud bankruptcy lawsuit plunge slump loss" * 5,
             "record profit surge rally upgrade breakthrough" * 5]
    for r in score_texts(texts):
        assert -1.0 <= r.score <= 1.0
        assert r.label in ("positive", "negative", "neutral")
    assert label_for(0.4) == "positive"
    assert label_for(-0.4) == "negative"
    assert label_for(0.0) == "neutral"


def test_empty_input_is_neutral():
    r = score_text("")
    assert r.score == 0.0 and r.hits == 0
