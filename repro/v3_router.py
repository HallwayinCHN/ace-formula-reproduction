"""Small, predeclared, question-only routing families. No target or model confidence at inference."""

import re

NUMBER = re.compile(r"(?<!\w)[+-]?\d[\d,]*(?:\.\d+)?")
HARD = re.compile(r"black.scholes|\boptions?\b|normal distribution|\bannuit|amorti|present value|future value|"
                  r"discount|\bnpv\b|\birr\b|internal rate|duration|convexity|compound|\bexp\b|\blog\b|\bsqrt\b", re.I)
AMBIGUOUS = re.compile(r"what happens|how does|without|no change|\bimpact\b|\beffect\b|insufficient|unknown", re.I)
RATIO = re.compile(r"%|percent|\brate\b|\bratio\b|\bmargin\b|\byield\b|\breturn\b|\bROI\b|\bROE\b", re.I)
SCALE = re.compile(r"\bthousand\b|\bmillion\b|\bbillion\b|\btrillion\b", re.I)
SUPPORTS = (3, 5, 8)


def profile(question):
    # Exclude the upstream universal formatting suffix, whose "5 million" example occurs in every input.
    text = question.split(" Your answer should be a plain floating point number,", 1)[0]
    count = len(NUMBER.findall(text))
    words = len(text.split())
    if HARD.search(text) or count > 5 or words > 75 or "^" in text:
        bucket = "protected_complex"
    elif AMBIGUOUS.search(text) or count < 2:
        bucket = "protected_ambiguous"
    elif RATIO.search(text):
        bucket = "ratio_or_percent"
    elif SCALE.search(text):
        bucket = "scaled_amount"
    else:
        bucket = "plain_numeric"
    return {"bucket": bucket, "numbers": count, "words": words,
            "protected": bucket.startswith("protected_")}


def fit(pairs, minimum_questions):
    """Each pair is one question with all three books. Any observed harmed answer vetoes its bucket."""
    buckets = {}
    for pair in pairs:
        key = pair["profile"]["bucket"]
        buckets.setdefault(key, []).append(pair)
    allowed = []
    evidence = {}
    for key, values in buckets.items():
        harmed = sum(sum(a and not b for a, b in zip(pair["original_correct"], pair["brief_correct"]))
                     for pair in values)
        evidence[key] = {"questions": len(values), "harmed_answers": harmed}
        if not key.startswith("protected_") and len(values) >= minimum_questions and harmed == 0:
            allowed.append(key)
    return {"allowed_buckets": sorted(allowed), "minimum_questions": minimum_questions,
            "evidence": evidence, "default": "original"}


def choose(question, rule):
    details = profile(question)
    brief = not details["protected"] and details["bucket"] in rule["allowed_buckets"]
    return brief, details
