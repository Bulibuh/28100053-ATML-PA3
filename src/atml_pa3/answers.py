"""Answer equivalence is fixed before selection; gold is used only for grading."""

import re
from functools import lru_cache
import numpy as np

NO_ANSWER = "NO_ANSWER"
DIGITS = dict(
    zip(
        "zero one two three four five six seven eight nine ten".split(),
        map(str, range(11)),
    )
)
DIGITS["none"] = "0"
# Contractions follow the official VQA evaluator convention (see ATTRIBUTIONS.md).
CONTRACTIONS = {
    "aint": "ain't",
    "arent": "aren't",
    "cant": "can't",
    "couldve": "could've",
    "couldnt": "couldn't",
    "couldn'tve": "couldn't've",
    "couldnt've": "couldn't've",
    "didnt": "didn't",
    "doesnt": "doesn't",
    "dont": "don't",
    "hadnt": "hadn't",
    "hadnt've": "hadn't've",
    "hadn'tve": "hadn't've",
    "hasnt": "hasn't",
    "havent": "haven't",
    "hed": "he'd",
    "hed've": "he'd've",
    "he'dve": "he'd've",
    "hes": "he's",
    "howd": "how'd",
    "howll": "how'll",
    "hows": "how's",
    "Id've": "I'd've",
    "I'dve": "I'd've",
    "Im": "I'm",
    "Ive": "I've",
    "isnt": "isn't",
    "itd": "it'd",
    "itd've": "it'd've",
    "it'dve": "it'd've",
    "itll": "it'll",
    "let's": "let's",
    "maam": "ma'am",
    "mightnt": "mightn't",
    "mightnt've": "mightn't've",
    "mightn'tve": "mightn't've",
    "mightve": "might've",
    "mustnt": "mustn't",
    "mustve": "must've",
    "neednt": "needn't",
    "notve": "not've",
    "oclock": "o'clock",
    "oughtnt": "oughtn't",
    "ow's'at": "'ow's'at",
    "'ows'at": "'ow's'at",
    "'ow'sat": "'ow's'at",
    "shant": "shan't",
    "shed've": "she'd've",
    "she'dve": "she'd've",
    "she's": "she's",
    "shouldve": "should've",
    "shouldnt": "shouldn't",
    "shouldnt've": "shouldn't've",
    "shouldn'tve": "shouldn't've",
    "somebody'd": "somebodyd",
    "somebodyd've": "somebody'd've",
    "somebody'dve": "somebody'd've",
    "somebodyll": "somebody'll",
    "somebodys": "somebody's",
    "someoned": "someone'd",
    "someoned've": "someone'd've",
    "someone'dve": "someone'd've",
    "someonell": "someone'll",
    "someones": "someone's",
    "somethingd": "something'd",
    "somethingd've": "something'd've",
    "something'dve": "something'd've",
    "somethingll": "something'll",
    "thats": "that's",
    "thered": "there'd",
    "thered've": "there'd've",
    "there'dve": "there'd've",
    "therere": "there're",
    "theres": "there's",
    "theyd": "they'd",
    "theyd've": "they'd've",
    "they'dve": "they'd've",
    "theyll": "they'll",
    "theyre": "they're",
    "theyve": "they've",
    "twas": "'twas",
    "wasnt": "wasn't",
    "wed've": "we'd've",
    "we'dve": "we'd've",
    "weve": "we've",
    "werent": "weren't",
    "whatll": "what'll",
    "whatre": "what're",
    "whats": "what's",
    "whatve": "what've",
    "whens": "when's",
    "whered": "where'd",
    "wheres": "where's",
    "whereve": "where've",
    "whod": "who'd",
    "whod've": "who'd've",
    "who'dve": "who'd've",
    "wholl": "who'll",
    "whos": "who's",
    "whove": "who've",
    "whyll": "why'll",
    "whyre": "why're",
    "whys": "why's",
    "wont": "won't",
    "wouldve": "would've",
    "wouldnt": "wouldn't",
    "wouldnt've": "wouldn't've",
    "wouldn'tve": "wouldn't've",
    "yall": "y'all",
    "yall'll": "y'all'll",
    "y'allll": "y'all'll",
    "yall'd've": "y'all'd've",
    "y'alld've": "y'all'd've",
    "y'all'dve": "y'all'd've",
    "youd": "you'd",
    "youd've": "you'd've",
    "you'dve": "you'd've",
    "youll": "you'll",
    "youre": "you're",
    "youve": "you've",
}


def vqa_normalize(answer):
    text = str(answer).replace("\n", " ").replace("\t", " ").strip().lower()
    # Official punctuation handling retains decimal points and removes digit-group commas.
    comma = bool(re.search(r"\d,\d", text))
    original = text
    for p in [
        ";",
        "/",
        "[",
        "]",
        '"',
        "{",
        "}",
        "(",
        ")",
        "=",
        "+",
        "\\",
        "_",
        "-",
        ">",
        "<",
        "@",
        "`",
        ",",
        "?",
        "!",
    ]:
        text = text.replace(
            p, "" if p + " " in original or " " + p in original or comma else " "
        )
    text = re.sub(r"(?!<=\d)(\.)(?!\d)", "", text)
    return " ".join(
        CONTRACTIONS.get(DIGITS.get(w, w), DIGITS.get(w, w))
        for w in text.split()
        if w not in ("a", "an", "the")
    )


def vqa_score(prediction, answers):
    if len(answers) != 10:
        raise ValueError("VQAv2 requires ten human answers")
    p = vqa_normalize(prediction)
    if not p:
        return 0.0
    refs = [vqa_normalize(x) for x in answers]
    return float(
        np.mean(
            [
                min(1.0, sum(p == refs[j] for j in range(10) if j != i) / 3.0)
                for i in range(10)
            ]
        )
    )


@lru_cache(maxsize=100000)
def math_parse(text):
    from math_verify import parse, LatexExtractionConfig, ExprExtractionConfig

    # Prefer already-extracted answer; parse both latex and plain expressions.
    return parse(
        str(text), extraction_config=[LatexExtractionConfig(), ExprExtractionConfig()]
    )


def math_equal(a, b):
    if a is None or b is None:
        return False
    a = str(a).strip()
    b = str(b).strip()
    if not a or not b or a == NO_ANSWER or b == NO_ANSWER:
        return False
    if a == b:
        return True
    from math_verify import verify

    pa, pb = math_parse(a), math_parse(b)
    return bool(pa and pb and verify(pa, pb))


def mathvista_normalize(answer, meta):
    if answer is None or not str(answer).strip():
        return NO_ANSWER
    value = str(answer).strip()
    if meta.get("question_type") == "multi_choice":
        choices = meta.get("choices") or []
        match = re.fullmatch(r"\(?([A-Za-z])\)?[.)]?", value)
        if match and 0 <= ord(match[1].upper()) - 65 < len(choices):
            return match[1].upper()
        for i, c in enumerate(choices):
            if value.casefold() == str(c).strip().casefold():
                return chr(65 + i)
        return NO_ANSWER
    try:
        if meta["answer_type"] == "integer":
            return str(int(float(value.replace(",", ""))))
        if meta["answer_type"] == "float":
            return str(round(float(value.replace(",", "")), int(meta["precision"])))
    except (ValueError, TypeError, OverflowError):
        return NO_ANSWER
    return value


def boxed(text):
    # Last balanced \\boxed{...}, supporting nested braces; unbalanced final box is missing.
    pos = text.rfind("\\boxed{")
    if pos < 0:
        return None
    start = pos + 7
    depth = 1
    for i in range(start, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                return text[start:i].strip() or None
    return None


def answer_groups(answers, dataset, meta=None):
    groups = []
    representatives = []
    for answer in answers:
        a = (
            NO_ANSWER
            if answer is None or not str(answer).strip()
            else str(answer).strip()
        )
        if dataset == "mathvista":
            a = mathvista_normalize(None if a == NO_ANSWER else a, meta)
        found = None
        for i, rep in enumerate(representatives):
            if a == rep or (dataset == "math" and math_equal(a, rep)):
                found = i
                break
        if found is None:
            found = len(representatives)
            representatives.append(a)
        groups.append(found)
    return np.array(groups), representatives
