import re
import lpr_config as config
from lpr import state


def _fix_char_at(ch: str, want_digit: bool) -> str:
    if want_digit:
        return config._LETTER_TO_DIGIT.get(ch, ch) if not ch.isdigit() else ch
    else:
        c = config._DIGIT_TO_LETTER.get(ch, ch) if ch.isdigit() else ch
        return c if c in config._VALID_SERIES else ch


def _normalize_plate_for_output(raw: str) -> str:
    if not raw:
        return ""
    return re.sub(r'[^A-Z0-9]', '', raw.upper())


def _plate_pattern_score(text: str) -> float:
    text = _normalize_plate_for_output(text)
    if len(text) < 7 or len(text) > 9:
        return 0.0
    if len(text) < 3 or not text[:2].isdigit():
        return 0.0

    best = 0.0
    for series_len in (1, 2):
        suffix_start = 2 + series_len
        if suffix_start >= len(text):
            continue
        series = text[2:suffix_start]
        suffix = text[suffix_start:]
        if len(suffix) not in (4, 5):
            continue
        if not series.isalpha() or any(ch not in config._VALID_SERIES for ch in series):
            continue
        if not suffix.isdigit():
            continue

        score = 5.0
        score += 0.5 if series_len == 1 else 0.35
        score += 0.6 if len(suffix) == 5 else 0.25
        best = max(best, score)
    return best


def _plate_quality_score(text: str, conf: float, width: int, height: int, association_score: float) -> float:
    text = _correct_vn_plate(text)
    score = _plate_pattern_score(text)
    if score <= 0.0:
        return 0.0

    score += conf * 2.0
    size_factor = min(1.0, max(0, width) * max(0, height) / 10000.0)
    score += size_factor
    score += max(0.0, association_score)

    return max(0.0, score)


def _square_join_variants(top: str, bot: str) -> list:
    top = _normalize_plate_for_output(top)
    bot = _normalize_plate_for_output(bot)
    if not top and not bot:
        return []

    variants = []

    def _push(value: str):
        value = _correct_vn_plate(value)
        if value and value not in variants:
            variants.append(value)

    _push(top + bot)

    if len(top) >= 4 and len(bot) >= 4 and bot[0].isdigit():
        province_series = top[:3]
        if (
            len(province_series) == 3
            and province_series[:2].isdigit()
            and province_series[2].isalpha()
            and province_series[2] in config._VALID_SERIES
        ):
            _push(province_series + bot)

    return variants


def _correct_vn_plate(raw: str) -> str:
    text = _normalize_plate_for_output(raw)
    if not text:
        return ""

    variants = [text]
    for series_len in (1, 2):
        suffix_start = 2 + series_len
        if suffix_start >= len(text):
            continue
        suffix_len = len(text) - suffix_start
        if suffix_len not in (4, 5):
            continue

        chars = list(text)
        for idx in range(min(2, len(chars))):
            chars[idx] = _fix_char_at(chars[idx], want_digit=True)
        for idx in range(2, min(suffix_start, len(chars))):
            chars[idx] = _fix_char_at(chars[idx], want_digit=False)
        for idx in range(suffix_start, len(chars)):
            chars[idx] = _fix_char_at(chars[idx], want_digit=True)
        variants.append("".join(chars))

    return max(variants, key=lambda v: (_plate_pattern_score(v), v == text))


def _text_similarity(a: str, b: str) -> float:
    a = _normalize_plate_for_output(a)
    b = _normalize_plate_for_output(b)
    if not a or not b:
        return 0.0
    max_len = max(len(a), len(b))
    same = sum(1 for x, y in zip(a, b) if x == y)
    return same / float(max_len)


def _levenshtein(a: str, b: str) -> int:
    if a == b:
        return 0
    if len(a) > len(b):
        a, b = b, a
    prev = list(range(len(a) + 1))
    for cb in b:
        curr = [prev[0] + 1]
        for i, ca in enumerate(a):
            curr.append(min(prev[i + 1] + 1, curr[i] + 1, prev[i] + (0 if ca == cb else 1)))
        prev = curr
    return prev[-1]


def _plate_similar_enough(a: str, b: str) -> bool:
    if a == b:
        return True
    for series_len in (1, 2):
        prefix = 2 + series_len
        if len(a) >= prefix + 3 and len(b) >= prefix + 3 and a[:prefix] == b[:prefix]:
            sa, sb = a[prefix:], b[prefix:]
            if abs(len(sa) - len(sb)) <= 1 and _levenshtein(sa, sb) <= 1:
                return True
    return False


def _major_plate_change(a: str, b: str) -> bool:
    a = _normalize_plate_for_output(a)
    b = _normalize_plate_for_output(b)
    if not a or not b:
        return False
    province_changed = len(a) >= 2 and len(b) >= 2 and a[:2] != b[:2]
    return province_changed or _text_similarity(a, b) < 0.55


def _should_replace_stable_text(current_text: str, current_score: float, current_votes: int,
                                new_text: str, new_score: float, new_votes: int) -> bool:
    if not current_text:
        return True
    if new_text == current_text:
        return new_score > current_score or new_votes > current_votes

    current_pattern = _plate_pattern_score(current_text)
    new_pattern = _plate_pattern_score(new_text)
    major_change = _major_plate_change(current_text, new_text)

    if major_change and new_pattern < current_pattern:
        return False

    vote_margin = config._PLATE_REPLACE_VOTE_MARGIN + (2 if major_change else 0)
    score_margin = config._STABLE_REPLACE_SCORE_MARGIN + (1.0 if major_change else 0.0)

    enough_votes = new_votes >= current_votes + vote_margin
    clearly_better_shape = new_pattern > current_pattern and new_score >= current_score + 0.5
    clearly_better_score = new_score >= current_score + score_margin and new_votes >= max(state.min_stable_votes, current_votes)
    return enough_votes or clearly_better_shape or clearly_better_score


def _is_valid_vn_plate_early(text: str) -> bool:
    return _plate_pattern_score(_correct_vn_plate(text)) > 0.0


def _stable_plate(track_key, raw_text: str, conf: float, width: int, height: int, assoc_score: float) -> str:
    if track_key not in state.plate_history:
        state.plate_history[track_key] = []

    hist = state.plate_history[track_key]

    if len(hist) > 30:
        hist = hist[-30:]
        state.plate_history[track_key] = hist

    if raw_text:
        norm = _correct_vn_plate(raw_text)
        score = _plate_quality_score(norm, conf, width, height, assoc_score)
        if score > 0:
            hist.append({"text": norm, "score": score})

    if not hist:
        return ""

    counts = {}
    best_score = {}
    for item in hist:
        txt = item["text"]
        counts[txt] = counts.get(txt, 0) + 1
        best_score[txt] = max(best_score.get(txt, 0), item["score"])

    all_texts = sorted(counts.keys(), key=lambda t: (best_score.get(t, 0.0), counts.get(t, 0)), reverse=True)
    cluster_for = {}
    for i, ti in enumerate(all_texts):
        if ti in cluster_for:
            continue
        cluster_for[ti] = ti
        for tj in all_texts[i + 1:]:
            if tj not in cluster_for and _plate_similar_enough(ti, tj):
                cluster_for[tj] = ti

    cluster_votes = {}
    cluster_score = {}
    for txt in all_texts:
        rep = cluster_for[txt]
        cluster_votes[rep] = cluster_votes.get(rep, 0) + counts[txt]
        cluster_score[rep] = max(cluster_score.get(rep, 0.0), best_score[txt])

    best_cand = ""
    best_val = -1.0
    for rep, c in cluster_votes.items():
        sc = cluster_score[rep]
        strong_single = (
            c >= 1
            and sc >= config._SINGLE_VOTE_ACCEPT_SCORE
            and _plate_pattern_score(rep) >= 5.5
        )
        if c >= state.min_stable_votes or strong_single:
            val = c + sc
            if val > best_val:
                best_val = val
                best_cand = rep

    return best_cand


def _plate_history_stats(track_key, text: str) -> tuple:
    text = _correct_vn_plate(text)
    hist = state.plate_history.get(track_key, [])
    votes = 0
    best = 0.0
    for item in hist:
        if item.get("text") == text:
            votes += 1
            best = max(best, float(item.get("score", 0.0)))
    return votes, best


def _normalize_plate_text(raw_text: str) -> str:
    return _correct_vn_plate(raw_text)


def _is_valid_vn_plate(text: str) -> bool:
    if text.count('-') != 1:
        return False
    prefix, suffix = text.split('-')
    if not (2 <= len(prefix) <= 5 and 3 <= len(suffix) <= 6):
        return False
    return True
