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


def _plate_quality_score(text: str, conf: float, width: int, height: int, association_score: float, is_moto: bool = False) -> float:
    text = _correct_vn_plate(text, is_moto=is_moto)
    score = _plate_pattern_score(text)
    if score <= 0.0:
        return 0.0

    score += conf * 2.0
    size_factor = min(1.0, max(0, width) * max(0, height) / 10000.0)
    score += size_factor
    score += max(0.0, association_score)

    return max(0.0, score)


def _correct_vn_plate(raw: str, is_moto: bool = False) -> str:
    if "-" in raw:
        parts = raw.split("-")
        if len(parts) == 2:
            top = re.sub(r'[^A-Z0-9]', '', parts[0].upper())
            bot = re.sub(r'[^A-Z0-9]', '', parts[1].upper())

            bot_norm = "".join(
                config._LETTER_TO_DIGIT.get(ch, ch) if not ch.isdigit() else ch
                for ch in bot
            )

            # Case 1: top ends with series digit (e.g. 89B0), bot is 3 digits (712)
            # → moto plate split across lines
            if (len(top) == 4 and len(bot_norm) == 3
                    and top[:2].isdigit() and top[2].isalpha() and top[3].isdigit()
                    and bot_norm.isdigit()):
                raw = f"{top}-{top[3]}{bot_norm}"

            # Case 2: moto 2-line — top=province+series (3 chars), bot=series_digit+number (4 chars).
            # "89B-0712" → series digit=0, number=712 → "89B0-712"
            elif (is_moto and len(top) == 3 and len(bot_norm) == 4
                  and bot_norm[0].isdigit() and bot_norm[1:].isdigit()):
                raw = f"{top}{bot_norm[0]}-{bot_norm[1:]}"

            # Case 3: car 2-line — CTC collapsed one '0' from number group (007→07)
            # "89B-0712" → "89B-00712"
            elif (not is_moto and len(top) == 3 and len(bot_norm) == 4
                  and top[:2].isdigit() and top[2].isalpha()
                  and top[2] in config._VALID_SERIES
                  and bot_norm.startswith("0") and bot_norm.isdigit()):
                raw = f"{top}-0{bot_norm}"

    text = _normalize_plate_for_output(raw)
    if not text:
        return ""

    # Pre-normalize suffix positions (letter→digit) so Rules below fire correctly
    # even when OCR confuses e.g. T→7, I→1, D→0 in the number suffix.
    # Guard: text[3] must not be a valid series letter (could be 2-letter series e.g. 29LD...).
    def _prenorm_suffix(t: str, start: int) -> str:
        fixed = list(t)
        for i in range(start, len(fixed)):
            if not fixed[i].isdigit():
                fixed[i] = config._LETTER_TO_DIGIT.get(fixed[i], fixed[i])
        return "".join(fixed)

    if len(text) == 6 and is_moto and text[:2].isdigit() and text[2].isalpha():
        text = _prenorm_suffix(text, 3)
    elif (len(text) == 7 and text[:2].isdigit() and text[2].isalpha()
            and text[3] not in config._VALID_SERIES):
        text = _prenorm_suffix(text, 3)
    elif len(text) == 8 and text[:2].isdigit() and text[2].isalpha() and text[3].isalpha():
        text = _prenorm_suffix(text, 4)

    # Rule 0: 6-char moto plate — CTC collapsed series digit with matching first number digit.
    # E.g. OCR "89B012" from plate "89B0-012" (series digit 0 + number 012, 0,0 collapsed).
    # Restore by duplicating text[3] (the surviving digit represents both collapsed chars).
    if (len(text) == 6 and is_moto
            and text[:2].isdigit() and text[2].isalpha()
            and text[2] in config._VALID_SERIES
            and text[3].isdigit() and text[4:].isdigit()):
        text = text[:4] + text[3] + text[4:]

    if len(text) == 7:
        # Rule A: moto — series digit was split from number group; plate is already correctly decoded.
        # 7-char moto output means NO CTC collapse occurred — return as-is.
        # The elif ensures Rule B (car) does not fire when is_moto=True.
        if is_moto:
            pass

        # Rule B: car — leading '0' in suffix is CTC-collapsed from '00'.
        # "89B0712" → "89B00712" (from plate 89B-007.12)
        elif text[:2].isdigit() and text[2].isalpha() and text[3] == "0" and text[4:].isdigit():
            text = text[:4] + "0" + text[4:]

    # Rule C: dual-series car plate "29LD0712" → "29LD00712"
    elif len(text) == 8:
        if text[:2].isdigit() and text[2:4].isalpha() and text[4] == "0" and text[5:].isdigit():
            text = text[:5] + "0" + text[5:]

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


def _stable_plate(track_key, raw_text: str, conf: float, width: int, height: int, assoc_score: float, is_moto: bool = False) -> str:
    if track_key not in state.plate_history:
        state.plate_history[track_key] = []

    hist = state.plate_history[track_key]

    max_hist = config._PLATE_HISTORY_LEN * 2
    if len(hist) > max_hist:
        hist = hist[-max_hist:]
        state.plate_history[track_key] = hist

    if raw_text:
        norm = _correct_vn_plate(raw_text, is_moto=is_moto)
        score = _plate_quality_score(norm, conf, width, height, assoc_score, is_moto=is_moto)
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


def _plate_history_stats(track_key, text: str, is_moto: bool = False) -> tuple:
    text = _correct_vn_plate(text, is_moto=is_moto)
    hist = state.plate_history.get(track_key, [])
    votes = 0
    best = 0.0
    for item in hist:
        if item.get("text") == text:
            votes += 1
            best = max(best, float(item.get("score", 0.0)))
    return votes, best


