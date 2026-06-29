import os
import re
import configparser
import lpr_config as config

_PATH_SUB_RE = re.compile(
    r"(=\s*)(" + "|".join(
        re.escape(p) for p in sorted(config._OLD_ROOTS, key=len, reverse=True)
    ) + r")(/|$)"
)


def _apply_property_overrides(text: str, overrides: dict) -> str:
    if not overrides:
        return text

    lines = text.splitlines()
    seen = set()
    key_patterns = {
        key: re.compile(r"^(\s*)(" + re.escape(key) + r")(\s*=\s*)(.*)$")
        for key in overrides
    }

    for idx, line in enumerate(lines):
        for key, pattern in key_patterns.items():
            match = pattern.match(line)
            if match:
                lines[idx] = "{}{}{}{}".format(match.group(1), key, match.group(3), overrides[key])
                seen.add(key)

    missing = [key for key in overrides if key not in seen]
    if missing:
        insert_at = None
        for idx, line in enumerate(lines):
            if line.strip() == "[property]":
                insert_at = idx + 1
                break
        if insert_at is None:
            insert_at = len(lines)
        additions = ["{}={}".format(key, overrides[key]) for key in missing]
        lines[insert_at:insert_at] = additions

    return "\n".join(lines) + ("\n" if text.endswith("\n") else "")


def _runtime_config_path(path: str, property_overrides: dict = None) -> str:
    os.makedirs(config.RUNTIME_CONFIG_DIR, exist_ok=True)
    with open(path, "r", encoding="utf-8") as f:
        text = f.read()
    text = _PATH_SUB_RE.sub(r"\g<1>" + config.PROJECT_ROOT + r"\3", text)
    text = _apply_property_overrides(text, property_overrides or {})
    out_path = os.path.join(config.RUNTIME_CONFIG_DIR, os.path.basename(path))
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(text)
    return out_path


def _pgie_engine_path_for_batch(batch_size: int) -> str:
    requested = max(1, int(batch_size))

    engine_path = os.path.join(config.PROJECT_ROOT, "models", "vehicle_parking_detect.onnx_b1_gpu0_fp16.engine")

    try:
        ini = configparser.ConfigParser()
        ini.read(config.PGIE_CONFIG_PATH)
        if ini.has_option("property", "model-engine-file"):
            configured_path = ini.get("property", "model-engine-file")
            for old_root in config._OLD_ROOTS:
                if configured_path == old_root or configured_path.startswith(old_root + os.sep):
                    configured_path = configured_path.replace(old_root, config.PROJECT_ROOT, 1)
            engine_path = configured_path
    except Exception:
        pass

    engine_dir = os.path.dirname(engine_path)
    engine_filename = os.path.basename(engine_path)

    match = re.search(r"_b(\d+)_", engine_filename)
    if not match:
        return engine_path

    prefix_part = engine_filename[:match.start()]
    suffix_part = engine_filename[match.end():]

    preferred_batches = (1, 4, 7, 8, 12, 16)
    for batch in preferred_batches:
        if requested <= batch:
            pref_filename = re.sub(r"_b\d+_", f"_b{batch}_", engine_filename)
            pref_path = os.path.join(engine_dir, pref_filename)
            if os.path.exists(pref_path):
                return pref_path
            pref_filename_onnx = re.sub(r"_b\d+_", f"_b{batch}_", engine_filename.replace(prefix_part, prefix_part + ".onnx", 1))
            pref_path_onnx = os.path.join(engine_dir, pref_filename_onnx)
            if os.path.exists(pref_path_onnx):
                return pref_path_onnx
            break

    exact_filename = re.sub(r"_b\d+_", f"_b{requested}_", engine_filename)
    exact_path = os.path.join(engine_dir, exact_filename)
    if os.path.exists(exact_path):
        return exact_path
    exact_filename_onnx = re.sub(r"_b\d+_", f"_b{requested}_", engine_filename.replace(prefix_part, prefix_part + ".onnx", 1))
    exact_path_onnx = os.path.join(engine_dir, exact_filename_onnx)
    if os.path.exists(exact_path_onnx):
        return exact_path_onnx

    pattern = re.compile(re.escape(prefix_part) + r"(?:\.onnx)?_b(\d+)_" + re.escape(suffix_part))

    candidates = []
    try:
        for name in os.listdir(engine_dir):
            m = pattern.match(name)
            if not m:
                continue
            batch = int(m.group(1))
            if batch >= requested:
                candidates.append((batch, os.path.join(engine_dir, name)))
    except OSError:
        candidates = []

    if candidates:
        return min(candidates, key=lambda item: item[0])[1]

    return exact_path
