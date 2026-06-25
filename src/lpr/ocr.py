"""
ocr.py — CTC decode for lpr_ocr.20240305.onnx (37-class vocab, blank=0).

Input tensor  : layer "output", float32 [T=15, 37], already softmax.
Vocab         : ❌(blank) 0-9 A-Z(no O) _(2-line separator)
Preprocessing : model normalizes internally (raw [0-255] → ImageNet norm).
"""

import ctypes
import sys
import pyds
import lpr_config as config

# Vocab matches C++ constexpr LPR_VOCAB in the model source
_NEW_VOCAB = [
    "❌",
    "0","1","2","3","4","5","6","7","8","9",
    "A","B","C","D","E","F","G","H","I","J",
    "K","L","M","N","P","Q","R","S","T","U",
    "V","W","X","Y","Z","_",
]
_NEW_BLANK = 0
_NEW_LAYER = "output"
_NEW_N_CLS = len(_NEW_VOCAB)  # 37

_warn_layer_missing = False


def _decode_new_ctc(floats, T, C):
    """Greedy CTC: argmax → collapse repeats → remove blank → _ to -."""
    chars, confs, prev = [], [], _NEW_BLANK
    for t in range(T):
        row_start = t * C
        best_idx = max(range(C), key=lambda i: floats[row_start + i])
        if best_idx != _NEW_BLANK and best_idx != prev:
            chars.append(_NEW_VOCAB[best_idx])
            confs.append(floats[row_start + best_idx])
        prev = best_idx
    text = "".join(chars).replace("_", "-")
    conf = sum(confs) / len(confs) if confs else 0.0
    return text, conf


def _read_lpr_text(obj_meta, gie_unique_id: int = config.SGIE3_UNIQUE_ID) -> tuple:
    """Read OCR text from lpr_ocr.20240305.onnx output tensor metadata."""
    global _warn_layer_missing
    l_user = obj_meta.obj_user_meta_list
    while l_user is not None:
        try:
            user_meta = pyds.NvDsUserMeta.cast(l_user.data)
        except StopIteration:
            break
        if user_meta.base_meta.meta_type == pyds.NvDsMetaType.NVDSINFER_TENSOR_OUTPUT_META:
            tensor_meta = pyds.NvDsInferTensorMeta.cast(user_meta.user_meta_data)
            if tensor_meta.unique_id == gie_unique_id:
                found = False
                for i in range(tensor_meta.num_output_layers):
                    layer = pyds.get_nvds_LayerInfo(tensor_meta, i)
                    if layer.layerName != _NEW_LAYER:
                        continue
                    found = True
                    dims = layer.dims
                    # dims can be [T, C] or [1, T, C]
                    if dims.numDims == 3:
                        T, C = dims.d[1], dims.d[2]
                    elif dims.numDims == 2:
                        T, C = dims.d[0], dims.d[1]
                    else:
                        break
                    if C != _NEW_N_CLS:
                        sys.stderr.write(
                            f"[WARN] ocr: expected {_NEW_N_CLS} classes, got {C}\n"
                        )
                        break
                    ptr = ctypes.cast(
                        pyds.get_ptr(layer.buffer),
                        ctypes.POINTER(ctypes.c_float),
                    )
                    floats = [ptr[j] for j in range(T * C)]
                    text, conf = _decode_new_ctc(floats, T, C)
                    return (text, conf) if text else ("", 0.0)
                if not found and not _warn_layer_missing:
                    sys.stderr.write(
                        f"[WARN] ocr: layer '{_NEW_LAYER}' not found in SGIE3 output.\n"
                    )
                    _warn_layer_missing = True
        try:
            l_user = l_user.next
        except StopIteration:
            break
    return "", 0.0
