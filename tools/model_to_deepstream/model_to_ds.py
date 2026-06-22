#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Convert a .pt or .onnx model into a DeepStream-ready output folder.

This tool intentionally stays self-contained. It does not modify the existing
DeepStream application; it only generates model assets, a custom parser,
config_pgie.txt, a sample usage script, and report.md.

Example usage for yolov11s_14_cls_20241224.onnx:
    python3 tools/model_to_deepstream/model_to_ds.py \
        --model models/yolov11s_14_cls_20241224.onnx \
        --output-dir models_converted/yolov11s_14_cls_20241224 \
        --input-shape 1,3,640,640 \
        --output-format flat6 \
        --num-classes 14 \
        --labels models/labels_vehicle.txt \
        --precision fp16 \
        --batch-size 1
"""

from __future__ import annotations

import argparse
import os
import shlex
import shutil
import subprocess
import sys
import textwrap
from dataclasses import dataclass, field
from pathlib import Path
from string import Template
from typing import Iterable


SUPPORTED_FORMATS = {
    "flat6": {
        "enum": "Flat6",
        "description": "Rows contain x1, y1, x2, y2, confidence, class_id in input pixels.",
    },
    "normalized_flat6": {
        "enum": "NormalizedFlat6",
        "description": "Rows contain normalized x1, y1, x2, y2, confidence, class_id.",
    },
    "yolo_raw": {
        "enum": "YoloRaw",
        "description": "Rows/channels contain cx, cy, w, h, class_scores... without objectness.",
    },
    "yolov5_raw": {
        "enum": "YoloV5Raw",
        "description": "Rows/channels contain cx, cy, w, h, objectness, class_scores...",
    },
}

PRECISION_TO_NETWORK_MODE = {
    "fp32": 0,
    "int8": 1,
    "fp16": 2,
}


class UserError(RuntimeError):
    """Error caused by invalid input or a missing required dependency."""


@dataclass
class CommandResult:
    command: list[str]
    returncode: int
    stdout: str = ""
    stderr: str = ""
    cwd: Path | None = None
    missing: bool = False

    @property
    def ok(self) -> bool:
        return self.returncode == 0 and not self.missing


@dataclass
class OnnxTensorInfo:
    name: str
    shape: list[str]


@dataclass
class OnnxMetadata:
    available: bool = False
    error: str | None = None
    inputs: list[OnnxTensorInfo] = field(default_factory=list)
    outputs: list[OnnxTensorInfo] = field(default_factory=list)
    opsets: list[str] = field(default_factory=list)


@dataclass
class BuildInfo:
    attempted: bool = False
    success: bool = False
    commands: list[CommandResult] = field(default_factory=list)
    symbol_check: CommandResult | None = None
    ldd_check: CommandResult | None = None
    error: str | None = None


@dataclass
class EngineBuildInfo:
    attempted: bool = False
    success: bool = False
    command: CommandResult | None = None
    error: str | None = None


PARSER_CPP_TEMPLATE = Template(
    r'''#include <algorithm>
#include <cmath>
#include <iostream>
#include <vector>

#include "nvdsinfer_custom_impl.h"

enum class ParserFormat {
    Flat6,
    NormalizedFlat6,
    YoloRaw,
    YoloV5Raw
};

static constexpr ParserFormat kParserFormat = ParserFormat::$FORMAT_ENUM;
static constexpr int kNumClasses = $NUM_CLASSES;
static constexpr const char *kOutputFormatName = "$OUTPUT_FORMAT_LABEL";

struct ParsedBox {
    float x1;
    float y1;
    float x2;
    float y2;
    float confidence;
    int class_id;
};

static inline float clampf(float value, float low, float high) {
    return std::max(low, std::min(value, high));
}

static int attrs_per_box() {
    switch (kParserFormat) {
    case ParserFormat::Flat6:
    case ParserFormat::NormalizedFlat6:
        return 6;
    case ParserFormat::YoloRaw:
        return 4 + kNumClasses;
    case ParserFormat::YoloV5Raw:
        return 5 + kNumClasses;
    }
    return 0;
}

static bool resolve_layout(const NvDsInferDims &dims, int attrs, int &num_boxes, bool &channel_major) {
    num_boxes = 0;
    channel_major = false;

    if (dims.numDims == 3) {
        // [1, N, attrs]
        if (dims.d[2] == attrs) {
            num_boxes = dims.d[1];
            channel_major = false;
            return num_boxes > 0;
        }
        // [1, attrs, N]
        if (dims.d[1] == attrs) {
            num_boxes = dims.d[2];
            channel_major = true;
            return num_boxes > 0;
        }
    } else if (dims.numDims == 2) {
        // [N, attrs]
        if (dims.d[1] == attrs) {
            num_boxes = dims.d[0];
            channel_major = false;
            return num_boxes > 0;
        }
        // [attrs, N]
        if (dims.d[0] == attrs) {
            num_boxes = dims.d[1];
            channel_major = true;
            return num_boxes > 0;
        }
    }

    return false;
}

static inline float read_attr(const float *data, int num_boxes, int attrs, int box_idx, int attr_idx, bool channel_major) {
    if (channel_major) {
        return data[attr_idx * num_boxes + box_idx];
    }
    return data[box_idx * attrs + attr_idx];
}

static float threshold_for_class(const NvDsInferParseDetectionParams &detection_params, int class_id) {
    float threshold = 0.25f;
    if (class_id >= 0 &&
        class_id < static_cast<int>(detection_params.perClassPreclusterThreshold.size())) {
        threshold = detection_params.perClassPreclusterThreshold[class_id];
    }
    return threshold;
}

static bool best_class(const float *data,
                       int num_boxes,
                       int attrs,
                       int box_idx,
                       int score_offset,
                       bool channel_major,
                       int &class_id,
                       float &score) {
    class_id = -1;
    score = 0.0f;

    for (int c = 0; c < kNumClasses; ++c) {
        const float candidate = read_attr(data, num_boxes, attrs, box_idx, score_offset + c, channel_major);
        if (!std::isfinite(candidate)) {
            continue;
        }
        if (class_id < 0 || candidate > score) {
            score = candidate;
            class_id = c;
        }
    }

    return class_id >= 0;
}

static bool parse_box(const float *data,
                      int num_boxes,
                      int attrs,
                      int box_idx,
                      bool channel_major,
                      const NvDsInferNetworkInfo &network_info,
                      ParsedBox &box) {
    if (kParserFormat == ParserFormat::Flat6 ||
        kParserFormat == ParserFormat::NormalizedFlat6) {
        box.x1 = read_attr(data, num_boxes, attrs, box_idx, 0, channel_major);
        box.y1 = read_attr(data, num_boxes, attrs, box_idx, 1, channel_major);
        box.x2 = read_attr(data, num_boxes, attrs, box_idx, 2, channel_major);
        box.y2 = read_attr(data, num_boxes, attrs, box_idx, 3, channel_major);
        box.confidence = read_attr(data, num_boxes, attrs, box_idx, 4, channel_major);
        box.class_id = static_cast<int>(std::round(read_attr(data, num_boxes, attrs, box_idx, 5, channel_major)));

        if (kParserFormat == ParserFormat::NormalizedFlat6) {
            box.x1 *= static_cast<float>(network_info.width);
            box.x2 *= static_cast<float>(network_info.width);
            box.y1 *= static_cast<float>(network_info.height);
            box.y2 *= static_cast<float>(network_info.height);
        }
    } else if (kParserFormat == ParserFormat::YoloRaw ||
               kParserFormat == ParserFormat::YoloV5Raw) {
        const float cx = read_attr(data, num_boxes, attrs, box_idx, 0, channel_major);
        const float cy = read_attr(data, num_boxes, attrs, box_idx, 1, channel_major);
        const float w = read_attr(data, num_boxes, attrs, box_idx, 2, channel_major);
        const float h = read_attr(data, num_boxes, attrs, box_idx, 3, channel_major);

        float best_score = 0.0f;
        int best_id = -1;

        if (kParserFormat == ParserFormat::YoloRaw) {
            if (!best_class(data, num_boxes, attrs, box_idx, 4, channel_major, best_id, best_score)) {
                return false;
            }
            box.confidence = best_score;
        } else {
            const float objectness = read_attr(data, num_boxes, attrs, box_idx, 4, channel_major);
            if (!best_class(data, num_boxes, attrs, box_idx, 5, channel_major, best_id, best_score)) {
                return false;
            }
            box.confidence = objectness * best_score;
        }

        box.class_id = best_id;
        box.x1 = cx - w * 0.5f;
        box.y1 = cy - h * 0.5f;
        box.x2 = cx + w * 0.5f;
        box.y2 = cy + h * 0.5f;
    } else {
        return false;
    }

    return std::isfinite(box.x1) && std::isfinite(box.y1) &&
           std::isfinite(box.x2) && std::isfinite(box.y2) &&
           std::isfinite(box.confidence);
}

extern "C" bool NvDsInferParseCustomAuto(
    std::vector<NvDsInferLayerInfo> const &outputLayersInfo,
    NvDsInferNetworkInfo const &networkInfo,
    NvDsInferParseDetectionParams const &detectionParams,
    std::vector<NvDsInferObjectDetectionInfo> &objectList) {

    if (outputLayersInfo.empty()) {
        std::cerr << "ERROR: No output layer found for parser " << kOutputFormatName << std::endl;
        return false;
    }

    const NvDsInferLayerInfo &layer = outputLayersInfo[0];
    const float *output = reinterpret_cast<const float *>(layer.buffer);
    if (output == nullptr) {
        std::cerr << "ERROR: Output buffer is null." << std::endl;
        return false;
    }

    const int attrs = attrs_per_box();
    int num_boxes = 0;
    bool channel_major = false;

    if (!resolve_layout(layer.inferDims, attrs, num_boxes, channel_major)) {
        std::cerr << "ERROR: Unsupported output shape for " << kOutputFormatName
                  << ". Expected attrs_per_box=" << attrs
                  << ", numDims=" << layer.inferDims.numDims << std::endl;
        return false;
    }

    objectList.reserve(objectList.size() + std::min(num_boxes, 300));

    for (int i = 0; i < num_boxes; ++i) {
        ParsedBox box = {};
        if (!parse_box(output, num_boxes, attrs, i, channel_major, networkInfo, box)) {
            continue;
        }

        if (box.class_id < 0 ||
            box.class_id >= kNumClasses ||
            box.class_id >= static_cast<int>(detectionParams.numClassesConfigured)) {
            continue;
        }

        const float threshold = threshold_for_class(detectionParams, box.class_id);
        if (box.confidence < threshold) {
            continue;
        }

        float x1 = std::min(box.x1, box.x2);
        float y1 = std::min(box.y1, box.y2);
        float x2 = std::max(box.x1, box.x2);
        float y2 = std::max(box.y1, box.y2);

        x1 = clampf(x1, 0.0f, static_cast<float>(networkInfo.width - 1));
        y1 = clampf(y1, 0.0f, static_cast<float>(networkInfo.height - 1));
        x2 = clampf(x2, 0.0f, static_cast<float>(networkInfo.width - 1));
        y2 = clampf(y2, 0.0f, static_cast<float>(networkInfo.height - 1));

        const float width = x2 - x1;
        const float height = y2 - y1;
        if (width <= 0.0f || height <= 0.0f) {
            continue;
        }

        NvDsInferObjectDetectionInfo obj = {};
        obj.classId = box.class_id;
        obj.detectionConfidence = box.confidence;
        obj.left = x1;
        obj.top = y1;
        obj.width = width;
        obj.height = height;

        objectList.push_back(obj);
    }

    return true;
}

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Winfinite-recursion"
CHECK_CUSTOM_PARSE_FUNC_PROTOTYPE(NvDsInferParseCustomAuto);
#pragma GCC diagnostic pop
'''
)


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate DeepStream nvinfer assets from a .pt or .onnx model.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--model", required=True, help="Path to .pt or .onnx model.")
    parser.add_argument("--output-dir", required=True, help="Output directory to generate.")
    parser.add_argument("--input-shape", required=True, help="Input shape, for example 1,3,640,640.")
    parser.add_argument("--input-name", default=None, help="Optional ONNX input tensor name.")
    parser.add_argument("--output-name", default=None, help="Optional ONNX output tensor name.")
    parser.add_argument(
        "--output-format",
        required=True,
        choices=sorted(SUPPORTED_FORMATS),
        help="Parser layout for the model output tensor.",
    )
    parser.add_argument("--num-classes", required=True, type=positive_int, help="Number of classes.")
    parser.add_argument("--labels", default=None, help="Existing labels.txt to copy.")
    parser.add_argument("--class-names", default=None, help='Comma-separated labels, for example "license_plate,car".')
    parser.add_argument(
        "--precision",
        choices=sorted(PRECISION_TO_NETWORK_MODE),
        default="fp16",
        help="DeepStream/TensorRT precision.",
    )
    parser.add_argument("--batch-size", type=positive_int, default=1, help="DeepStream batch-size.")
    parser.add_argument("--export-onnx", action="store_true", help="Export .pt input to ONNX using Ultralytics.")
    parser.add_argument("--opset", type=positive_int, default=17, help="ONNX opset for .pt export.")
    parser.add_argument("--imgsz", type=positive_int, default=None, help="Image size for .pt export.")
    parser.add_argument("--force", action="store_true", help="Overwrite output-dir if it already exists.")
    parser.add_argument("--build-engine", action="store_true", help="Try to build TensorRT engine using trtexec.")
    parser.add_argument(
        "--deepstream-auto-engine",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Let DeepStream build/load the TensorRT engine from ONNX.",
    )
    parser.add_argument(
        "--no-cluster",
        action="store_true",
        help="Use cluster-mode=4 for outputs that already have NMS/post-processing.",
    )
    return parser.parse_args(argv)


def positive_int(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"expected integer, got {value!r}") from exc
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be > 0")
    return parsed


def parse_input_shape(value: str) -> list[int]:
    parts = [part.strip() for part in value.split(",") if part.strip()]
    if len(parts) != 4:
        raise UserError("--input-shape must have 4 comma-separated integers, for example 1,3,640,640")

    try:
        shape = [int(part) for part in parts]
    except ValueError as exc:
        raise UserError(f"--input-shape contains a non-integer value: {value}") from exc

    if any(dim <= 0 for dim in shape):
        raise UserError("--input-shape dimensions must all be > 0")

    return shape


def path_is_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.resolve().relative_to(parent.resolve())
        return True
    except ValueError:
        return False


def prepare_output_dir(output_dir: Path, force: bool, protected_paths: Iterable[Path]) -> None:
    resolved = output_dir.resolve()
    dangerous = {Path("/").resolve(), Path.home().resolve(), Path.cwd().resolve()}
    if resolved in dangerous:
        raise UserError(f"Refusing to use dangerous output directory: {resolved}")

    if output_dir.exists():
        if not force:
            raise UserError(f"Output directory already exists: {output_dir}. Use --force to overwrite it.")
        for protected in protected_paths:
            if protected and protected.exists() and path_is_relative_to(protected, output_dir):
                raise UserError(
                    f"Refusing to delete output-dir because it contains input file: {protected}"
                )
        shutil.rmtree(output_dir)

    output_dir.mkdir(parents=True, exist_ok=False)


def run_command(command: list[str], cwd: Path | None = None) -> CommandResult:
    try:
        completed = subprocess.run(
            command,
            cwd=str(cwd) if cwd else None,
            text=True,
            capture_output=True,
            check=False,
        )
    except FileNotFoundError:
        return CommandResult(
            command=command,
            cwd=cwd,
            returncode=127,
            stderr=f"Command not found: {command[0]}",
            missing=True,
        )
    return CommandResult(
        command=command,
        cwd=cwd,
        returncode=completed.returncode,
        stdout=completed.stdout,
        stderr=completed.stderr,
    )


def command_to_text(command: list[str]) -> str:
    return " ".join(shlex.quote(part) for part in command)


def strip_template_indent(text: str, spaces: int = 8) -> str:
    """Remove the function-body indentation from generated text templates."""
    prefix = " " * spaces
    lines = text.splitlines()
    stripped = [line[len(prefix):] if line.startswith(prefix) else line for line in lines]
    trailing_newline = "\n" if text.endswith("\n") else ""
    return "\n".join(stripped) + trailing_newline


def export_pt_to_onnx(model_path: Path, output_dir: Path, imgsz: int, opset: int) -> Path:
    try:
        from ultralytics import YOLO  # type: ignore
    except Exception as exc:  # pragma: no cover - depends on external install
        raise UserError(
            "Cannot export .pt because ultralytics is not installed. "
            "Install ultralytics or provide a .onnx model directly."
        ) from exc

    try:
        model = YOLO(str(model_path))
        exported = model.export(
            format="onnx",
            imgsz=imgsz,
            opset=opset,
            simplify=True,
            dynamic=False,
            nms=False,
        )
    except Exception as exc:  # pragma: no cover - depends on model/runtime
        raise UserError(
            f"Ultralytics export failed for {model_path}: {exc}. "
            "Please export ONNX manually and run the tool with --model your_model.onnx."
        ) from exc

    candidates: list[Path] = []
    if exported:
        candidates.append(Path(exported))
    candidates.append(model_path.with_suffix(".onnx"))
    candidates.extend(sorted(model_path.parent.glob("*.onnx"), key=lambda p: p.stat().st_mtime, reverse=True))

    for candidate in candidates:
        if candidate.exists():
            dest = output_dir / "model.onnx"
            if candidate.resolve() != dest.resolve():
                shutil.copy2(candidate, dest)
            return dest

    raise UserError(
        "Ultralytics reported export success, but no ONNX file was found. "
        "Please provide the exported ONNX manually."
    )


def copy_or_export_model(args: argparse.Namespace, output_dir: Path, input_shape: list[int]) -> Path:
    model_path = Path(args.model).expanduser().resolve()
    if not model_path.exists():
        raise UserError(f"Model file not found: {model_path}")

    suffix = model_path.suffix.lower()
    dest = output_dir / "model.onnx"

    if suffix == ".onnx":
        if model_path.resolve() != dest.resolve():
            shutil.copy2(model_path, dest)
        return dest

    if suffix == ".pt":
        if not args.export_onnx:
            raise UserError("Input model is .pt. Re-run with --export-onnx or provide a .onnx model.")
        imgsz = args.imgsz or max(input_shape[2], input_shape[3])
        return export_pt_to_onnx(model_path, output_dir, imgsz=imgsz, opset=args.opset)

    raise UserError(f"Unsupported model extension {suffix!r}. Expected .onnx or .pt.")


def value_info_shape(value_info: object) -> list[str]:
    tensor_type = value_info.type.tensor_type
    if not tensor_type.HasField("shape"):
        return []

    dims: list[str] = []
    for dim in tensor_type.shape.dim:
        if dim.dim_value > 0:
            dims.append(str(dim.dim_value))
        elif dim.dim_param:
            dims.append(dim.dim_param)
        else:
            dims.append("?")
    return dims


def read_onnx_metadata(onnx_path: Path) -> OnnxMetadata:
    metadata = OnnxMetadata()
    try:
        import onnx  # type: ignore
    except Exception as exc:
        metadata.error = f"onnx Python package is not available: {exc}"
        return metadata

    try:
        model = onnx.load(str(onnx_path))
        initializer_names = {initializer.name for initializer in model.graph.initializer}
        graph_inputs = [item for item in model.graph.input if item.name not in initializer_names]
        metadata.inputs = [OnnxTensorInfo(item.name, value_info_shape(item)) for item in graph_inputs]
        metadata.outputs = [OnnxTensorInfo(item.name, value_info_shape(item)) for item in model.graph.output]
        metadata.opsets = [
            f"{opset.domain or 'ai.onnx'}:{opset.version}"
            for opset in model.opset_import
        ]
        metadata.available = True
    except Exception as exc:
        metadata.error = f"Failed to read ONNX metadata: {exc}"
    return metadata


def labels_from_args(args: argparse.Namespace, output_dir: Path) -> tuple[Path, list[str], list[str]]:
    warnings: list[str] = []
    labels_path = output_dir / "labels.txt"

    if args.labels:
        source = Path(args.labels).expanduser().resolve()
        if not source.exists():
            raise UserError(f"Labels file not found: {source}")
        shutil.copy2(source, labels_path)
        labels = [line.strip() for line in labels_path.read_text(encoding="utf-8").splitlines() if line.strip()]
        if len(labels) != args.num_classes:
            warnings.append(
                f"labels.txt has {len(labels)} non-empty labels, but --num-classes is {args.num_classes}."
            )
        return labels_path, labels, warnings

    if args.class_names:
        labels = [item.strip() for item in args.class_names.split(",") if item.strip()]
        if len(labels) != args.num_classes:
            raise UserError(
                f"--class-names contains {len(labels)} labels, but --num-classes is {args.num_classes}."
            )
    else:
        labels = [f"class_{idx}" for idx in range(args.num_classes)]

    labels_path.write_text("\n".join(labels) + "\n", encoding="utf-8")
    return labels_path, labels, warnings


def generate_parser_cpp(output_format: str, num_classes: int, custom_parser_dir: Path) -> Path:
    info = SUPPORTED_FORMATS[output_format]
    cpp = PARSER_CPP_TEMPLATE.substitute(
        FORMAT_ENUM=info["enum"],
        NUM_CLASSES=str(num_classes),
        OUTPUT_FORMAT_LABEL=output_format,
    )
    parser_path = custom_parser_dir / "nvdsinfer_custom_parser.cpp"
    parser_path.write_text(cpp, encoding="utf-8")
    return parser_path


def generate_makefile(custom_parser_dir: Path) -> Path:
    makefile = strip_template_indent(
        """\
        DS_PATH ?= /opt/nvidia/deepstream/deepstream-9.0

        CXX := g++
        TARGET_LIB := libnvdsinfer_custom_parser.so
        SRCFILES := nvdsinfer_custom_parser.cpp

        CXXFLAGS := -Wall -Wno-infinite-recursion -std=c++17 -shared -fPIC
        INCLUDES := -I$(DS_PATH)/sources/includes \\
                    -I/usr/local/cuda/include

        all: $(TARGET_LIB)

        $(TARGET_LIB): $(SRCFILES)
\t$(CXX) -o $@ $^ $(CXXFLAGS) $(INCLUDES)

        clean:
\trm -f $(TARGET_LIB)
        """
    )
    makefile_path = custom_parser_dir / "Makefile"
    makefile_path.write_text(makefile, encoding="utf-8")
    return makefile_path


def build_parser(custom_parser_dir: Path) -> BuildInfo:
    info = BuildInfo(attempted=True)
    for command in (["make", "clean"], ["make"]):
        result = run_command(command, cwd=custom_parser_dir)
        info.commands.append(result)
        if not result.ok:
            info.error = result.stderr.strip() or result.stdout.strip() or f"{command_to_text(command)} failed"
            return info

    so_path = custom_parser_dir / "libnvdsinfer_custom_parser.so"
    if not so_path.exists():
        info.error = f"Build finished, but {so_path} was not created."
        return info

    nm_result = run_command(["nm", "-D", str(so_path)], cwd=custom_parser_dir)
    if nm_result.ok and "NvDsInferParseCustomAuto" in nm_result.stdout:
        info.symbol_check = CommandResult(
            command=["nm", "-D", str(so_path), "|", "grep", "NvDsInferParseCustomAuto"],
            cwd=custom_parser_dir,
            returncode=0,
            stdout="\n".join(
                line for line in nm_result.stdout.splitlines() if "NvDsInferParseCustomAuto" in line
            )
            + "\n",
        )
    else:
        info.symbol_check = CommandResult(
            command=["nm", "-D", str(so_path), "|", "grep", "NvDsInferParseCustomAuto"],
            cwd=custom_parser_dir,
            returncode=1,
            stdout=nm_result.stdout,
            stderr=nm_result.stderr or "NvDsInferParseCustomAuto symbol not found",
        )
        info.error = info.symbol_check.stderr.strip()
        return info

    info.ldd_check = run_command(["ldd", str(so_path)], cwd=custom_parser_dir)
    info.success = info.ldd_check.ok
    if not info.success:
        info.error = info.ldd_check.stderr.strip() or info.ldd_check.stdout.strip()
    return info


def effective_input_name(args: argparse.Namespace, metadata: OnnxMetadata) -> str | None:
    if args.input_name:
        return args.input_name
    if metadata.inputs:
        return metadata.inputs[0].name
    return None


def build_engine(
    args: argparse.Namespace,
    onnx_path: Path,
    engine_path: Path,
    input_shape: list[int],
    input_name: str | None,
) -> EngineBuildInfo:
    info = EngineBuildInfo(attempted=args.build_engine)
    if not args.build_engine:
        return info

    command = ["trtexec", f"--onnx={onnx_path}", f"--saveEngine={engine_path}"]
    if args.precision == "fp16":
        command.append("--fp16")
    elif args.precision == "int8":
        command.append("--int8")

    if input_name:
        shape = [args.batch_size, input_shape[1], input_shape[2], input_shape[3]]
        command.append(f"--shapes={input_name}:{'x'.join(str(dim) for dim in shape)}")

    result = run_command(command)
    info.command = result
    info.success = result.ok and engine_path.exists()
    if not info.success:
        info.error = result.stderr.strip() or result.stdout.strip() or "trtexec did not create the engine file."
    return info


def write_config(
    args: argparse.Namespace,
    output_dir: Path,
    onnx_path: Path,
    labels_path: Path,
    input_shape: list[int],
    engine_path: Path,
) -> Path:
    channels, height, width = input_shape[1], input_shape[2], input_shape[3]
    network_mode = PRECISION_TO_NETWORK_MODE[args.precision]
    cluster_mode = 4 if args.no_cluster else 2
    onnx_line = f"onnx-file={onnx_path}" if args.deepstream_auto_engine else f"# onnx-file={onnx_path}"

    config = textwrap.dedent(
        f"""\
        [property]
        gpu-id=0

        {onnx_line}
        model-engine-file={engine_path}
        labelfile-path={labels_path}

        batch-size={args.batch_size}
        network-mode={network_mode}
        network-type=0
        process-mode=1
        gie-unique-id=1
        interval=0

        num-detected-classes={args.num_classes}

        infer-dims={channels};{height};{width}
        maintain-aspect-ratio=1
        symmetric-padding=1

        net-scale-factor=0.00392156862745098
        model-color-format=0

        custom-lib-path={output_dir / "custom_parser" / "libnvdsinfer_custom_parser.so"}
        parse-bbox-func-name=NvDsInferParseCustomAuto

        cluster-mode={cluster_mode}
        pre-cluster-threshold=0.25
        nms-iou-threshold=0.45
        topk=300

        [class-attrs-all]
        pre-cluster-threshold=0.25
        nms-iou-threshold=0.45
        topk=300
        """
    )
    config_path = output_dir / "config_pgie.txt"
    config_path.write_text(config, encoding="utf-8")
    return config_path


def write_run_script(output_dir: Path, config_path: Path) -> Path:
    script = textwrap.dedent(
        f"""\
        #!/bin/bash
        set -e

        MODEL_DIR="$(cd "$(dirname "$0")" && pwd)"
        CONFIG="$MODEL_DIR/config_pgie.txt"

        echo "[INFO] Generated DeepStream config:"
        echo "$CONFIG"

        echo ""
        echo "Ban co the dung config nay trong Python DeepStream app:"
        echo "pgie.set_property(\\"config-file-path\\", \\"$CONFIG\\")"

        echo ""
        echo "Hoac neu app cua ban ho tro --pgie-config:"
        echo "python3 /workspace/ds_lpr_project/src/app_lpr.py <video_or_rtsp_url> --pgie-config $CONFIG"

        echo ""
        echo "Config path absolute when generated:"
        echo "{config_path}"
        """
    )
    script_path = output_dir / "run_deepstream_example.sh"
    script_path.write_text(script, encoding="utf-8")
    script_path.chmod(0o755)
    return script_path


def format_tensor_infos(items: list[OnnxTensorInfo]) -> str:
    if not items:
        return "- N/A"
    return "\n".join(f"- `{item.name}` shape `[{', '.join(item.shape) if item.shape else '?'}]`" for item in items)


def format_command_result(result: CommandResult | None) -> str:
    if result is None:
        return "- Not run"

    lines = [
        f"- Command: `{command_to_text(result.command)}`",
        f"- Return code: `{result.returncode}`",
    ]
    if result.cwd:
        lines.append(f"- CWD: `{result.cwd}`")
    if result.stdout.strip():
        lines.append("\nstdout:\n```text\n" + result.stdout.strip() + "\n```")
    if result.stderr.strip():
        lines.append("\nstderr:\n```text\n" + result.stderr.strip() + "\n```")
    return "\n".join(lines)


def write_report(
    args: argparse.Namespace,
    output_dir: Path,
    source_model: Path,
    onnx_path: Path,
    labels_path: Path,
    labels: list[str],
    input_shape: list[int],
    metadata: OnnxMetadata,
    config_path: Path,
    parser_path: Path,
    makefile_path: Path,
    build_info: BuildInfo,
    engine_info: EngineBuildInfo,
    engine_path: Path,
    warnings: list[str],
) -> Path:
    parser_so_path = output_dir / "custom_parser" / "libnvdsinfer_custom_parser.so"
    engine_strategy = "DeepStream auto build from ONNX" if args.deepstream_auto_engine else "Prebuilt engine only"
    if args.build_engine:
        engine_strategy = "trtexec build" if engine_info.success else "trtexec attempted; DeepStream fallback depends on config"

    if args.precision == "int8":
        warnings.append("INT8 usually requires calibration or explicit dynamic ranges; this tool does not generate calibration data.")
    if not args.deepstream_auto_engine and not engine_path.exists():
        warnings.append("--no-deepstream-auto-engine was used, but no TensorRT engine exists yet.")

    report = strip_template_indent(
        f"""\
        # Model To DeepStream Report

        ## 1. Model goc
        `{source_model}`

        ## 2. Model ONNX output
        `{onnx_path}`

        ## 3. Input shape nguoi dung nhap
        `{','.join(str(dim) for dim in input_shape)}`

        ## 4. Output format nguoi dung nhap
        `{args.output_format}` - {SUPPORTED_FORMATS[args.output_format]["description"]}

        ## 5. So class
        `{args.num_classes}`

        ## 6. Labels path
        `{labels_path}`

        Labels:
        ```text
        {os.linesep.join(labels)}
        ```

        ## 7. Parser type
        Custom C++ parser for `{args.output_format}`

        ## 8. Parser function
        `NvDsInferParseCustomAuto`

        ## 9. Parser .so path
        `{parser_so_path}`

        Source:
        - `{parser_path}`
        - `{makefile_path}`

        ## 10. Config nvinfer path
        `{config_path}`

        ## 11. Engine strategy
        `{engine_strategy}`

        Engine path:
        `{engine_path}`

        ## 12. Lenh kiem tra .so
        ```bash
        cd {output_dir}
        nm -D custom_parser/libnvdsinfer_custom_parser.so | grep NvDsInferParseCustomAuto
        ldd custom_parser/libnvdsinfer_custom_parser.so
        ```

        ## 13. Cach dung config trong DeepStream Python Apps
        ```python
        pgie.set_property("config-file-path", "{config_path}")
        ```

        Hoac voi app co tham so config:

        ```bash
        python3 /workspace/ds_lpr_project/src/app_lpr.py <video_or_rtsp_url> --pgie-config {config_path}
        ```

        ## 14. Cac loi thuong gap
        - Missing `nvdsinfer_custom_impl.h`: install/mount DeepStream 9.0, or run `make DS_PATH=/path/to/deepstream-9.0`.
        - Symbol not found: confirm `parse-bbox-func-name=NvDsInferParseCustomAuto` and rebuild the parser.
        - No detections: verify `--output-format`, coordinate scale, labels, and `num-detected-classes`.
        - Wrong boxes: choose `normalized_flat6` only when x/y are 0..1; use `flat6` for pixel coordinates.
        - INT8 engine build failure: provide calibration/dynamic range support outside this generator.

        ## ONNX metadata
        """
    )

    if metadata.available:
        report += "\nInputs:\n" + format_tensor_infos(metadata.inputs) + "\n"
        report += "\nOutputs:\n" + format_tensor_infos(metadata.outputs) + "\n"
        report += "\nOpset:\n" + ("\n".join(f"- `{item}`" for item in metadata.opsets) if metadata.opsets else "- N/A") + "\n"
    else:
        report += f"\nONNX metadata unavailable: {metadata.error or 'unknown error'}\n"

    report += "\n## Parser build\n"
    report += f"- Attempted: `{build_info.attempted}`\n"
    report += f"- Success: `{build_info.success}`\n"
    if build_info.error:
        report += f"- Error: `{build_info.error}`\n"
    for result in build_info.commands:
        report += "\n" + format_command_result(result) + "\n"
    if build_info.symbol_check:
        report += "\nSymbol check:\n" + format_command_result(build_info.symbol_check) + "\n"
    if build_info.ldd_check:
        report += "\nldd check:\n" + format_command_result(build_info.ldd_check) + "\n"

    report += "\n## TensorRT engine build\n"
    report += f"- Attempted: `{engine_info.attempted}`\n"
    report += f"- Success: `{engine_info.success}`\n"
    if engine_info.error:
        report += f"- Error: `{engine_info.error}`\n"
    if engine_info.command:
        report += "\n" + format_command_result(engine_info.command) + "\n"

    if warnings:
        report += "\n## Warnings\n"
        report += "\n".join(f"- {warning}" for warning in warnings) + "\n"

    report_path = output_dir / "report.md"
    report_path.write_text(report, encoding="utf-8")
    return report_path


def validate_user_names(args: argparse.Namespace, metadata: OnnxMetadata) -> list[str]:
    warnings: list[str] = []
    if args.input_name and metadata.inputs:
        known = {item.name for item in metadata.inputs}
        if args.input_name not in known:
            warnings.append(f"--input-name `{args.input_name}` was not found in ONNX inputs: {sorted(known)}")
    if args.output_name and metadata.outputs:
        known = {item.name for item in metadata.outputs}
        if args.output_name not in known:
            warnings.append(f"--output-name `{args.output_name}` was not found in ONNX outputs: {sorted(known)}")
    return warnings


def print_result(result: CommandResult) -> None:
    print(f"$ {command_to_text(result.command)}")
    if result.stdout:
        print(result.stdout.rstrip())
    if result.stderr:
        print(result.stderr.rstrip())
    if result.returncode != 0:
        print(f"[WARN] command exited with code {result.returncode}")


def final_checks(output_dir: Path, config_path: Path) -> None:
    print("\n[INFO] Final verification commands")
    checks = [
        ["ls", "-lh", "model.onnx"],
        ["ls", "-lh", "labels.txt"],
        ["ls", "-lh", "config_pgie.txt"],
        ["ls", "-lh", "custom_parser/libnvdsinfer_custom_parser.so"],
    ]
    for command in checks:
        print_result(run_command(command, cwd=output_dir))

    so_path = output_dir / "custom_parser" / "libnvdsinfer_custom_parser.so"
    nm_result = run_command(["nm", "-D", str(so_path)], cwd=output_dir)
    if nm_result.ok and "NvDsInferParseCustomAuto" in nm_result.stdout:
        filtered = "\n".join(line for line in nm_result.stdout.splitlines() if "NvDsInferParseCustomAuto" in line)
        print("$ nm -D custom_parser/libnvdsinfer_custom_parser.so | grep NvDsInferParseCustomAuto")
        print(filtered)
    else:
        print("$ nm -D custom_parser/libnvdsinfer_custom_parser.so | grep NvDsInferParseCustomAuto")
        if nm_result.stderr.strip():
            print(nm_result.stderr.strip())
        print("[WARN] NvDsInferParseCustomAuto symbol check failed")

    grep_result = run_command(
        [
            "grep",
            "-nE",
            "onnx-file|model-engine-file|custom-lib-path|parse-bbox-func-name|num-detected-classes",
            config_path.name,
        ],
        cwd=output_dir,
    )
    print_result(grep_result)


def main(argv: list[str]) -> int:
    try:
        args = parse_args(argv)
        input_shape = parse_input_shape(args.input_shape)

        source_model = Path(args.model).expanduser().resolve()
        labels_source = Path(args.labels).expanduser().resolve() if args.labels else None
        output_dir = Path(args.output_dir).expanduser().resolve()
        protected = [path for path in (source_model, labels_source) if path is not None]

        if not source_model.exists():
            raise UserError(f"Model file not found: {source_model}")
        if labels_source and not labels_source.exists():
            raise UserError(f"Labels file not found: {labels_source}")

        prepare_output_dir(output_dir, force=args.force, protected_paths=protected)

        onnx_path = copy_or_export_model(args, output_dir, input_shape)
        metadata = read_onnx_metadata(onnx_path)
        warnings = validate_user_names(args, metadata)

        labels_path, labels, label_warnings = labels_from_args(args, output_dir)
        warnings.extend(label_warnings)

        custom_parser_dir = output_dir / "custom_parser"
        custom_parser_dir.mkdir(parents=True, exist_ok=True)
        parser_path = generate_parser_cpp(args.output_format, args.num_classes, custom_parser_dir)
        makefile_path = generate_makefile(custom_parser_dir)

        build_info = build_parser(custom_parser_dir)

        engine_path = output_dir / f"model_b{args.batch_size}_gpu0_{args.precision}.engine"
        input_name = effective_input_name(args, metadata)
        engine_info = build_engine(args, onnx_path, engine_path, input_shape, input_name)

        config_path = write_config(args, output_dir, onnx_path, labels_path, input_shape, engine_path)
        run_script = write_run_script(output_dir, config_path)
        report_path = write_report(
            args=args,
            output_dir=output_dir,
            source_model=source_model,
            onnx_path=onnx_path,
            labels_path=labels_path,
            labels=labels,
            input_shape=input_shape,
            metadata=metadata,
            config_path=config_path,
            parser_path=parser_path,
            makefile_path=makefile_path,
            build_info=build_info,
            engine_info=engine_info,
            engine_path=engine_path,
            warnings=warnings,
        )

        print(f"[OK] Generated DeepStream model package: {output_dir}")
        print(f"[OK] ONNX: {onnx_path}")
        print(f"[OK] Labels: {labels_path}")
        print(f"[OK] Parser source: {parser_path}")
        print(f"[OK] Config: {config_path}")
        print(f"[OK] Example script: {run_script}")
        print(f"[OK] Report: {report_path}")

        if not build_info.success:
            print("[WARN] Parser .so build did not succeed. See report.md for build logs.")
        if args.build_engine and not engine_info.success:
            print("[WARN] TensorRT engine build did not succeed. See report.md for trtexec logs.")

        final_checks(output_dir, config_path)
        return 0

    except UserError as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
