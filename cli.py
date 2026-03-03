#!/usr/bin/env python3
"""
Luminousl - Model deployment toolkit CLI

Usage:
    luminousl onnx health-check <model.onnx>
    luminousl onnx to-strongly <input.onnx> [output.onnx]
    luminousl onnx topological-sort <model.onnx>
    luminousl quant optimize-qdq <model.onnx> [-o output.onnx]
    luminousl trt build <model.onnx> -o <engine.trt> [--fp16] [--int8]
    luminousl trt exec <engine.trt> --input <input.bin> --output <output.bin>
    luminousl matool process <model> [best|onnx|trt]
    luminousl perfview create <type> <model> [options]
    luminousl polygraph compare <config.yaml>
"""

import argparse
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def main():
    parser = argparse.ArgumentParser(
        prog="luminousl",
        description="Model deployment toolkit for Zdrive platform"
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    onnx_parser = subparsers.add_parser("onnx", help="ONNX model utilities")
    onnx_subparsers = onnx_parser.add_subparsers(dest="subcommand")

    health_check_parser = onnx_subparsers.add_parser(
        "health-check",
        help="Check ONNX model health"
    )
    health_check_parser.add_argument("model", help="Input ONNX model path")
    health_check_parser.add_argument("--layers", type=str, default=None)
    health_check_parser.add_argument("--profile", type=str, default=None)
    health_check_parser.add_argument("--float-limit", type=float, default=10000)
    health_check_parser.add_argument("--qscale-limit", type=float, default=1000)
    health_check_parser.add_argument("--qscale-underflow-limit", type=float, default=1e-5)
    health_check_parser.add_argument("-s", "--summary", action="store_true")
    health_check_parser.add_argument("--filter", nargs="+", default=[])

    to_strongly_parser = onnx_subparsers.add_parser(
        "to-strongly",
        help="Convert ONNX to FP16 strongly"
    )
    to_strongly_parser.add_argument("input", help="Input ONNX model")
    to_strongly_parser.add_argument("output", nargs="?", help="Output ONNX model")

    topo_sort_parser = onnx_subparsers.add_parser(
        "topological-sort",
        help="Topological sort ONNX nodes"
    )
    topo_sort_parser.add_argument("model", help="Input ONNX model")

    quant_parser = subparsers.add_parser("quant", help="Quantization utilities")
    quant_subparsers = quant_parser.add_subparsers(dest="subcommand")

    optimize_parser = quant_subparsers.add_parser(
        "optimize-qdq",
        help="Optimize QDQ scales"
    )
    optimize_parser.add_argument("model", help="Input ONNX model")
    optimize_parser.add_argument("-o", "--output", help="Output ONNX model")

    trt_parser = subparsers.add_parser("trt", help="TensorRT utilities")
    trt_subparsers = trt_parser.add_subparsers(dest="subcommand")

    build_parser = trt_subparsers.add_parser(
        "build",
        help="Build TensorRT engine from ONNX"
    )
    build_parser.add_argument("model", help="Input ONNX model")
    build_parser.add_argument("-o", "--output", required=True, help="Output engine path")
    build_parser.add_argument("--fp16", action="store_true", help="Enable FP16")
    build_parser.add_argument("--int8", action="store_true", help="Enable INT8")
    build_parser.add_argument("--calibration-data", type=str)
    build_parser.add_argument("--explicit-batch", action="store_true")

    exec_parser = trt_subparsers.add_parser(
        "exec",
        help="Run inference with TensorRT engine"
    )
    exec_parser.add_argument("engine", help="TensorRT engine path")
    exec_parser.add_argument("--input", required=True, help="Input binary file")
    exec_parser.add_argument("--output", required=True, help="Output binary file")

    matool_parser = subparsers.add_parser(
        "matool",
        help="TensorRT model analysis tool"
    )
    matool_parser.add_argument("action", choices=["process", "compare"])
    matool_parser.add_argument("target", help="Model or profile path")
    matool_parser.add_argument("--type", choices=["best", "onnx", "trt"], default="best")

    perfview_parser = subparsers.add_parser(
        "perfview",
        help="Performance analysis tool"
    )
    perfview_parser.add_argument("action", choices=["create", "view"])
    perfview_parser.add_argument("type", choices=["onnx", "trex"])
    perfview_parser.add_argument("model", help="Model path")

    polygraph_parser = subparsers.add_parser(
        "polygraph",
        help="Numerical accuracy comparison"
    )
    polygraph_parser.add_argument("action", choices=["compare"])
    polygraph_parser.add_argument("config", help="Config YAML file")

    args = parser.parse_args()

    if args.command == "onnx" and args.subcommand == "health-check":
        from luminousl.onnx_utils import health_check
        import onnx
        model = onnx.load(args.model)
        checker = health_check.HealthChecker(
            model, args.layers, args.profile,
            args.float_limit, args.qscale_limit,
            args.qscale_underflow_limit,
            os.path.splitext(os.path.basename(args.model))[0]
        )
        checker.check_all()
        filtered_types = health_check.HealthChecker.get_filtered_types(args.filter)
        checker.print_issues(filtered_types, args.summary)

    elif args.command == "onnx" and args.subcommand == "to-strongly":
        from luminousl.onnx_utils.onnx2strongly import main as onnx2strongly_main
        sys.argv = ["onnx2strongly", args.input]
        if args.output:
            sys.argv.append(args.output)
        onnx2strongly_main()

    elif args.command == "onnx" and args.subcommand == "topological-sort":
        from luminousl.onnx_utils import topological_sort
        print("Topological sort: use topological_sort.py directly")

    elif args.command == "quant" and args.subcommand == "optimize-qdq":
        from luminousl.quantization_utils import optimize_qdq_scales
        output = args.output or args.model.replace(".onnx", "_opt.onnx")
        optimize_qdq_scales.main(args.model, output)

    elif args.command == "trt" and args.subcommand == "build":
        from luminousl.tensorrt_utils import onnx_to_trt
        print(f"Building TensorRT engine: {args.model} -> {args.output}")
        print("Note: Please use the module directly for full functionality")

    elif args.command == "trt" and args.subcommand == "exec":
        from luminousl.tensorrt_utils import trt_execution
        print(f"Running inference: {args.engine}")
        print("Note: Please use the module directly for full functionality")

    elif args.command == "matool":
        print(f"MATool: {args.action} {args.target}")
        print("Note: Please use the module directly for full functionality")

    elif args.command == "perfview":
        print(f"Perfview: {args.action} {args.type} {args.model}")
        print("Note: Please use the module directly for full functionality")

    elif args.command == "polygraph":
        print(f"Polygraph: {args.action} {args.config}")
        print("Note: Please use the module directly for full functionality")

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
