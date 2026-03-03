import re
import copy
from collections import OrderedDict
import traceback
import json
import argparse
import os

class KernelCompareReport:
    CRITICAL_ONNX = {
        "AveragePool",
        "BatchNormalization",
        "Conv",
        "DeConv",
        "ConvTranspose",
        "DeformConv",
        "Gemm",
        "GlobalAveragePool",
        "GlobalLpPool",
        "GlobalMaxPool",
        "GroupNormalization",
        "InstanceNormalization",
        "LayerNormalization",
        "ReduceMean",
        "LpNormalization",
        "LpPool",
        "MatMul",
        "MaxPool",
        "MeanVarianceNormalization",
        "Resize",
        "Softmax",
        "Upsample",
    }

    CRITICAL_KERNEL = {
        "correlation",
        "CaskGemmConvolution",
        "conv_act_pool",
        "deconv",
        "CaskConvActPool",
        "gemm",
        "fusion",
        "CaskConvolution",
        "maxpool",
        "avgpool",
        "CaskPooling",
        "CaskSoftMaxV2",
    }

    DATA_GETTER = OrderedDict(
        [
            ("Name", lambda l: l["Name"]),
            (
                "LayerType",
                lambda l: f'{"★" if l["is_critical_kernel"] else ""}{l["LayerType"]}',
            ),
            ("TacticName", lambda l: l.get("TacticName")),
            ("ONNX", lambda l: "\n".join(l["source_onnx"])),
            ("ms", lambda l: l["profile"]["medianMs"]),
            ("%", lambda l: l["profile"]["percentage"] / 100.0),
        ]
    )

    DATA_REDUCE = OrderedDict([("ms", lambda x, y: y if x is None else x + y)])

    DATA_REDUCE_COMPARE = OrderedDict(
        [
            (
                "Δms",
                lambda x, y: (x["ms"] / y["ms"] - 1 if x["ms"] and y["ms"] else None),
            )
        ]
    )

    onnx_parser = re.compile(r"\[ONNX Layer: [^\]]+\]")
    onnx_name_parser = re.compile(r"\[ONNX Layer: ([^\]]+)\]")

    def __init__(self):
        self.match_results = OrderedDict()
        self.num_compare_sheets = 0

    def create_match_report(
        self,
        layer_name_type_mapping,
        name_a,
        graph_a,
        profile_a,
        name_b,
        graph_b,
        profile_b,
    ):
        kernelinfo_a = self.preprocess_kernelinfo(layer_name_type_mapping, graph_a, profile_a)
        kernelinfo_b = self.preprocess_kernelinfo(layer_name_type_mapping, graph_b, profile_b)
        match_result = self.match(name_a, kernelinfo_a, name_b, kernelinfo_b)
        return match_result

    @staticmethod
    def match(name_a, kernelinfo_a, name_b, kernelinfo_b):
        # the following implements a regular "union find set" algorithm
        union_idx = [*range(len(kernelinfo_a) + len(kernelinfo_b))]
        union_has_critical_kernel = [
            l["is_critical_kernel"] for l in kernelinfo_a + kernelinfo_b
        ]

        def find_root(a):
            if union_idx[a] == a:
                return a
            union_idx[a] = find_root(union_idx[a])
            return union_idx[a]

        def ufs_merge(a, b):
            root_a = find_root(a)
            root_b = find_root(b)
            common_root = min(root_a, root_b)
            union_idx[root_a] = common_root
            union_idx[root_b] = common_root
            union_has_critical_kernel[common_root] = (
                union_has_critical_kernel[root_b] or union_has_critical_kernel[root_a]
            )

        for kernels, idx_offset in (kernelinfo_a, 0), (kernelinfo_b, len(kernelinfo_a)):
            for critical_kernel_idx, critical_kernel in enumerate(kernels):
                if not critical_kernel["is_critical_kernel"]:
                    continue
                critical_kernel_idx = critical_kernel_idx + idx_offset
                for kernel_idx, kernel in enumerate(kernels):
                    kernel_idx = kernel_idx + idx_offset
                    if critical_kernel_idx == kernel_idx:
                        continue
                    has_critical_connection = union_has_critical_kernel[
                        find_root(kernel_idx)
                    ]
                    is_critical = kernel["is_critical_kernel"]
                    if critical_kernel["source_critical_onnx"] & kernel["source_onnx"]:
                        if has_critical_connection:
                            if is_critical:
                                ufs_merge(critical_kernel_idx, kernel_idx)
                        else:
                            ufs_merge(critical_kernel_idx, kernel_idx)

        for a_idx, kernel_a in enumerate(kernelinfo_a):
            for b_idx, kernel_b in enumerate(kernelinfo_b):
                b_idx = b_idx + len(kernelinfo_a)
                if kernel_a["is_critical_kernel"] and kernel_b["is_critical_kernel"]:
                    if (
                        kernel_a["source_critical_onnx"]
                        & kernel_b["source_critical_onnx"]
                    ):
                        ufs_merge(a_idx, b_idx)

        for a_idx, kernel_a in enumerate(kernelinfo_a):
            for b_idx, kernel_b in enumerate(kernelinfo_b):
                b_idx = b_idx + len(kernelinfo_a)
                a_has_critical_connection = union_has_critical_kernel[find_root(a_idx)]
                b_has_critical_connection = union_has_critical_kernel[find_root(b_idx)]
                if kernel_a["is_critical_kernel"] and kernel_b["is_critical_kernel"]:
                    continue
                if kernel_a["source_onnx"] & kernel_b["source_onnx"]:
                    if a_has_critical_connection and b_has_critical_connection:
                        continue
                    else:
                        ufs_merge(a_idx, b_idx)

        groups = OrderedDict()
        for a_idx, kernel_a in enumerate(kernelinfo_a):
            kernel_a["group_idx"] = find_root(a_idx)
            if kernel_a["group_idx"] not in groups:
                groups[kernel_a["group_idx"]] = OrderedDict(
                    [(name_a, []), (name_b, [])]
                )
            groups[kernel_a["group_idx"]][name_a].append(kernel_a)
        for b_idx, kernel_b in enumerate(kernelinfo_b):
            kernel_b["group_idx"] = find_root(len(kernelinfo_a) + b_idx)
            if kernel_b["group_idx"] not in groups:
                groups[kernel_b["group_idx"]] = OrderedDict(
                    [(name_a, []), (name_b, [])]
                )
            groups[kernel_b["group_idx"]][name_b].append(kernel_b)

        for key in groups:
            for column in groups[key]:
                items = groups[key][column]
                for row in items:
                    row["source_onnx"] = list(row["source_onnx"])
                    row["source_critical_onnx"] = list(row["source_critical_onnx"])

        return groups

    @classmethod
    def is_critical_onnx(cls, op_type):
        return op_type in cls.CRITICAL_ONNX

    @classmethod
    def is_critical_kernel(cls, kernel):
        if kernel["LayerType"] in cls.CRITICAL_KERNEL:
            return True
        if kernel["LayerType"] == "kgen":
            tactic = kernel.get("TacticName", "")
            if "_mha" in tactic:
                return True
            if "MaxSubExp" in tactic:
                return True
            if "Mea" in tactic:
                return True
            if "Iot" in tactic:
                return True
        return False

    @classmethod
    def preprocess_kernelinfo(cls, layer_name_type_mapping, graph, profile):
        kernelinfo = []
        for kernel in graph["Layers"]:
            new_kernel = copy.copy(kernel)
            # source_onnx = [
            #     cls.onnx_name_parser.search(l).group(1)
            #     for l in cls.onnx_parser.findall(new_kernel["Metadata"])
            # ]
            source_onnx = new_kernel.get("ONNXNames", [])
            new_kernel["source_onnx"] = set(source_onnx)
            critical_onnx = [
                l for l in source_onnx if cls.is_critical_onnx(layer_name_type_mapping.get(l, "Identity"))
            ]
            new_kernel["source_critical_onnx"] = set(critical_onnx)
            new_kernel["profile"] = profile.get(kernel["Name"], {})
            kernelinfo.append(new_kernel)

            new_kernel["is_critical_kernel"] = cls.is_critical_kernel(new_kernel)
        return kernelinfo

def generate_kernel_match_report(layer_name_type_mapping, name_a, layers_a, profile_a, name_b, layers_b, profile_b, saved_json):
    try:
        kcr = KernelCompareReport()
        report = kcr.create_match_report(
            layer_name_type_mapping=layer_name_type_mapping,
            name_a=name_a,
            graph_a=layers_a,
            profile_a=profile_a,
            name_b=name_b,
            graph_b=layers_b,
            profile_b=profile_b,
        )
        saved_json["kernel_match"] = report
        return True
    except Exception as e:
        print(traceback.format_exc())
    return False
