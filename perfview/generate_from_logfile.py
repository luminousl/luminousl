import numpy as np
import re
import os
import json

def parse_layer_info(input_str):
    # 初始化结果字典
    result = {
        "Metadata": "",
        "Name": "",
        "LayerType": "",
        "Inputs": [],
        "Outputs": [],
        "TacticName": "",
        "StreamId": -1,
    }
    
    # 提取 Name
    name_match = re.search(r'Name: (.*), LayerType:', input_str)
    if name_match:
        result['Name'] = name_match.group(1)
    
    # 提取 LayerType
    layer_type_match = re.search(r'LayerType: (.*?),', input_str)
    if layer_type_match:
        result['LayerType'] = layer_type_match.group(1)
    
    # 提取 Inputs
    inputs_match = re.search(r'Inputs: \[(.*)\], Outputs:', input_str)
    if inputs_match:
        inputs_str = inputs_match.group(1)
        inputs = []
        # 使用正则表达式匹配每个输入对象
        input_pattern = r'\{ Name: (.*?), Dimensions: \[(.*?)\], Format/Datatype: (.*?) \}'
        for match in re.finditer(input_pattern, inputs_str):
            input_obj = {
                'Name': match.group(1),
                'Dimensions': [int(x) if x.strip().isdigit() else x.strip() for x in match.group(2).split(',')],
                'Format/Datatype': match.group(3)
            }
            inputs.append(input_obj)
        result['Inputs'] = inputs
    
    # 提取 Outputs
    outputs_match = re.search(r'Outputs: \[(.*?)\], TacticName:', input_str)
    if outputs_match:
        outputs_str = outputs_match.group(1)
        outputs = []
        # 使用正则表达式匹配每个输出对象
        output_pattern = r'\{ Name: (.*?), Dimensions: \[(.*?)\], Format/Datatype: (.*?) \}'
        for match in re.finditer(output_pattern, outputs_str):
            output_obj = {
                'Name': match.group(1),
                'Dimensions': [int(x) if x.strip().isdigit() else x.strip() for x in match.group(2).split(',')],
                'Format/Datatype': match.group(3)
            }
            outputs.append(output_obj)
        result['Outputs'] = outputs
    
    # 提取 TacticName
    tactic_match = re.search(r'TacticName: (.*?),', input_str)
    if tactic_match:
        result['TacticName'] = tactic_match.group(1)
    
    # 提取 StreamId
    stream_match = re.search(r'StreamId: (\d+)', input_str)
    if stream_match:
        result['StreamId'] = int(stream_match.group(1))
    
    # 提取 Metadata
    metadata_match = re.search(r'Metadata: (.*?)$', input_str)
    if metadata_match:
        result['Metadata'] = metadata_match.group(1)
    return result

def parse_layer_info_and_performance_and_model(file, projection_code=None):
    with open(file, "r") as f:
        data = f.read()

    onnx_path = None
    if projection_code is not None:
        p = data.find(" === Model Options ===")
        if p != -1:
            q = data.find(" Model: ", p + len(" === Model Options ==="))
            if q != -1:
                e = data.find("\n", q + len(" Model: "))
                if e != -1:
                    onnx_path = data[q + len(" Model: "):e].strip()
    
    if onnx_path is not None and not os.path.exists(onnx_path) and projection_code is not None and projection_code != "":
        try:
            code_lines = "\n".join(["\t" + line for line in projection_code.strip().split("\n")])
            code = "def project_onnx_path(path):\n" + code_lines
            defines = {}
            exec(code, {}, defines)
            projected_onnx_path = defines["project_onnx_path"](onnx_path)
            if projected_onnx_path is not None and os.path.exists(projected_onnx_path):
                onnx_path = projected_onnx_path
        except Exception as e:
            raise ValueError("Failed to run projection code, error: " + str(e))
    if onnx_path is not None and not os.path.exists(onnx_path):
        raise ValueError("Can not found the onnx model from path: " + onnx_path)

    layers = re.findall(r"(?:Layers:|Bindings:)([\s\S]*?)(?=\n\n|\n\[)", data)
    if len(layers) < 2:
        raise ValueError("Invalid log file, layers and bindings are not found")

    layers_info = layers[0]
    bindings = layers[1].strip().split("\n")
    layers_info_list = []
    for layer in layers_info.strip().split("\n"):
        layers_info_list.append(parse_layer_info(layer))
    
    performance_list = None
    perfs = re.findall(r"\[\d{2}/\d{2}/\d{4}-\d{2}:\d{2}:\d{2}\] \[I\] === Profile \(\d+ iterations \) ===[\s\S]*?(?=\[\d{2}/\d{2}/\d{4}-\d{2}:\d{2}:\d{2}\] \[I\] \n)", data)
    if len(perfs) == 1:
        performance_array = re.findall(r"\[\d{2}/\d{2}/\d{4}-\d{2}:\d{2}:\d{2}\] \[I\]\s+(\d+\.\d+)\s+([\d\.]+)\s+([\d\.]+)\s+([\d\.]+)\s+(.*)", perfs[0])
        if len(performance_array) > 0 and performance_array[-1][4] == "Total":
            performance_array.pop()
        
        if len(performance_array) > 0:
            performance_list = [{"count": len(performance_array)}] + [{
                "name": item[4],
                "timeMs": float(item[0]),
                "averageMs": float(item[1]),
                "medianMs": float(item[2]),
                "percentage": float(item[3])
            } for item in performance_array]
    return dict(Layers=layers_info_list, Bindings=bindings), performance_list, onnx_path

if __name__ == "__main__":
    layers, performance, onnx_path = parse_layer_info_and_performance_and_model("aaa.txt")
    print(json.dumps(layers, indent=4))
    # print(performance)
    # print(onnx_path)