import pandas as pd
import json

print(json.loads(pd.read_csv("/datav/jingweid/onnx_playground/XiaoMi-4957859-A-weakly_int8-dense.onnx.multistreams.plan.csv").to_json(orient="split")))