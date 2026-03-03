import onnx
import onnx_graphsurgeon as gs
import numpy as np


def insert_qdq_before_tensor(graph, tensor, scale=None, scale_dtype=np.float32):
    output_nodes = [node for node in tensor.outputs]
    
    tensor_mid = gs.Variable(name=f"{tensor.name}_q", dtype=np.int8)
    Q_scale = gs.Constant(name=f"{tensor.name}_qscale", values=np.array(scale, dtype=scale_dtype))
    DQ_scale = gs.Constant(name=f"{tensor.name}_dqscale", values=np.array(scale, dtype=scale_dtype))
    Q_zero_point = gs.Constant(name=f"{tensor.name}_qzero", values=np.array(0.0, dtype=np.int8))
    DQ_zero_point = gs.Constant(name=f"{tensor.name}_dqzero", values=np.array(0.0, dtype=np.int8))
    tensor_out = gs.Variable(name=f"{tensor.name}_qdq", dtype=tensor.dtype)
    graph.layer(
        op="QuantizeLinear", 
        name=f"{tensor.name}_QuantizeLinear",
        inputs=[tensor, Q_scale, Q_zero_point], 
        outputs=[tensor_mid], 
    )
    graph.layer(
        op="DequantizeLinear", 
        name=f"{tensor.name}_DequantizeLinear",
        inputs=[tensor_mid, DQ_scale, DQ_zero_point], 
        outputs=[tensor_out], 
    )
    for node in output_nodes:
        idx = node.inputs.index(tensor)
        node.inputs[idx] = tensor_out

def remove_unnecessary_add_qdq(graph):
    pass



