import onnx
import onnx_graphsurgeon as gs
import numpy as np

def onnx_generate_example1(onnx_path):
    tensor1 = gs.Variable(name="inp",dtype=np.float32, shape=[3,3,3])
    tensor2 = gs.Variable(name="out",dtype=np.float32, shape=[3,3,3])
    node1= gs.Node(op="Inverse", name="Inverse", inputs=[tensor1], outputs=[tensor2], domain='ai.onnx.contrib')

    graph = gs.Graph(
        nodes=[node1], inputs=[tensor1], outputs=[tensor2],
    )
    sub_model = gs.export_onnx(graph,ir_version=9)
    onnx.save(sub_model, onnx_path)
    print("=== onnx exported to: ", onnx_path)
    print(f"节点数量: {len(sub_model.graph.node)} 输入数量:{len(graph.inputs)} 输出数量:{len(graph.outputs)}")   


if __name__ == "__main__":
    onnx_generate_example1("../onnx_sample/inverse_plugin.onnx")