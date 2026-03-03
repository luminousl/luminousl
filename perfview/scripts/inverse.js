import { Console } from "console";
import fs from "fs";

class Tensor{
    constructor(name, type, shape, dtype, value){
        this.name = name;
        this.type = type;
        this.shape = shape;
        this.dtype = dtype;
        this.value = value;
        this.parent = null;
        this.uses   = new Set();
    }
};

class Node{
    constructor(name, inputs, outputs, optype, idd, domain, attrs){
        this.name    = name;
        this.inputs  = inputs;
        this.outputs = outputs;
        this.attrs   = attrs;
        this.optype  = optype;
        this.idd     = idd;
        this.domain  = domain;
    }
};

class DAG{
    build(graph){
        this.initializer_mapping = {};
        this.tensor_info_mapping = {};
        this.inputs  = [];
        this.outputs = [];
        this.nodes   = [];
        this.tensors = [];
        this.tensor_mapping = {};

        for(const init of graph.initializer)
            this.initializer_mapping[init.name] = init;

        for(const tensor of graph.tensor_info)
            this.tensor_info_mapping[tensor.name] = tensor;

        const get_tensor_or_create = (name, type, shape, dtype, value)=>{
            if(name in this.tensor_mapping){
                return this.tensor_mapping[name];
            }
            const tensor = new Tensor(name, type, shape, dtype, value);
            this.tensor_mapping[name] = tensor;
            return tensor;
        };

        for(const inp of graph.input)
            this.inputs.push(get_tensor_or_create(inp.name, "Variable", inp.shape, inp.dtype, null));

        for(const out of graph.output)
            this.outputs.push(get_tensor_or_create(out.name, "Variable", out.shape, out.dtype, null));

        for(const node of graph.node){
            const inputs  = [];
            const outputs = [];
            for(const inp of node.input){
                let inp_tensor = null;
                if(inp in this.initializer_mapping){
                    const data = this.initializer_mapping[inp];
                    inp_tensor = get_tensor_or_create(inp, "Constant", data.shape, data.dtype, data.data_view);
                }else if(inp in this.tensor_info_mapping){
                    const info = this.tensor_info_mapping[inp];
                    inp_tensor = get_tensor_or_create(inp, "Variable", info.shape, info.dtype, null);
                }else if(inp == ""){
                    inp_tensor = get_tensor_or_create(inp, "Constant", [], "None", null);
                }else{
                    inp_tensor = get_tensor_or_create(inp, "Variable", [], "None", null);
                }
                inputs.push(inp_tensor);
                this.tensors.push(inp_tensor);
            }

            for(const out of node.output){
                let out_tensor = null;
                if(out in this.tensor_info_mapping){
                    const info = this.tensor_info_mapping[out];
                    out_tensor = get_tensor_or_create(out, "Variable", info.shape, info.dtype, null);
                }else{
                    out_tensor = get_tensor_or_create(out, "Variable", [], "None", null);
                }
                outputs.push(out_tensor);
                this.tensors.push(out_tensor);
            }

            const node_new_represent = new Node(node.name, inputs, outputs, node.optype, node.idd, node.domain, node.attrs);
            for(const tensor of inputs)
                tensor.uses.add(node_new_represent);
            
            for(const tensor of outputs)
                tensor.parent = node_new_represent;

            this.nodes.push(node_new_represent);
        }
        this.topsort();
    }

    topsort(){
        const sorted_nodes = [];
        const tensor_ready_mapping = {};
        const node_ready_mapping   = {};
        const inputs = [];

        const tensor_ready = (tensor)=>{
            if(!(tensor.name in tensor_ready_mapping)){
                tensor_ready_mapping[tensor.name] = {
                    tensor: tensor,
                    ready: false
                };
            }

            if(tensor_ready_mapping[tensor.name].ready)
                return;

            if(tensor.parent == null){
                tensor_ready_mapping[tensor.name].ready = true;

                if(tensor.type == "Variable"){
                    inputs.push(tensor);
                }
                return;
            }

            if(tensor.parent.idd in node_ready_mapping)
                return;
            
            node_ready_mapping[tensor.parent.idd] = true;
            for(const input_tensor of tensor.parent.inputs)
                tensor_ready(input_tensor);

            tensor_ready_mapping[tensor.name].ready = true;
            sorted_nodes.push(tensor.parent);
        };

        for(const tensor of this.outputs)
            tensor_ready(tensor);

        if(this.inputs.length != inputs.length){
            console.log("Mismatched graph input tensors, old " + this.inputs.length + ", actual " + inputs.length);
        }

        const input_indice_pairs = inputs.map((x)=>{return [x, this.inputs.indexOf(x)];});
        this.inputs = input_indice_pairs.sort((a, b)=>{return a[1] - b[1];}).map((x)=>{return x[0];});
        if(this.nodes.length != sorted_nodes.length){
            console.log("Mismatched graph nodes, old " + this.nodes.length + ", actual " + sorted_nodes.length);
        }
        this.nodes = sorted_nodes;
    }

    generate_node_init_arguments(node){
        if(node.attrs.length == 0) return "";
        if(node.optype == "Conv"){
            const attrs = {};
            for(const attr of node.attrs)
                attrs[attr.name] = attr;

            const argument = [];
            if(attrs.kernel_shape)
                argument.push("kernel=" + attrs.kernel_shape.value[0]);
            if(attrs.pads)
                argument.push("padding=" + attrs.pads.value[0]);
            if(attrs.strides)
                argument.push("stride=" + attrs.strides.value[0]);
            if(attrs.dilations)
                argument.push("dilation=" + attrs.dilations.value[0]);
            if(attrs.group)
                argument.push("group=" + attrs.group.value);
            return argument.join(", ");
        }

        return node.attrs.map((x)=>{return x.name + "=" + (x.dtype.indexOf("array") != -1 ? "[" + x.value.join(", ") + "]" : x.value);}).join(", ");
    }

    generate_node_init_codeline(node, op_index_mapping, node_desc_mapping){
        let name = node.optype.toLowerCase();
        const shortnames = {
            "BatchNormalization": "bn",
            "QuantizeLinear": "q",
            "DequantizeLinear": "dq",
        };
        if(node.optype in shortnames){
            name = shortnames[node.optype];
        }
        if(!(name in op_index_mapping))
            op_index_mapping[name] = {index: 0};

        op_index_mapping[name].index += 1;
        const index = op_index_mapping[name].index;
        const code_line = "this." + name + index + " = new nn." + node.optype + "(" + this.generate_node_init_arguments(node) + ");";
        node_desc_mapping[node.idd] = {
            var_name: name + index
        };
        return code_line;
    }

    get_tensor_variable_desc(tensor, variables_desc_mapping, prefix="x"){
        if(!(prefix in variables_desc_mapping.index_mapping)){
            variables_desc_mapping.index_mapping[prefix] = 0;
        }

        if(!(tensor.name in variables_desc_mapping.tensor_mapping)){
            variables_desc_mapping.index_mapping[prefix] += 1;
            const index = variables_desc_mapping.index_mapping[prefix];
            variables_desc_mapping.tensor_mapping[tensor.name] = {
                tensor: tensor,
                var_name: prefix + index,
                var_index: index,
                defined: false
            }
        }
        return variables_desc_mapping.tensor_mapping[tensor.name];
    }

    generate_node_forwared_codeline(node, node_desc_mapping, variables_desc_mapping){
        const desc = node_desc_mapping[node.idd];
        const inputs  = node.inputs.map((x)=>{return this.get_tensor_variable_desc(x, variables_desc_mapping, x.type == "Variable" ? "x" : "c");});
        const outputs = node.outputs.map((x)=>{return this.get_tensor_variable_desc(x, variables_desc_mapping);});
        const input_arguments = inputs.map((x)=>{return x.tensor.type == "Constant" ? "this." + x.var_name : x.var_name;}).join(", ");
        let output_arguments  = outputs.map((x)=>{return x.var_name;}).join(", ");

        if(outputs.length > 1)
            output_arguments = "[" + output_arguments + "]";

        return output_arguments + " = this." + desc.var_name + "(" + input_arguments + ");";
    }

    generate_load_weight_codeline(node, variables_desc_mapping){
        const inputs = node.inputs.filter((x)=>{return x.type == "Constant";}).map((x)=>{return this.get_tensor_variable_desc(x, variables_desc_mapping, "c");});
        if(inputs.length == 0) return [];
        const inputs_undefined = node.inputs.filter((x)=>{return x.type == "Constant" && !variables_desc_mapping.tensor_mapping[x.name].defined;}).map((x)=>{return this.get_tensor_variable_desc(x, variables_desc_mapping, "c");});
        for(const tensor of node.inputs){
            variables_desc_mapping.tensor_mapping[tensor.name].defined = true;
        }
        return inputs_undefined.map((x)=>{return "this." + x.var_name + " = nn.load(\"" + x.tensor.name + "\");";});
    }

    generate_code(){
        const code_lines = [];
        const init_part = [];
        const forward_part = [];
        const op_index_mapping = {};
        const node_desc_mapping = {};
        const variables_desc_mapping = {tensor_mapping: {}, index_mapping: {}};
        const load_weight_part = [];
        const load_weight_define = [];

        for(const node of this.nodes)
            init_part.push(this.generate_node_init_codeline(node, op_index_mapping, node_desc_mapping));

        let input_descs = this.inputs.map((x)=>{return this.get_tensor_variable_desc(x, variables_desc_mapping, "input");})
        let output_descs = this.outputs.map((x)=>{return this.get_tensor_variable_desc(x, variables_desc_mapping, "output");})
        for(const node of this.nodes){
            forward_part.push(this.generate_node_forwared_codeline(node, node_desc_mapping, variables_desc_mapping));
        }

        for(const node of this.nodes){
            const codeline = this.generate_load_weight_codeline(node, variables_desc_mapping);
            load_weight_define.push(...codeline);
        }

        const input_arguments = input_descs.map((x)=>{return x.var_name;}).join(", ");
        let output_arguments = output_descs.map((x)=>{return x.var_name;}).join(", ");
        if(output_descs.length > 1)
            output_arguments = "[" + output_arguments + "]";

        code_lines.push("class Model{");
        code_lines.push("  constructor(){");
            code_lines.push(...init_part.map((x)=>{return "    " + x;}));
        code_lines.push("  }");
        code_lines.push("");
        code_lines.push("  init_weights(){");
            code_lines.push(...load_weight_define.map((x)=>{return "    " + x;}));
            code_lines.push(...load_weight_part.map((x)=>{return "    " + x;}));
        code_lines.push("  }");
        code_lines.push("");
        code_lines.push("  forward(" + input_arguments + "){");
            code_lines.push(...forward_part.map((x)=>{return "    " + x;}));
        code_lines.push("    return " + output_arguments + ";");
        code_lines.push("  }");
        code_lines.push("}");
        console.log(code_lines.join("\n"));
    }
};

let graph = JSON.parse(fs.readFileSync("static/02_vision_2nd-sg12.json"));
const dag = new DAG(graph);
dag.build(graph);
dag.generate_code();