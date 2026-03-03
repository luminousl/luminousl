import * as dagre from "/static/graph.js"

const meta_data_file = "/static/onnx-metadata.json";
const node_color_formater = {
    Conv: {fill: "rgb(51, 85, 136)"},
    Relu: {fill: "rgb(112, 41, 33)"},
    Sigmoid: {fill: "rgb(112, 41, 33)"},
    Softmax: {fill: "rgb(112, 41, 33)"},
    BatchNormalization: {fill: "rgb(51, 85, 68)"},
    LayerNormalization: {fill: "rgb(51, 85, 68)"},
    Gather: {fill: "rgb(51, 85, 68)"},
    MaxPool: {fill: "rgb(51, 85, 68)"},
    GlobalAveragePool: {fill: "rgb(51, 85, 68)"},
    Transpose: {fill: "rgb(51, 85, 68)"},
    ConvTranspose: {fill: "rgb(51, 85, 136)"},
    Gemm: {fill: "rgb(51, 85, 68)"},
    MatMul: {fill: "rgb(112, 41, 33)"},
    Flatten: {fill: "rgb(108, 79, 71)"},
    Reshape: {fill: "rgb(108, 79, 71)"},
};

const onnxdtype_to_string = {
    1: "float32",
    2: "uint8",
    3: "int8",
    4: "uint16",
    5: "int16",
    6: "int32",
    7: "int64",
    8: "string",
    9: "bool",
    10: "float16",
    11: "double",
    12: "uint32",
    13: "uint64",
    14: "complex64",
    15: "complex128",
    16: "bfloat16",
    17: "float8e4m3fn",
    18: "float8e4m3fnuz",
    19: "float8e5m2",
    20: "float8e5m2fnuz",
    21: "uint4",
    22: "int4"
};

class AdditionUI extends Vue{
    constructor(app, ui_element_id){
        super({
        el: "#" + ui_element_id,
        data: {
        },
        mounted(){
        },
        methods: {
            
        }
        });
        this.app = app;
    }

    initialize(){
    }
};

class ONNXRender{
    measure_node_size(name, isIOTensor){
        const title = document.createElementNS("http://www.w3.org/2000/svg", "text");
        title.innerHTML = name;
        title.setAttribute("x", 5);
        title.setAttribute("y", 19);
        title.classList.add("g-node-title");
        this.screen.appendChild(title);

        const box = title.getBBox();
        this.screen.removeChild(title);
        box.width  += 12;
        box.height += 8;
        return box;
    }

    draw_link(edge){
        const px = this.size_config.content_padding_width;
        const py = this.size_config.content_padding_height;
        const points = edge.points;
        const line = document.createElementNS("http://www.w3.org/2000/svg", "path");
        let line_argument = "M" + (points[0].x + px) + " " + (points[0].y + py);
        let i = 0
        for(; i + 3 <= points.length; i += 3){
            const a = points[i + 0];
            const b = points[i + 1];
            const c = points[i + 2];
            line_argument += " C" + (a.x + px) + " " + (a.y + py) + " " + (b.x + px) + " " + (b.y + py) + " " + (c.x + px) + " " + (c.y + py);
        }

        for(; i < points.length; ++i){
            const a = points[i];
            line_argument += " L" + (a.x + px) + " " + (a.y + py);
        }
        line.setAttribute("d", line_argument);
        line.setAttribute("marker-end", "url(#arrow)");
        line.classList.add("g-node-link");
        edge.label_text_element = null;
        edge.line_element = line;

        if(edge.name_show){
            const text = document.createElementNS("http://www.w3.org/2000/svg", "text");
            text.setAttribute("transform", "translate(" + (edge.x + px) + "," + (edge.y + py - 10) + ")");
            text.classList.add("g-node-link-shape");
            text.innerHTML = "<tspan xml:space=\"preserve\" dy=\"1em\" x=\"1\">" + edge.name_show + "</tspan>";
            edge.label_text_element = text;
            this.screen.insertBefore(text, this.screen.firstChild);
        }
        this.screen.insertBefore(line, this.screen.firstChild);
    }

    draw_node(node){
        const raw = node.raw;
        const title = document.createElementNS("http://www.w3.org/2000/svg", "text");
        const isIOTensor = node.type == "input" || node.type == "output";
        const className  = isIOTensor ? "io-tensor-node" : "operator-node";
        title.innerHTML = isIOTensor ? raw.name : raw.optype;
        title.setAttribute("x", 5);
        title.setAttribute("y", 19);
        title.classList.add("g-node-title");

        let left = node.x - node.width  * 0.5;
        let top  = node.y - node.height * 0.5 + 1;

        const body = document.createElementNS("http://www.w3.org/2000/svg", "g");
        body.setAttribute("transform", "translate(" + left + "," + top + ")");
        body.classList.add("g-node");
        body.classList.add(className);
        const background = document.createElementNS("http://www.w3.org/2000/svg", "rect");
        background.setAttribute("x", 0);
        background.setAttribute("y", 0);
        background.setAttribute("rx", 5);
        background.classList.add("g-node-background");
        background.setAttribute("width",  node.width);
        background.setAttribute("height", node.height);

        if(!isIOTensor){
            background.style["fill"] = raw.optype in node_color_formater ? node_color_formater[raw.optype].fill : "#000";
        }

        node.element = body;
        body.appendChild(background);
        body.appendChild(title);
        this.screen.appendChild(body);

        body.addEventListener("pointerdown", (e)=>{
            e.preventDefault();
            e.stopPropagation();

            const clear = !(e.metaKey || e.ctrlKey);
            this.select_nodes({clear: clear, current_node_id:node.raw.idd}, node.raw.idd);
        });
    }

    format_shape(tensor_info, io, type="shape_right"){
        if(tensor_info == null || tensor_info.shape == null){
            return null;
        }

        if(type == "shape_right"){
            return (tensor_info.dtype ? "[" + tensor_info.dtype + "] " : "") + tensor_info.shape.join("x");
        }else if(type == "shape_left"){
            return tensor_info.shape.join("x") + (tensor_info.dtype ? " [" + tensor_info.dtype + "]" : "");
        }else if(type == "shape_only"){
            return tensor_info.shape.join("x");
        }
    }

    convert_to_display_string(attr, node){
        if(node.optype == "Cast"){
            if(attr.name == "to"){
                if(attr.value in onnxdtype_to_string)
                    return onnxdtype_to_string[attr.value];
            }
        }
        if(attr.value instanceof String){
            return attr.value;
        }else if(attr.value instanceof Array){
            return attr.value.join(", ");
        }else{
            return attr.value + "";
        }
    }

    network_layout(graph, tensor_info){
        const nodes_with_addition_info = [];
        const nodes_by_input  = {};
        const nodes_by_output = {};
        const edges = [];
        const node_mapping = {};
    
        for(const input of graph.input){
            const box = this.measure_node_size(input.name, true);
            const instance = {
                width: box.width,
                height: box.height,
                v: input.idd + "",
                parent: null,
                type: "input",
                raw: {
                    optype: "Input",
                    name: input.name,
                    input: [],
                    output: [input.name],
                    idd: input.idd,
                    dtype: input.dtype,
                    shape: input.shape,
                    attrs: [{
                            dtype: "int32",
                            name: "dtype",
                            value: input.dtype,
                            value_show: input.dtype
                        },
                        {
                            dtype: "int32_array",
                            name: "shape",
                            value: input.shape,
                            value_show: "[" + input.shape.join(", ") + "]"
                        },
                    ],
                },
            };
            nodes_with_addition_info.push(instance);
            nodes_by_output[input.name] = [instance];
        }
    
        for(const node of graph.node){
            const box  = this.measure_node_size(node.optype, true);
            const instance = {
                width: box.width,
                height: box.height,
                v: node.idd + "",
                parent: null,
                type: "operation",
                raw: node
            };
            nodes_with_addition_info.push(instance);
    
            for(const tensor of node.input){
                if(tensor == "") continue;
                if(!(tensor in nodes_by_input)){
                    nodes_by_input[tensor] = [];
                }
                nodes_by_input[tensor].push(instance);
            }
    
            for(const tensor of node.output){
                if(tensor == "") continue;
                if(!(tensor in nodes_by_output)){
                    nodes_by_output[tensor] = [];
                }
                nodes_by_output[tensor].push(instance);
            }
        }
    
        for(const output of graph.output){
            const box = this.measure_node_size(output.name, true);
            const instance = {
                width: box.width,
                height: box.height,
                v: output.idd + "",
                parent: null,
                type: "output",
                raw: {
                    optype: "Output",
                    name: output.name,
                    input: [output.name],
                    domain: "",
                    dtype: output.dtype,
                    shape: output.shape,
                    attrs: [{
                            dtype: "int32",
                            name: "dtype",
                            value: output.dtype,
                            value_show: output.dtype
                        },
                        {
                            dtype: "int32_array",
                            name: "shape",
                            value: output.shape,
                            value_show: "[" + output.shape.join(", ") + "]"
                        },
                    ],
                    output: [],
                    idd: output.idd
                },
            };
            nodes_with_addition_info.push(instance);
            if(!(output.name in nodes_by_input)){
                nodes_by_input[output.name] = [];
            }
            nodes_by_input[output.name].push(instance);
        }

        for(const node of nodes_with_addition_info){
            node.inputs = [];
            node.outputs = [];
            node_mapping[node.raw.idd] = node;
        }
    
        for(const node of nodes_with_addition_info){
            let edge_ins, edge_outs;
            if(node.type == "input"){
                edge_ins = [node];
                edge_outs = [];
                for(const item of nodes_by_input[node.raw.name]){
                    edge_outs.push({node: item, name: node.raw.name, name_show: this.format_shape(node.raw, null, "shape_left")});
                }
            }else if(node.type == "output"){
                edge_ins = nodes_by_output[node.raw.name];
                edge_outs = [{node: node, name: node.raw.name, name_show: this.format_shape(node.raw, null, "shape_left")}];
            }else if(node.type == "operation"){
                edge_ins = [node];
                edge_outs = [];
                for(const out of node.raw.output){
                    if(!(out in nodes_by_input)) continue;
                    for(const item of nodes_by_input[out]){
                        edge_outs.push({node: item, name: out, name_show: this.format_shape(tensor_info[out], null, "shape_only")});
                    }
                }
            }
            
            for(const i of edge_ins){
                for(const o of edge_outs){
                    const edge = {
                        v: i.v,       // from, input
                        w: o.node.v,  // to,   output
                        labeloffset: 15,
                        labelpos: "r",
                        minlen: 1,
                        weight: 1,
                        width: 1,
                        height: 1,
                        name: o.name,
                        name_show: o.name_show
                    };
                    edges.push(edge);
                    i.outputs.push(edge);
                    o.node.inputs.push(edge);
                }
            }
        }

        if(graph.layout){
            for(let i = 0; i < graph.layout.nodes.length; ++i){
                const a = graph.layout.nodes[i];
                const b = nodes_with_addition_info[i];
                b.x = a.x;
                b.y = a.y;
            }
            
            for(let i = 0; i < graph.layout.edges.length; ++i){
                const a = graph.layout.edges[i];
                const b = edges[i];
                b.x = a.x;
                b.y = a.y;
                b.points = a.points;
            }
        }else{
            dagre.layout(nodes_with_addition_info, edges, {nodesep: 50, ranksep: 20}, {});
        }

        for(const node of nodes_with_addition_info){
            delete node.v;
            delete node.parent;
        }

        for(const edge of edges){
            edge.from = node_mapping[parseInt(edge.v)];
            edge.to   = node_mapping[parseInt(edge.w)];
            delete edge.v;
            delete edge.w;
            delete edge.weight;
            delete edge.width;
            delete edge.minlen;
            delete edge.labelpos;
            delete edge.labeloffset;
            delete edge.height;
        }
        return {nodes: nodes_with_addition_info, edges: edges, node_mapping: node_mapping, nodes_by_output: nodes_by_output, nodes_by_input: nodes_by_input};
    }

    render_graph(graph){
        this.inputs = [];
        this.outputs = [];
        this.initializer_by_name = {};
        this.selected_nodes = [];
        this.selected_nodes_current = null;
        this.current_preview_nodes = [];

        this.graph = graph;
        this.tensor_info_map = {};
        this.node_mapping = {};
        this.edge_mapping = {};
        for(const init of graph.initializer)
            this.initializer_by_name[init.name] = init;

        for(const info of graph.tensor_info)
            this.tensor_info_map[info.name] = info;

        for(const inp of graph.input)
            this.tensor_info_map[inp.name] = inp;

        for(const out of graph.output)
            this.tensor_info_map[out.name] = out;

        for(const node of graph.node){
            for(const attr of node.attrs){
                attr.value_show = this.convert_to_display_string(attr, node);
            }
        }
        const layout_info = this.network_layout(graph, this.tensor_info_map);
        this.nodes = layout_info.nodes;
        this.edges = layout_info.edges;
        this.node_mapping = layout_info.node_mapping;
        this.node_name_lookup_table = {};
        this.nodes_by_output = layout_info.nodes_by_output;
        this.nodes_by_input  = layout_info.nodes_by_input;

        if(graph.kernel_match){
            const list_groups = [];
            const group_indexs = Object.keys(graph.kernel_match);
            let max_group_index = 0;
            for(let i = 0; i < group_indexs.length; ++i){
                max_group_index = Math.max(max_group_index, group_indexs[i]);
            }
            for(let i = 0; i < max_group_index + 1; ++i){
                if(!(i in graph.kernel_match)) continue;
                const group = graph.kernel_match[i];
                group.index = list_groups.length;
                list_groups.push(group);
            }
            graph.kernel_match = list_groups;
        }

        // set for profile data.
        for(const node of this.nodes){
            node.raw.profiles = [];
            node.raw.profiles_compared = [];
        }

        let max_x = 0, max_y = 0;
        for(const node of this.nodes){
            max_x = Math.max(max_x, node.x + node.width * 0.5);
            max_y = Math.max(max_y, node.y + node.height * 0.5);
            if(node.raw.name && node.raw.name != "")
                this.node_name_lookup_table[node.raw.name] = node;
        }

        if(graph.layerinfo){
            for(const layer of graph.layerinfo.Layers){
                for(const name of layer.ONNXNames){
                    if(name in this.node_name_lookup_table){
                        const node = this.node_name_lookup_table[name];
                        if(graph.profile && (layer.Name in graph.profile)){
                            layer.Profile = graph.profile[layer.Name];
                            layer.Latency = layer.Profile.averageMs;
                        }else{
                            layer.Latency = 0;
                        }
                        node.raw.profiles.push(layer);
                    }
                }
            }
        }
        
        if(graph.layerinfo_compared){
            for(const layer of graph.layerinfo_compared.Layers){
                for(const name of layer.ONNXNames){
                    if(name in this.node_name_lookup_table){
                        const node = this.node_name_lookup_table[name];
                        if(graph.profile_compared && (layer.Name in graph.profile_compared)){
                            layer.Profile = graph.profile_compared[layer.Name];
                            layer.Latency = layer.Profile.averageMs;
                        }else{
                            layer.Latency = 0;
                        }
                        node.raw.profiles_compared.push(layer);
                    }
                }
            }
        }

        this.size_config = {
            content_padding_width: 300,
            content_padding_height: 100,
            content_width: max_x + 600,
            content_height: max_y + 200,
        }

        for(const node of this.nodes){
            node.x += this.size_config.content_padding_width;
            node.y += this.size_config.content_padding_height;
        }

        for(const inp of graph.input){
            this.inputs.push(this.node_mapping[inp.idd]);
        }

        for(const out of graph.output){
            this.outputs.push(this.node_mapping[out.idd]);
        }

        this.screen.setAttribute("width", this.size_config.content_width);
        this.screen.setAttribute("height", this.size_config.content_height);
        this.screen.setAttribute("viewBox", "0 0 " + this.size_config.content_width + " " + this.size_config.content_height);

        for(const node of this.nodes)
            this.draw_node(node);

        for(const edge of this.edges)
            this.draw_link(edge);
    }

    preview_nodes(...node_ids){
        for(const node of this.selected_nodes)
            node.element.classList.remove("g-node-focused-highlight");
        
        for(const node of this.current_preview_nodes)
            node.element.classList.remove("g-node-focused-highlight");

        this.current_preview_nodes = [];
        for(let i = 0; i < node_ids.length; ++i){
            const node_id = node_ids[i];
            if(!(node_id in this.node_mapping)) continue;
            const node = this.node_mapping[node_id];
            node.element.classList.add("g-node-focused-highlight");
            this.current_preview_nodes.push(node);
        }
    }

    automatic_complete_select_nodes() {
        var node_input_dependent = new Set();
        var nodes_inst = {};
        var tensor_producer = {};
        var tensor_consumers = {};
        const nodes = this.nodes;
        for (const n of nodes) {
            for (const a of n.raw.input) {
                if (!(a in tensor_consumers)) {
                    tensor_consumers[a] = [];
                }
                tensor_consumers[a].push(n);
            }
            for (const a of n.raw.output) {
                tensor_producer[a] = n;
            }
        }
        
        for(const n of this.selected_nodes) {
            nodes_inst[n.raw.idd] = n;
        }

        var nodes_visited = new Set();
        var nodes_fifo = [];
        for (const v of this.inputs) {
            if (v.raw.name in tensor_consumers) {
                for (const nn of tensor_consumers[v.raw.name]) {
                    if (!nodes_visited.has(nn.raw.idd)) {
                        nodes_fifo.push(nn);
                    }
                }
            }
        }
        while (nodes_fifo.length) {
            var n = nodes_fifo.shift(0);
            node_input_dependent.add(n.raw.idd);
            if (!nodes_visited.has(n.raw.idd)) {
                nodes_visited.add(n.raw.idd);
                for (const a of n.raw.output){
                    if (a in tensor_consumers) {
                        for (const nn of tensor_consumers[a]) {
                            if (!nodes_visited.has(nn.raw.idd)) {
                                nodes_fifo.push(nn);
                            }
                        }
                    }
                }
            }
        }
    
        const PREDECESSOR = 1;
        const SUCCESSOR = 2;
        const CONSTANT = 4;
    
        var node_st = {};
    
        // search_down
        nodes_visited = new Set();
        nodes_fifo = Object.values(nodes_inst);
        while (nodes_fifo.length) {
            var n = nodes_fifo.shift(0);
            if (!nodes_visited.has(n.raw.idd)) {
                nodes_visited.add(n.raw.idd);
                node_st[n.raw.idd] = ((node_st[n.raw.idd] || 0) | SUCCESSOR);
                for (const a of n.raw.output){
                    if (a in tensor_consumers) {
                        for (const nn of tensor_consumers[a]) {
                            if (!nodes_visited.has(nn.raw.idd)) {
                                nodes_fifo.push(nn);
                            }
                        }
                    }
                }
            }
        }
    
        // search_up_constant
        nodes_visited = new Set();
        nodes_fifo = Object.values(nodes_inst);
        while (nodes_fifo.length) {
            var n = nodes_fifo.shift(0);
            if (!nodes_visited.has(n.raw.idd)) {
                nodes_visited.add(n.raw.idd);
                node_st[n.raw.idd] = ((node_st[n.raw.idd] || 0) | PREDECESSOR);
                if (!node_input_dependent.has(n.raw.idd)) {
                    node_st[n.raw.idd] = ((node_st[n.raw.idd] || 0) | CONSTANT);
                }
                for (const a of n.raw.input){
                    if (a in tensor_producer) {
                        const nn = tensor_producer[a];
                        if ((!((node_st[nn.raw.idd] || 0) & SUCCESSOR)) && node_input_dependent.has(nn.raw.idd)) {
                            continue;
                        }
                        if (!nodes_visited.has(nn.raw.idd)) {
                            nodes_fifo.push(nn);
                        }
                    }
                }
            }
        }
    
        var partition_nodes = {};
        for (const n of nodes) {
            const st = (node_st[n.raw.idd] || 0);
            if ((st & PREDECESSOR) && ((st & SUCCESSOR) || (st & CONSTANT))) {
                partition_nodes[n.raw.idd] = n;
            }
        }
        this.select_nodes({clear: true}, ...Object.values(partition_nodes).map((a)=>{return a.raw.idd;}));
    }

    select_nodes(option, ...node_ids){
        for(const node of this.selected_nodes){
            node.element.classList.remove("g-node-focused");
            node.element.classList.remove("g-node-focused-highlight");
            node.element.classList.remove("g-node-focused-compared");
            node.element.classList.remove("g-node-focused-compared-highlight");
        }

        let focused_class_name = "g-node-focused";
        let focused_current_class_name = "g-node-focused-highlight";
        if(option.render_style == "compared"){
            focused_class_name = "g-node-focused-compared";
            focused_current_class_name = "g-node-focused-compared-highlight";
        }

        for(const node of this.current_preview_nodes)
            node.element.classList.remove(focused_current_class_name);

        if(option.clear){
            this.selected_nodes = [];
        }

        let current_node_id = 0;
        if("current_node_id" in option && option.current_node_id != null){
            current_node_id = option.current_node_id;
        }

        let current_node = null;
        for(let i = 0; i < node_ids.length; ++i){
            const node_id = node_ids[i];
            if(!(node_id in this.node_mapping)) continue;
            const node = this.node_mapping[node_id];
            if(node_id == current_node_id){
                current_node = node;
            }
            if(this.selected_nodes.indexOf(node) != -1) continue;
            this.selected_nodes.push(node);
        }

        this.selected_nodes_current = current_node;
        for(const node of this.selected_nodes){
            if(node == current_node){
                node.element.classList.add(focused_current_class_name);
            }else{
                node.element.classList.add(focused_class_name);
            }
        }

        let selected_node_ids = [];
        let select_nodes_total_latency = 0;
        let select_nodes_total_compared_latency = 0;
        const repeated_mapping = {};
        const repeated_compared_mapping = {};
        for(const node of this.selected_nodes){
            if(node != current_node){
                selected_node_ids.push(node.raw.idd);
            }
            for(const profile of node.raw.profiles){
                if(profile.Name in repeated_mapping) continue;
                repeated_mapping[profile.Name] = true;
                select_nodes_total_latency += profile.Profile ? profile.Profile.averageMs : 0;
            }
            for(const profile of node.raw.profiles_compared){
                if(profile.Name in repeated_compared_mapping) continue;
                repeated_compared_mapping[profile.Name] = true;
                select_nodes_total_compared_latency += profile.Profile ? profile.Profile.averageMs : 0;
            }
        }
    }

    scroll_to(node_id, behavior="instant", pos="top"){
        if(!(node_id in this.node_mapping)) return;
        const node   = this.node_mapping[node_id];
        const topgap = pos == "top" ? 50 : window.innerHeight * 0.25;
        window.scrollTo({
            left: node.x * this.screen_scale - window.innerWidth * 0.5,
            top:  node.y * this.screen_scale - topgap,
            behavior: behavior
        });
    }

    setup_pointer_move_event(){
        let screen_pointer = {down: false};
        this.screen.parentElement.addEventListener("pointerdown", (e)=>{
            if(!(e.button == 0 && e.buttons == 1)) return;

            const mode = e.metaKey || e.ctrlKey ? "select_nodes_by_bbox" : "move_screen";
            e.preventDefault();
            e.stopPropagation();

            const current_select_node_ids = {};
            for(const node of this.selected_nodes)
                current_select_node_ids[node.raw.idd] = true;

            screen_pointer = {
                x: e.x,
                y: e.y,
                scrollx: window.scrollX,
                scrolly: window.scrollY,
                down: true,
                mode: mode,
                floating_nodes: [],
                current_select_node_ids: current_select_node_ids
            };
            if(mode == "select_nodes_by_bbox"){
                this.select_nodes_bbox_element.style["left"] = e.x;
                this.select_nodes_bbox_element.style["top"] = e.y;
                this.select_nodes_bbox_element.style["width"] = 1;
                this.select_nodes_bbox_element.style["height"] = 1;
                this.select_nodes_bbox_element.style["display"] = "unset";
            }
            this.screen.style["cursor"] = mode == "move_screen" ? "move" : "crosshair";
            this.screen.setPointerCapture(e.pointerId);
        });

        this.screen.parentElement.addEventListener("pointermove", (e)=>{
            if(!screen_pointer.down) return;
            e.preventDefault();
            e.stopPropagation();

            if(screen_pointer.mode == "move_screen"){
                window.scrollTo({
                    left: screen_pointer.x - e.x + screen_pointer.scrollx,
                    top: screen_pointer.y - e.y + screen_pointer.scrolly,
                });
            }else if(screen_pointer.mode == "select_nodes_by_bbox"){
                let left   = Math.min(screen_pointer.x, e.x) - 2;
                let top    = Math.min(screen_pointer.y, e.y) - 2;
                let width  = Math.abs(screen_pointer.x - e.x);
                let height = Math.abs(screen_pointer.y - e.y);
                let right  = left + width;
                let bottom = top  + height;
                this.select_nodes_bbox_element.style["left"]   = left;
                this.select_nodes_bbox_element.style["top"]    = top;
                this.select_nodes_bbox_element.style["width"]  = width - 2;
                this.select_nodes_bbox_element.style["height"] = height - 2;

                const sb = this.screen.getBoundingClientRect();
                const pb = this.screen.parentElement.getBoundingClientRect();
                const diff_x = sb.left - pb.left;
                const diff_y = sb.top - pb.top;
                left   = (left + window.scrollX - diff_x) / this.screen_scale;
                top    = (top + window.scrollY - diff_y) / this.screen_scale;
                right  = (right + window.scrollX - diff_x) / this.screen_scale;
                bottom = (bottom + window.scrollY - diff_y) / this.screen_scale;

                for(const node of screen_pointer.floating_nodes){
                    if(node.raw.idd in screen_pointer.current_select_node_ids){
                        const idx = this.selected_nodes.indexOf(node);
                        this.selected_nodes.splice(idx, 1);
                        delete screen_pointer.current_select_node_ids[node.raw.idd];
                    }
                    node.element.classList.remove("g-node-focused");
                    node.element.classList.remove("g-node-focused-highlight");
                    node.element.classList.remove("g-node-focused-compared");
                    node.element.classList.remove("g-node-focused-compared-highlight");
                }

                const select_nodes = [];
                let select_nodes_total_latency = 0;
                let select_nodes_total_compared_latency = 0;
                const repeated_mapping = {};
                const repeated_compared_mapping = {};
                for(const node of this.nodes){
                    const nleft = node.x - node.width * 0.5;
                    const ntop  = node.y - node.height * 0.5;
                    const nright  = nleft + node.width;
                    const nbottom = ntop + node.height;

                    const min_right  = Math.min(right, nright);
                    const min_bottom = Math.min(bottom,  nbottom);
                    const max_left   = Math.max(left, nleft);
                    const max_top    = Math.max(top, ntop);
                    if(min_right >= max_left && min_bottom >= max_top){
                        select_nodes.push(node);
                        node.element.classList.add("g-node-focused");
                        for(const profile of node.raw.profiles){
                            if(profile.Name in repeated_mapping) continue;
                            repeated_mapping[profile.Name] = true;
                            select_nodes_total_latency += profile.Profile ? profile.Profile.averageMs : 0;
                        }
                        for(const profile of node.raw.profiles_compared){
                            if(profile.Name in repeated_compared_mapping) continue;
                            repeated_compared_mapping[profile.Name] = true;
                            select_nodes_total_compared_latency += profile.Profile ? profile.Profile.averageMs : 0;
                        }
                    }
                }
                screen_pointer.floating_nodes = select_nodes;
            }
        });
        this.screen.parentElement.addEventListener("pointerup", (e)=>{
            if(!screen_pointer.down) return;
            e.preventDefault();
            e.stopPropagation();
            if(screen_pointer.mode == "select_nodes_by_bbox"){
                const node_ids = [];
                for(const node of screen_pointer.floating_nodes){
                    node_ids.push(node.raw.idd);
                }
                this.select_nodes({clear: false}, ...node_ids);
                this.select_nodes_bbox_element.style["display"] = "none";
            }
            this.screen.style["cursor"] = "auto";
            this.screen.releasePointerCapture(e.pointerId);
            screen_pointer.down = false;
        });
    }

    setup_global_keys_event(){
        const _this = this;
        const allow_keys = {
           
        };

        if(!_this.addition_ui.with_kernel_match_report){
            delete allow_keys["i"];
            delete allow_keys["q"];
        }

        if(!_this.addition_ui.with_profile_data){
            delete allow_keys["u"];
        }

        window.addEventListener("keydown", (e)=>{
            if((e.metaKey || e.ctrlKey)){
                const key = e.key.toLowerCase();
                if(!(key in allow_keys))
                    return;
                
                e.preventDefault();
                e.stopPropagation();
                allow_keys[key]();
            }else{
                const key = e.key.toLowerCase();
                if(key == "c"){
                    if(e.target.localName == "input" || e.target.localName == "textarea") return;
                    e.preventDefault();
                    e.stopPropagation();
                    this.select_nodes({clear: true});
                }else if(key == 'f'){
                    if(e.target.localName == "input" || e.target.localName == "textarea") return;
                    e.preventDefault();
                    e.stopPropagation();
                    this.automatic_complete_select_nodes();
                }else if(key == "escape"){
                    const _this = this;
                    const actions = [
                        
                    ];
                    for(const item of actions){
                        const condition = item[0];
                        const action    = item[1]
                        if(condition){
                            e.preventDefault();
                            e.stopPropagation();
                            action();
                            break;
                        }
                    }
                }
            }
        });
    }

    parse_hash(){
        if(!(location.hash && location.hash.length > 1))
            return {};

        const result = {};
        const vars   = location.hash.substring(1).split("&");
        for(const item of vars){
            const p = item.indexOf("=");
            if(p != -1){
                const key   = item.substring(0, p);
                const value = item.substring(p + 1);
                result[key] = value;
            }
        }
        return result;
    }

    setup_default_view(){
        const hash_param = this.parse_hash();
        if("current" in hash_param)
            hash_param.current = parseInt(hash_param.current);

        const node_ids = "current" in hash_param ? [hash_param.current] : [];
        let do_not_need_scroll_to = false;
        if(hash_param.select != null){
            const node_ids_string = hash_param.select.split(",");
            for(const idd of node_ids_string){
                if(idd.length > 0)
                    node_ids.push(parseInt(idd));
            }
        }
        if(node_ids.length > 0){
            this.select_nodes({clear: true, current_node_id: hash_param.current}, ...node_ids);
            this.scroll_to(node_ids[0], "instant", "center");
            do_not_need_scroll_to = true;
        }

        if(this.inputs.length > 0 && !do_not_need_scroll_to)
            this.scroll_to(this.inputs[0].raw.idd, "instant", "top");

        if("issue" in hash_param){
            hash_param.issue = parseInt(hash_param.issue);
            this.addition_ui.issuelist_view_plane_to_bottom(hash_param.issue);
        }
    }

    zoom(scale, cx, cy){
        const old_scale = this.screen_scale;
        this.screen_scale = Math.max(0.05, Math.min(this.screen_scale + scale, 1.5));
        if(this.screen_scale == old_scale) return;
        const sb = this.screen.getBoundingClientRect();
        const pb = this.screen.parentElement.getBoundingClientRect();
        const diff_x = sb.left - pb.left;
        const diff_y = sb.top - pb.top;
        const ax = (cx + window.scrollX - diff_x) / old_scale * this.screen_scale - cx + diff_x;
        const ay = (cy + window.scrollY - diff_y) / old_scale * this.screen_scale - cy + diff_y;
        const new_width  = this.size_config.content_width * this.screen_scale;
        const new_height = this.size_config.content_height * this.screen_scale;
        this.screen.setAttribute("width", new_width);
        this.screen.setAttribute("height", new_height);
        window.scrollTo({left: ax, top: ay});
    }

    zoom_out(){
        // small
        this.zoom(0.15, window.innerWidth * 0.5, window.innerHeight * 0.5);
    }

    zoom_in(){
        // big
        this.zoom(-0.15, window.innerWidth * 0.5, window.innerHeight * 0.5);
    }

    async initialize(graph){
        this.screen_scale = 1.0;
        this.render_graph(graph);
        this.addition_ui.initialize();
        this.setup_pointer_move_event();
        this.loadingInstance.close();
        this.setup_default_view();
        this.setup_global_keys_event();
    }

    mount(screen_element_id, ui_element_id, graph_url){
        this.addition_ui = new AdditionUI(this, ui_element_id);
        this.screen      = document.getElementById(screen_element_id);
        this.graph_url   = graph_url;
        this.metadata    = {};
        this.select_nodes_bbox_element = document.getElementById("select-nodes-bbox");
        this.select_nodes_bbox_element.style["display"] = "none";

        this.loadingInstance = window.ELEMENT.Loading.service({ fullscreen: true });
        fetch(meta_data_file).then(async (rep)=>{
            let metadata = await rep.json();
            for(const item of metadata){
                this.metadata[item.name] = item;
            }
        });

        fetch(graph_url).then(async (rep)=>{
            let graph = await rep.json();
            this.initialize(graph);
        });
    }
};

const meta_mapping = {};
for(const item of document.getElementsByTagName("meta")){
    meta_mapping[item.name] = item.content;
}

const render = new ONNXRender();
render.mount("screen", "addition-ui", meta_mapping["onnx-json-url"]);