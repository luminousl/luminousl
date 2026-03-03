import { layout } from "./static/graph.js";
import fs from "fs";

const args = process.argv.slice(2);
if(args.length != 1 && args.length != 2){
    console.log("No graph file provided.");
    process.exit(0);
}

const graph_file   = args[0];
const caching_file = args.length == 2 ? args[1] : null;

function simplify_title(title){
    if(title == null || title == "") return "";
    const p = title.lastIndexOf("/");
    if(p != -1) return title.substring(p + 1);
    return title;
}

function measure_node_size(name, io){
    name = simplify_title(name);
    return {width: name.length * 8.5 + 12, height: 21 + 8};
}

function network_layout(graph){
    const nodes_with_addition_info = [];
    const nodes_by_input  = {};
    const nodes_by_output = {};
    const edges = [];

    for(const input of graph.input){
        const box = measure_node_size(input.name, true);
        const instance = {
            width: box.width,
            height: box.height,
            v: input.idd + "",
            parent: null,
            type: "input",
            raw: input
        };
        nodes_with_addition_info.push(instance);
        nodes_by_output[input.name] = [instance];
    }

    for(const node of graph.node){
        const box  = measure_node_size(node.optype, true);
        const instance = {
            width: box.width,
            height: box.height,
            v: node.idd + "",
            parent: null,
            type: "node",
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
        const box = measure_node_size(output.name, true);
        const instance = {
            width: box.width,
            height: box.height,
            v: output.idd + "",
            parent: null,
            type: "output",
            raw: output
        };
        nodes_with_addition_info.push(instance);
        if(!(output.name in nodes_by_input)){
            nodes_by_input[output.name] = [];
        }
        nodes_by_input[output.name].push(instance);
    }

    for(const node of nodes_with_addition_info){
        let edge_ins, edge_outs;
        if(node.type == "input"){
            edge_ins = [node];
            edge_outs = nodes_by_input[node.raw.name];
            if(edge_outs == null){
                edge_outs = [];
            }
        }else if(node.type == "output"){
            edge_ins = nodes_by_output[node.raw.name];
            edge_outs = [node];
            if(edge_ins == null) edge_ins = [];
        }else if(node.type == "node"){
            edge_ins = [node];
            edge_outs = [];
            for(const out of node.raw.output){
                if(!(out in nodes_by_input)) continue;
                for(const item of nodes_by_input[out]){
                    edge_outs.push(item);
                }
            }
        }

        for(const i of edge_ins){
            for(const o of edge_outs){
                const edge = {
                    v: i.v,
                    w: o.v,
                    labeloffset: 15,
                    labelpos: "r",
                    minlen: 1,
                    weight: 1,
                    width: 1,
                    height: 1,
                };
                edges.push(edge);
            }
        }
    }
    layout(nodes_with_addition_info, edges, {nodesep: 50, ranksep: 20}, {});

    const saved_nodes = [];
    const saved_edges = [];
    for(const node of nodes_with_addition_info){
        saved_nodes.push({
            x: node.x,
            y: node.y
        });
    }

    for(const edge of edges){
        saved_edges.push({
            x: edge.x,
            y: edge.y,
            points: edge.points
        });
    }
    return {nodes: saved_nodes, edges: saved_edges};
}

let graph = JSON.parse(fs.readFileSync(graph_file));
const layout_result = network_layout(graph);
graph["layout"] = layout_result;
fs.writeFileSync(graph_file, JSON.stringify(graph));

if(caching_file != null){
    fs.writeFileSync(caching_file, JSON.stringify(layout_result));
}