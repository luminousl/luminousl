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

const knows_keywords = new Set([
    "Input",
    "Output",
    "Q",
    "DQ",
    "QDQ",
    "Add",
    "Sub",
    "Mul",
    "Div",
    "Relu",
    "Sigmoid",
    "Tanh",
    "Exp",
    "Log",
    "Sqrt",
    "Abs",
    "Neg",
    "Floor",
    "Ceil",
    "Round",
    "Sin",
    "Cos",
    "Tan",
    "Asin",
    "Acos",
    "Atan",
    "Atan2",
    "Sinh",
    "Cosh",
    "Conv",
    "Gemm",
    "Matmul",
    "MatMul",
    "MaxPool",
    "AvgPool",
    "GlobalAvgPool",
    "GlobalAveragePool",
    "GlobalAveragePooling",
    "Linear",
    "Dense",
    "LayerNorm",
    "BN",
    "BatchNorm",
    "BatchNormalization",
    "Dropout",
    "Softmax",
    "LogSoftmax",
    "Gelu",
    "SiLU",
    "Concat",
    "Split",
    "Slice",
    "Clip",
    "HardSigmoid",
    "HardSwish",
    "Transpose",
    "Reshape",
    "Flatten",
    "Unsqueeze",
    "Squeeze",
    "Deconv",
    "ConvTranspose",
    "LSTM",
    "RNN",
    "LSTMCell",
    "GRUCell",
    "Gather",
    "Scatter",
    "ScatterND",
    "Upsample",
    "Resize",
    "ArgMax",
    "ArgMin",
    "TopK",
    "OneHot",
    "Softplus",
    "Softsign",
    "Cast",
    "QuantizeLinear",
    "DequantizeLinear"
]);

class AdditionUI extends Vue{
    constructor(app, ui_element_id){
        super({
        el: "#" + ui_element_id,
        data: {
            global_theme: "light",
            highlight_precision: "weakly",
            node_info_dlg: {
                show: false,
                current_node_id: 0,
                active_collapse: ['1', '2', '3', '4', '5', '6'],
                current_node: {
                    raw: {
                        attrs: [],
                    }
                },
                current_profile_list: [],
                current_inputs_label: [],
                current_outputs_label: [],
                current_inputs_value: [],
                current_outputs_value: [],
                current_inputs_node_ids: [],
                current_outputs_node_ids: [],
                current_profile_compared_list: [],
            },
            tensor_info_dlg: {
                show: false,
                current_tensor_id: 0,
                active_collapse: ['1', '2'],
                current_tensor: {},
                producers: [],
                consumers: [],
            },
            model_info_dlg: {
                show: false,
                input: [],
                output: [],
                active_collapse: ['1', '2']
            },
            health_check_dlg: {
                show: false,
                issues: [],
                loading: false
            },
            new_issue_dlg:{
                show: false,
                keywords: "",
                description: "",
                associate_nodes: [],
                associate_current_node: null,
                mode: "",
                edited_issue_id: null,
                current_issue: null
            },
            select_list_dlg: {
                show: false,
                current_select_nodes: [],
                keeped_nodes: []
            },
            bird_eye_view: {
                show: false
            },
            coder_view_plane: {
                pointer_down: false,
                down_position: {x: 0, y: 0},
                show: false,
                code: [
                    "for(const node of app.nodes){",
                    "  if(node.raw.optype == \"Conv\"){",
                    // "    const node_latency = node.raw.profiles.map((x)=>{return x.Profile.averageMs}).reduce((acc, cur) => acc + cur);",
                    "    print(node.raw.name, \"Profiles \" + node.raw.profiles.length);",
                    "  }",
                    "}",
                ].join("\n"),
                console_log: "",
                y_pos_bottom_mode: false,
            },
            profile_view_plane: {
                pointer_down: false,
                down_position: {x: 0, y: 0},
                show: false,
                profiles: [],
                // default_order: [],
                // max_latency_order: [],
                // current_order: [],
                y_pos_bottom_mode: false,
                trtperf_columns: [] // [{prop_name: "aa", show_name: "AA", width: 80}]
            },
            subgraph_preview_dlg: {
                show: false,
                selected_nodes: [],
                load_onnx: false,
                current_code_tab: {"profiling_list": []},
                code_tabs: [
                    {"name": "current", "code": "", "profiling_list": []},
                ],
                input_name_dialog_show: false,
                input_name: "graph0",
                current_profiling_tab: "ONNX View",
                profiling_script: {
                    show: false,
                    code: ""
                },
                attached_files: {
                    show: false,
                    progress: 100,
                    progress_show: false,
                    file_list: [
                        {name: "test", size: "12KB", date: (new Date()).toString()},
                        {name: "test", size: "12KB", date: (new Date()).toString()},
                        {name: "test", size: "12KB", date: (new Date()).toString()},
                        {name: "test", size: "12KB", date: (new Date()).toString()},
                        {name: "test", size: "12KB", date: (new Date()).toString()},
                        {name: "test", size: "12KB", date: (new Date()).toString()},
                    ]
                }
            },
            compare_view_plane: {
                pointer_down: false,
                down_position: {x: 0, y: 0},
                show: false,
                profiles: [],
                // default_groups_order: [],
                // max_latency_groups_order: [],
                // max_issue_groups_order: [],
                // max_percentage_groups_order: [],
                // current_groups_order: [],
                y_pos_bottom_mode: false,
                total_diff_latency: 0,
                current_row: null,
                total_profile_latency: 0,
                total_compared_latency: 0,
                total_latency_policy_remove_empty_group: true
            },
            perlayer_compare_view_plane: {
                pointer_down: false,
                down_position: {x: 0, y: 0},
                show: false,
                nodes: [],
                // default_groups_order: [],
                // max_latency_groups_order: [],
                // max_issue_groups_order: [],
                // max_percentage_groups_order: [],
                // current_groups_order: [],
                y_pos_bottom_mode: false,
                total_diff_latency: 0,
                current_row: null,
                total_profile_latency: 0,
                total_compared_latency: 0,
                total_latency_policy_remove_empty_group: true
            },
            search_plane: {
                pointer_down: false,
                down_position: {x: 0, y: 0},
                show: false,
                filter: "",
                data: []
            },
            node_info_plane: {
                pointer_down: false,
                down_position: {x: 0, y: 0}
            },
            tensor_info_plane: {
                pointer_down: false,
                down_position: {x: 0, y: 0}
            },
            model_info_plane: {
                pointer_down: false,
                down_position: {x: 0, y: 0}
            },
            health_check_plane: {
                pointer_down: false,
                down_position: {x: 0, y: 0},
                y_pos_bottom_mode: false
            },
            select_list_plane: {
                pointer_down: false,
                down_position: {x: 0, y: 0}
            },
            issuelist_view_plane: {
                pointer_down: false,
                down_position: {x: 0, y: 0},
                y_pos_bottom_mode: false,
                show: false,
                issues: [],
                current_issue: null,
                loading: false
            },
            viewinfo_view_plane: {
                pointer_down: false,
                down_position: {x: 0, y: 0},
                show: false,
                active_collapse: ["basic", "layers_summary"],
                basic_info: {
                    model_name: "",
                    size_bytes: 0,
                    virtual_folder: "",
                    description: "",
                    create_time: "",
                    file_id: 0
                },
                files: [],
                layers_summary: [],
                performance_summary: "",
                performance_summary_compared: ""
            },
            filter_by_keywords_dlg: {
                show: false,
                keywords: "",
                filter_by_what: ["TRT Name", "OP Type", "OP Name", "Tactic Name"],
                filter_mode: "Include",
                mode: ""
            },
            with_kernel_match_report: false,
            with_profile_data: false,
            have_logfile: false,
            selected_nodes_summary: {
                show: false,
                realtime: {
                    show: false,
                    major_latency: 0,
                    major_latency_title: "Profile A Latency",
                    compared_latency: 0,
                    compared_latency_title: "Profile B Latency",
                    total_nodes: 0,
                },
                selected: {
                    show: true,
                    major_latency: 0,
                    major_latency_title: "Profile A Latency",
                    compared_latency: 0,
                    compared_latency_title: "Profile B Latency",
                    total_nodes: 0,
                }
            }
        },
        mounted(){
            this.hook_plane_move("profile_view_plane");
            this.hook_plane_move("compare_view_plane");
            this.hook_plane_move("search_plane");
            this.hook_plane_move("node_info_plane");
            this.hook_plane_move("model_info_plane");
            this.hook_plane_move("select_list_plane");
            this.hook_plane_move("issuelist_view_plane");
            this.hook_plane_move("viewinfo_view_plane");
            this.hook_plane_move("perlayer_compare_view_plane");
            this.hook_plane_move("tensor_info_plane");
            this.hook_plane_move("health_check_plane");
            // this.hook_plane_move("coder_view_plane");
        },
        methods: {
            hook_plane_move(name){
                const viewinfo_header = this.$refs[name].$el.getElementsByClassName("el-card__header")[0];
                viewinfo_header.addEventListener("pointerup", (e)=>{this.view_move_pointer_event(viewinfo_header, e, this[name], this.$refs[name].$el)});
                viewinfo_header.addEventListener("pointerdown", (e)=>{this.view_move_pointer_event(viewinfo_header, e, this[name], this.$refs[name].$el)});
                viewinfo_header.addEventListener("pointermove", (e)=>{this.view_move_pointer_event(viewinfo_header, e, this[name], this.$refs[name].$el)});
            },
            view_move_pointer_event(elm, e, plane, container){
                if(e.target.classList.contains("avoid-pointer-event") || e.target.parentElement.classList.contains("avoid-pointer-event"))
                    return;

                e.preventDefault();
                e.stopPropagation();
                if(e.type == "pointerdown"){
                    elm.setPointerCapture(e.pointerId);
                    plane.pointer_down = true;
                    plane.down_position = {
                        x: e.x, y: e.y, 
                        cx: container.offsetLeft,
                        cy: container.offsetTop
                    };
                }else if(e.type == "pointermove"){
                    if(plane.pointer_down){
                        const p = plane.down_position;
                        container.style["left"] = e.x - p.x + p.cx;
                        container.style["top"]  = e.y - p.y + p.cy;
                    }
                }else if(e.type == "pointerup"){
                    if(plane.pointer_down){
                        elm.releasePointerCapture(e.pointerId);
                        plane.pointer_down = false;
                    }
                }
            },
            copy_content(content, tips){
                const handler = (event)=>{
                    event.clipboardData.setData("text/plain", content);
                    event.preventDefault();
                    document.removeEventListener('copy', handler, true);
                };
                document.addEventListener('copy', handler, true);
                document.execCommand('copy');
                this.$message({
                    message: tips,
                    center: true,
                    type: "success"
                });
            },
            select_list_copy_names(){
                let names = this.select_list_dlg.current_select_nodes.map((x)=>{return "\"" + x.raw_name + "\"";}).join(", ");
                names = "[" + names + "]";
                this.copy_content(names, "Names have been copied!");
            },
            new_issue_dlg_copy_url_click(){
                this.new_issue_dlg.show = false;

                const handler = (event)=>{
                    event.clipboardData.setData("text/plain", location.origin + location.pathname + "#issue=" + this.new_issue_dlg.current_issue.issue_id);
                    event.preventDefault();
                    document.removeEventListener('copy', handler, true);
                };
                document.addEventListener('copy', handler, true);
                document.execCommand('copy');
                this.$message({
                    message: 'URL has been copied!',
                    center: true,
                    type: "success"
                });
            },
            issuelist_issues_table_dbclick(row){
                this.issuelist_view_plane.current_issue = row;

                const issue = this.issuelist_view_plane.current_issue;
                this.new_issue_dlg.associate_nodes = issue.associate_nodes.node_ids.map((idd)=>{return app.node_mapping[idd];});
                this.new_issue_dlg.associate_current_node = app.node_mapping[issue.associate_nodes.current_node_id];
                this.new_issue_dlg.keywords = issue.keywords;
                this.new_issue_dlg.description = issue.description;
                this.new_issue_dlg.current_issue = issue;
                this.new_issue_dlg.mode = "readonly";
                this.new_issue_dlg.show = true;
            },
            issuelist_issues_table_click(row){
                this.issuelist_view_plane.current_issue = row;
                if(row){
                    let current_id = row.associate_nodes.current_node_id;
                    app.select_nodes({clear: true, current_node_id: current_id, issue_id:row.issue_id}, ...row.associate_nodes.node_ids);
                    if(row.associate_nodes.current_node_id == null)
                        current_id = row.associate_nodes.node_ids[0];
                    app.scroll_to(current_id, "smooth");

                    if(this.node_info_dlg.show && current_id in app.node_mapping){
                        this.show_node(app.node_mapping[current_id]);
                    }
                }
            },
            convert_issue_to_show(issue){
                return {
                    issue_id: issue.issue_id, 
                    keywords: issue.keywords, 
                    description: issue.description, 
                    creator: issue.creator, 
                    create_time: issue.create_time, 
                    update_time: issue.update_time,
                    associate_nodes: issue.associate_nodes,
                    num_associate_nodes: issue.associate_nodes.node_ids.length
                };
            },
            query(url, body=null, pre_callback=null, post_callback=null, final_callback=null){
                const _this = this;
                var xhr = new XMLHttpRequest();
                xhr.open('POST', url, true);
                xhr.setRequestHeader("Content-Type", "application/json;charset=UTF-8");
                xhr.onload = function (e) {
                    if(pre_callback != null) pre_callback(xhr);
                    if (xhr.status == 200) {
                        const resp = JSON.parse(xhr.responseText);
                        if(resp.status != "ok"){
                            if(final_callback != null) final_callback(xhr);
                            _this.$message({
                                message: "Failed to query url: " + url + ", error: " + resp.message,
                                center: true,
                                type: "error"
                            });
                            return;
                        }
                        if(post_callback != null) post_callback(resp.data);
                    }
                    if(final_callback != null) final_callback(xhr);
                };
                xhr.send(body != null ? JSON.stringify(body) : null);
            },
            refresh_issuelist(callback){
                this.issuelist_view_plane.current_issue = null;
                this.issuelist_view_plane.loading = true;
                const _this = this;
                // this.loadingInstance = window.ELEMENT.Loading.service({ fullscreen: true });
                var xhr = new XMLHttpRequest();
                xhr.open('POST', '/get_issuelist_by_view_id/' + app.view_id, true);
                xhr.setRequestHeader("Content-Type", "application/json;charset=UTF-8");
                xhr.onload = function (e) {
                    // _this.loadingInstance.close();
                    _this.issuelist_view_plane.loading = false;
                    if(callback) callback();
                    if (xhr.status == 200) {
                        const resp = JSON.parse(xhr.responseText);
                        if(resp.status != "ok"){
                            _this.$message({
                                message: 'Failed to create issue, error: ' + resp.message,
                                center: true,
                                type: "error"
                            });
                            return;
                        }
                        _this.issuelist_view_plane.issues = resp.data.issues.map(_this.convert_issue_to_show);
                    }
                };
                xhr.send();
            },
            new_issue_dlg_confirm(){
                const _this = this;
                // this.loadingInstance = window.Element.Loading.service({ fullscreen: true });
                var xhr = new XMLHttpRequest();
                if(this.new_issue_dlg.mode == "new"){
                    xhr.open('POST', '/create_new_issue', true);
                }else if(this.new_issue_dlg.mode == "edit"){
                    xhr.open('POST', '/update_issue', true);
                }else{
                    console.log("Unknow mode: " + this.new_issue_dlg.mode);
                    return;
                }

                xhr.setRequestHeader("Content-Type", "application/json;charset=UTF-8");
                xhr.onload = function (e) {
                    _this.new_issue_dlg.show = false;
                    // _this.loadingInstance.close();
                    if (xhr.status == 200) {
                        const resp = JSON.parse(xhr.responseText);
                        if(resp.status != "ok"){
                            _this.$message({
                                message: 'Failed to create issue, error: ' + resp.message,
                                center: true,
                                type: "error"
                            });
                            return;
                        }
                        if(_this.new_issue_dlg.mode == "new"){
                            _this.issuelist_view_plane.issues.push(_this.convert_issue_to_show(resp.data));
                        }else if(_this.new_issue_dlg.mode == "edit"){
                            for(let i = 0; i < _this.issuelist_view_plane.issues.length; ++i){
                                const issue = _this.issuelist_view_plane.issues[i];
                                if(issue.issue_id == _this.new_issue_dlg.edited_issue_id){
                                    issue.issue_id = resp.data.issue_id;
                                    issue.keywords = resp.data.keywords;
                                    issue.description = resp.data.description;
                                    issue.creator = resp.data.creator;
                                    issue.create_time = resp.data.create_time;
                                    issue.update_time = resp.data.update_time;
                                    issue.associate_nodes = resp.data.associate_nodes;
                                    issue.num_associate_nodes = resp.data.associate_nodes.node_ids.length;
                                    break;
                                }
                            }
                        }
                    }
                };
                const posted_issue = {
                    keywords: this.new_issue_dlg.keywords, 
                    description: this.new_issue_dlg.description, 
                    creator: "",
                    view_id: app.view_id,
                    associate_nodes: {
                        node_ids: this.new_issue_dlg.associate_nodes.map((x)=>{return x.raw.idd;}),
                        current_node_id: this.new_issue_dlg.associate_current_node ? this.new_issue_dlg.associate_current_node.raw.idd : null
                    }
                };
                if(this.new_issue_dlg.mode == "edit"){
                    posted_issue.issue_id = this.new_issue_dlg.edited_issue_id;
                }
                xhr.send(JSON.stringify(posted_issue));
            },
            new_issue_dlg_remove_associate_node(node){
                const index = this.new_issue_dlg.associate_nodes.indexOf(node);
                this.new_issue_dlg.associate_nodes.splice(index, 1);
            },
            new_issue_click(){
                this.new_issue_dlg.associate_nodes = [...app.selected_nodes];
                this.new_issue_dlg.associate_current_node = app.selected_nodes_current;
                this.new_issue_dlg.show = true;
                this.new_issue_dlg.current_issue = null;
                this.new_issue_dlg.mode = "new";
                this.$nextTick(()=>{
                    this.$refs.new_issue_dlg_keywords.focus();
                    this.$refs.new_issue_dlg_keywords.select();
                });
            },
            keeped_nodes_clean(){
                this.select_list_dlg.keeped_nodes = [];
            },
            add_keeped_node(node_id){
                for(const item of this.select_list_dlg.keeped_nodes){
                    if(item.idd == node_id)
                        return;
                }
                const node = app.node_mapping[node_id];
                let show_name = node.raw.idd + " [" + node.raw.optype + "] " + node.raw.name;
                if(node.type == "input" || node.type == "output"){
                    show_name = node.raw.idd + " [" + node.type + "] " + node.raw.name;
                }
                this.select_list_dlg.keeped_nodes.push({
                    name: show_name,
                    idd: node.raw.idd
                });
            },
            inverse_array(list){
                const result = [];
                for(let i = list.length - 1; i >= 0; --i)
                    result.push(list[i]);
                return result;
            },
            coder_text_area_tab_keydown(e){
                e.preventDefault();
                e.stopImmediatePropagation();

                var shift = e.shiftKey;
                var el = this.$refs.coder_codeblocks.$refs.textarea;
                var start = el.selectionStart,
                    end = el.selectionEnd, 
                    value = el.value;

                var lineStart = value.lastIndexOf('\n', start),
                    lineEnd = value.indexOf('\n', end),
                    offset = 0;

                if (lineStart === -1) lineStart = 0;
                if (lineEnd === -1) lineEnd = value.length;

                if (lineStart === lineEnd);
                else if (lineStart !== 0) lineStart += 1;

                var nspaces = 2;
                var spaces  = " ".repeat(nspaces);
                var lines = value.substring(lineStart, lineEnd).split('\n');
                if (lines.length > 1) {
                    if(!shift){
                        offset = lines.length;
                        lines = spaces + lines.join('\n' + spaces);
                        el.value = value.substring(0, lineStart) + lines + value.substring(lineEnd);
                        el.selectionStart = start + 2;
                        el.selectionEnd = end + offset * 2;
                    }else{
                        let num_offset_chars = 0;
                        let num_frist_offset_chars = 0;
                        for(let i = 0; i < lines.length; ++i){
                            for(let j = 0; j < nspaces; ++j){
                                if(lines[i].startsWith(" ")){
                                    lines[i] = lines[i].substring(1);
                                    num_offset_chars += 1;
                                    if(i == 0) num_frist_offset_chars += 1;
                                }
                            }
                        }
                        el.value = value.substring(0, lineStart) + lines.join('\n') + value.substring(lineEnd);
                        el.selectionStart = start - num_frist_offset_chars;
                        el.selectionEnd = end - num_offset_chars;
                    }
                } else {
                    el.value = value.substring(0, start) + spaces + value.substring(end);
                    el.selectionStart = el.selectionEnd = start + nspaces;
                }
                this.coder_view_plane.code = el.value;
                e.preventDefault();
            },
            async coder_run_code(e){
                if(!(e.metaKey || e.ctrlKey)) return;
                try{
                    let __code__ = this.coder_view_plane.code;
                    const __output_lines__ = ["Running result:"];
                    const __replacer__ = (key, val)=>{
                        if(val instanceof Object && (key == "from" || key == "to") && "raw" in val){
                            return val.raw.idd;
                        }
                        if(key == "vm") return "";
                        return val;
                    }
                    const print = (...vals)=>{
                        const outs = [];
                        for(const val of vals){
                            outs.push("" + JSON.stringify(val, __replacer__, 4));
                        }
                        if(outs.length > 0){
                            __output_lines__.push(outs.join("  "));
                            this.coder_view_plane.console_log = __output_lines__.join("\n");
                        }
                    };
                    if(__code__.indexOf("\n") == -1 && __code__.indexOf("print") == -1 && __code__.indexOf("console.log") == -1 && __code__.indexOf(";") == -1){
                        __code__ = "print(" + __code__ + ")";
                    }
                    __code__ = [
                        "const __main__ = (app, print)=>{",
                        __code__,
                        "}",
                        "export default __main__;"
                    ].join("\n")
                    const module_url = URL.createObjectURL(new Blob([__code__], { type: 'text/javascript' }));
                    const module_instance = await import(module_url);
                    module_instance.default(app, print);
                    // this.coder_view_plane.console_log = __output_lines__.join("\n");
                }catch(e){
                    this.coder_view_plane.console_log = "Failed to run the code:\n" + e.message;
                }
            },
            profile_view_defaultview_click(){
                this.update_profile_view_table_data(app.graph.layerinfo, app.graph.profile, this.profile_view_plane.default_order);
                this.profile_view_plane.default_order = this.inverse_array(this.profile_view_plane.default_order);
            },
            profile_view_maxlatency_click(){
                this.update_profile_view_table_data(app.graph.layerinfo, app.graph.profile, this.profile_view_plane.max_latency_order);
                this.profile_view_plane.max_latency_order = this.inverse_array(this.profile_view_plane.max_latency_order);
            },
            filter_by_keywords_dlg_clean(){
                if(this.filter_by_keywords_dlg.mode == "compare_view"){
                    this.update_compare_view_table_data(app.graph.kernel_match, this.compare_view_plane.current_groups_order);
                }else if(this.filter_by_keywords_dlg.mode == "perlayer_compare_view"){
                    this.update_perlayer_compare_view_table_data(app.nodes, this.perlayer_compare_view_plane.current_groups_order);
                }
                this.filter_by_keywords_dlg.show = false;
            },
            filter_by_keywords_dlg_confirm(e){
                e.preventDefault();
                e.stopPropagation();
                const keywords = this.filter_by_keywords_dlg.keywords.toLowerCase();
                const subset_groups = [];
                const filter_columns = {};
                for(const keyword of this.filter_by_keywords_dlg.filter_by_what)
                    filter_columns[keyword] = true;

                const match_func_mapping = {
                    "Include": (target, keywords)=>{
                        if(!target) return 0;
                        return target.toLowerCase().indexOf(keywords) != -1 ? 1 : 0;
                    },
                    "Exclude": (target, keywords)=>{
                        if(!target) return 0;
                        return target.toLowerCase().indexOf(keywords) == -1 ? 1 : 0;
                    },
                    "Starts With": (target, keywords)=>{
                        if(!target) return 0;
                        return target.toLowerCase().startsWith(keywords) ? 1 : 0;
                    },
                    "Ends With": (target, keywords)=>{
                        if(!target) return 0;
                        return target.toLowerCase().endsWith(keywords) ? 1 : 0;
                    },
                    "Equal To": (target, keywords)=>{
                        if(!target) return 0;
                        return target.toLowerCase() == keywords ? 1 : 0;
                    },
                };
                const match_func = match_func_mapping[this.filter_by_keywords_dlg.filter_mode];
                if(this.filter_by_keywords_dlg.mode == "compare_view"){
                    for(let i = 0; i < app.graph.kernel_match.length; ++i){
                        const group = app.graph.kernel_match[i];
                        let num_matched_profile = 0;
                        for(const profile of group.ThorU){
                            if("Tactic Name" in filter_columns){
                                num_matched_profile += match_func(profile.TacticName, keywords);
                            }

                            if("TRT Name" in filter_columns){
                                num_matched_profile += match_func(profile.Name, keywords);
                            }

                            if("OP Name" in filter_columns){
                                for(const name of profile.ONNXNames)
                                    num_matched_profile += match_func(name, keywords);
                            }

                            if("OP Type" in filter_columns){
                                for(const name of profile.ONNXNames){
                                    if(app.node_name_lookup_table[name])
                                        num_matched_profile += match_func(app.node_name_lookup_table[name].raw.optype, keywords);
                                }
                            }
                        }
                        if(num_matched_profile > 0){
                            subset_groups.push(group);
                        }
                    }
                    this.update_compare_view_table_data(subset_groups);
                }else if(this.filter_by_keywords_dlg.mode == "perlayer_compare_view"){
                    for(let i = 0; i < app.nodes.length; ++i){
                        const node = app.nodes[i];
                        let num_matched_profile = 0;
                        for(const profile of (node.raw.profiles || [])){
                            if(num_matched_profile == 0 && "Tactic Name" in filter_columns){
                                num_matched_profile += match_func(profile.TacticName, keywords);
                            }

                            if(num_matched_profile == 0 && "TRT Name" in filter_columns){
                                num_matched_profile += match_func(profile.Name, keywords);
                            }

                            if(num_matched_profile == 0 && "OP Name" in filter_columns){
                                num_matched_profile += match_func(node.raw.name, keywords);
                            }

                            if(num_matched_profile == 0 && "OP Type" in filter_columns){
                                num_matched_profile += match_func(node.raw.optype, keywords);
                            }
                        }
                        if(num_matched_profile > 0){
                            subset_groups.push(node);
                        }
                    }
                    this.update_perlayer_compare_view_table_data(subset_groups);
                }
                this.filter_by_keywords_dlg.show = false;
            },
            compare_view_filter_by_keywords_click(){
                this.filter_by_keywords_dlg.show = true;
                this.filter_by_keywords_dlg.mode = "compare_view";
                this.$nextTick(()=>{
                    this.$refs.filter_by_keywords_dlg_keywords.focus();
                    this.$refs.filter_by_keywords_dlg_keywords.select();
                });
            },
            perlayer_compare_view_filter_by_keywords_click(){
                this.filter_by_keywords_dlg.show = true;
                this.filter_by_keywords_dlg.mode = "perlayer_compare_view";
                this.$nextTick(()=>{
                    this.$refs.filter_by_keywords_dlg_keywords.focus();
                    this.$refs.filter_by_keywords_dlg_keywords.select();
                });
            },
            copy_profile_view_table_data(){
                const handler = (event)=>{
                    event.clipboardData.setData("text/html", this.$refs.profile_view_table.$el.outerHTML);
                    event.preventDefault();
                    document.removeEventListener('copy', handler, true);
                };
                document.addEventListener('copy', handler, true);
                document.execCommand('copy');
                this.$message({
                    message: 'All table data has been copied to the clipboard. Please paste it into a google sheet or excel to view it properly!',
                    center: true,
                    type: "success"
                });
            },
            compare_view_copy_content(){
                const handler = (event)=>{
                    event.clipboardData.setData("text/html", this.$refs.compare_view_table.$el.outerHTML);
                    event.preventDefault();
                    document.removeEventListener('copy', handler, true);
                };
                document.addEventListener('copy', handler, true);
                document.execCommand('copy');
                this.$message({
                    message: 'All table data has been copied to the clipboard. Please paste it into a google sheet or excel to view it properly!',
                    center: true,
                    type: "success"
                });
            },
            perlayer_compare_view_copy_content(){
                const handler = (event)=>{
                    event.clipboardData.setData("text/html", this.$refs.perlayer_compare_view_table.$el.outerHTML);
                    event.preventDefault();
                    document.removeEventListener('copy', handler, true);
                };
                document.addEventListener('copy', handler, true);
                document.execCommand('copy');
                this.$message({
                    message: 'All table data has been copied to the clipboard. Please paste it into a google sheet or excel to view it properly!',
                    center: true,
                    type: "success"
                });
            },
            compare_view_max_percentage_click(){
                this.update_compare_view_table_data(app.graph.kernel_match, this.compare_view_plane.max_percentage_groups_order);
                // this.compare_view_plane.max_percentage_groups_order = this.inverse_array(this.compare_view_plane.max_percentage_groups_order);
            },
            compare_view_maxissue_click(){
                this.update_compare_view_table_data(app.graph.kernel_match, this.compare_view_plane.max_issue_groups_order);
                // this.compare_view_plane.max_issue_groups_order = this.inverse_array(this.compare_view_plane.max_issue_groups_order);
            },
            compare_view_defaultview_click(){
                this.update_compare_view_table_data(app.graph.kernel_match, this.compare_view_plane.default_groups_order);
                // this.compare_view_plane.default_groups_order = this.inverse_array(this.compare_view_plane.default_groups_order);
            },
            compare_view_maxlatency_click(){
                this.update_compare_view_table_data(app.graph.kernel_match, this.compare_view_plane.max_latency_groups_order);
                // this.compare_view_plane.max_latency_groups_order = this.inverse_array(this.compare_view_plane.max_latency_groups_order);
            },
            perlayer_compare_view_max_percentage_click(){
                this.update_perlayer_compare_view_table_data(app.nodes, this.perlayer_compare_view_plane.max_percentage_groups_order);
                // this.compare_view_plane.max_percentage_groups_order = this.inverse_array(this.compare_view_plane.max_percentage_groups_order);
            },
            perlayer_compare_view_maxissue_click(){
                this.update_perlayer_compare_view_table_data(app.nodes, this.perlayer_compare_view_plane.max_issue_groups_order);
                // this.compare_view_plane.max_issue_groups_order = this.inverse_array(this.compare_view_plane.max_issue_groups_order);
            },
            perlayer_compare_view_defaultview_click(){
                this.update_perlayer_compare_view_table_data(app.nodes, this.perlayer_compare_view_plane.default_groups_order);
                // this.compare_view_plane.default_groups_order = this.inverse_array(this.compare_view_plane.default_groups_order);
            },
            perlayer_compare_view_maxlatency_click(){
                this.update_perlayer_compare_view_table_data(app.nodes, this.perlayer_compare_view_plane.max_latency_groups_order);
                // this.compare_view_plane.max_latency_groups_order = this.inverse_array(this.compare_view_plane.max_latency_groups_order);
            },
            coder_view_plane_to_bottom(){
                this.coder_view_plane.y_pos_bottom_mode = !this.coder_view_plane.y_pos_bottom_mode;
                if(this.coder_view_plane.y_pos_bottom_mode){
                    this.coder_view_plane.old_top = this.$refs.coder_view_plane.$el.offsetTop;
                    this.$refs.coder_view_plane.$el.style["top"] = "calc(200%)";
                }else{
                    this.$refs.coder_view_plane.$el.style["top"] = this.coder_view_plane.old_top;
                    this.$nextTick(()=>{
                        this.$refs.coder_codeblocks.focus();
                        this.$refs.coder_codeblocks.select();
                    });
                }
            },
            health_check_dlg_to_bottom(){
                this.health_check_plane.y_pos_bottom_mode = !this.health_check_plane.y_pos_bottom_mode;
                if(this.health_check_plane.y_pos_bottom_mode){
                    this.health_check_plane.old_top = this.$refs.health_check_plane.$el.offsetTop;
                    this.$refs.health_check_plane.$el.style["top"] = "calc(200%)";
                }else{
                    this.$refs.health_check_plane.$el.style["top"] = this.health_check_plane.old_top;
                }
            },
            profile_view_plane_to_bottom(){
                this.profile_view_plane.y_pos_bottom_mode = !this.profile_view_plane.y_pos_bottom_mode;
                if(this.profile_view_plane.y_pos_bottom_mode){
                    this.profile_view_plane.old_top = this.$refs.profile_view_plane.$el.offsetTop;
                    this.$refs.profile_view_plane.$el.style["top"] = "calc(200%)";
                }else{
                    this.$refs.profile_view_plane.$el.style["top"] = this.profile_view_plane.old_top;
                }
            },
            compare_view_total_latency_policy_remove_empty_group_change(){
                this.update_compare_view_table_data(app.graph.kernel_match, this.compare_view_plane.current_groups_order);
            },
            compare_view_plane_to_bottom(){
                this.compare_view_plane.y_pos_bottom_mode = !this.compare_view_plane.y_pos_bottom_mode;
                if(this.compare_view_plane.y_pos_bottom_mode){
                    this.compare_view_plane.old_top = this.$refs.compare_view_plane.$el.offsetTop;
                    this.$refs.compare_view_plane.$el.style["top"] = "calc(200%)";
                }else{
                    this.$refs.compare_view_plane.$el.style["top"] = this.compare_view_plane.old_top;
                }
            },
            perlayer_compare_view_plane_to_bottom(){
                this.perlayer_compare_view_plane.y_pos_bottom_mode = !this.perlayer_compare_view_plane.y_pos_bottom_mode;
                if(this.perlayer_compare_view_plane.y_pos_bottom_mode){
                    this.perlayer_compare_view_plane.old_top = this.$refs.perlayer_compare_view_plane.$el.offsetTop;
                    this.$refs.perlayer_compare_view_plane.$el.style["top"] = "calc(200%)";
                }else{
                    this.$refs.perlayer_compare_view_plane.$el.style["top"] = this.perlayer_compare_view_plane.old_top;
                }
            },
            keepd_nodes_ondrop(e){
                e.preventDefault();
                const node_id = e.dataTransfer.getData("selected_node_idd");
                if(!node_id) return;
                this.add_keeped_node(node_id);
            },
            keepd_nodes_dragover(e){
                e.preventDefault();
            },
            select_list_drag_start(event, item){
                event.dataTransfer.setData("selected_node_idd", item.idd);
            },
            open_search_plane(){
                if(this.search_plane.show){
                    this.search_plane.show = false;
                    return;
                }
                this.search_plane.show = true;

                this.$nextTick(()=>{
                    this.$refs.search_plane_filter.focus();
                    this.$refs.search_plane_filter.select();
                });
            },
            scroll_to_nodes(...node_ids){
                const current_node_id = node_ids.length > 0 ? node_ids[0] : null;
                app.select_nodes({clear: false, current_node_id: current_node_id}, ...node_ids);
                app.scroll_to(current_node_id, "smooth", "center");
            },
            show_node(node){
                this.show_node_info_dlg(node);
                // if(node.type == "input" || node.type == "output"){
                // this.show_model_info_dlg(app.inputs, app.outputs);
                // }else if(node.type == "operation"){
                // this.show_node_info_dlg(node);
                // }
            },
            node_info_profile_locate_to(current_node, profile, source, profile_index){
                this.scroll_to_and_select_onnx_nodes_by_name(current_node.raw.idd, profile.ONNXNames, source);
                if(source == "major"){
                    current_node.current_focus_profile = profile_index;
                }
                if(this.with_profile_report && this.profile_view_plane.show && source == "major"){
                    this.profile_table_scroll_to_profile(profile.Profile.idd);
                }
                if(this.with_kernel_match_report && this.compare_view_plane.show && source == "major"){
                    this.compare_table_scroll_to_profile(profile.Profile.idd);
                }
            },
            update_profile_view_table_data(layers, profiles, order){
                const profiles_show = [];
                for(let i = 0; i < layers.Layers.length; ++i){
                    const layer = layers.Layers[order[i]];
                    const profile = profiles[layer.Name];
                    const onnx_layer_op_types = [];
                    for(const name of layer.ONNXNames){
                        if(name in app.node_name_lookup_table)
                            onnx_layer_op_types.push(app.node_name_lookup_table[name].raw.optype)
                        else{
                            onnx_layer_op_types.push("Unknow");
                        }
                    }
                    
                    const profile_item = {
                        idd: profile.idd,
                        name: layer.Name,
                        latency: profile ? profile.averageMs : -1,
                        percentage: profile ? profile.percentage : -1,
                        onnx_layers: layer.ONNXNames,
                        onnx_layers_string: onnx_layer_op_types.join(", "),
                        layer_type: layer.LayerType,
                        tactic_name: layer.TacticName
                    };

                    for(const column of this.profile_view_plane.trtperf_columns){
                        profile_item[column.prop_name] = profile[column.prop_name];
                    }
                    profiles_show.push(profile_item);
                }
                this.profile_view_plane.profiles = profiles_show;
                this.profile_view_plane.current_order = order;
            },
            open_profile_view_plane(){
                if(!this.profile_view_plane.show){
                    this.profile_view_maxlatency_click();
                    this.profile_view_plane.show = true;
                    return;
                }
                this.profile_view_plane_to_bottom();
            },
            open_coder_view_plane(){
                if(!this.coder_view_plane.show){
                    this.coder_view_plane.show = true;
                    this.$nextTick(()=>{
                        this.$refs.coder_codeblocks.focus();
                        this.$refs.coder_codeblocks.select();
                    });
                    return;
                }
                this.coder_view_plane_to_bottom();
            },
            init_table_order(){
                if(this.with_kernel_match_report){
                    const max_issue_indexs   = [];
                    const max_latency_indexs = [];
                    const default_groups_order = [];
                    const max_percentage_indexs = [];
                    for(let i = 0; i < app.graph.kernel_match.length; ++i){
                        default_groups_order.push(i);

                        const group = app.graph.kernel_match[i];
                        const ThorU = group.ThorU;
                        const OrinX = group.OrinX;
                        const max_num_items = Math.max(ThorU.length, OrinX.length);
                        let ThorU_latency_total = 0;
                        let OrinX_latency_total = 0;
                        for(let j = 0; j < max_num_items; ++j){
                            let thoru_item = ThorU[j];
                            let orinx_item = OrinX[j];
                            ThorU_latency_total += thoru_item && thoru_item.profile ? thoru_item.profile.averageMs : 0;
                            OrinX_latency_total += orinx_item && orinx_item.profile ? orinx_item.profile.averageMs : 0;
                        }
                        max_latency_indexs.push([i, ThorU_latency_total]);
                        // max_issue_indexs.push([i, ThorU_latency_total + (ThorU_latency_total > OrinX_latency_total ? 10000 : 0)]);
                        max_issue_indexs.push([i, ThorU_latency_total - OrinX_latency_total]);
                        max_percentage_indexs.push([i, (OrinX_latency_total - ThorU_latency_total) / ThorU_latency_total]);
                    }
                
                    this.compare_view_plane.default_groups_order = default_groups_order;
                    max_latency_indexs.sort((a, b)=>{return b[1] - a[1]});
                    max_issue_indexs.sort((a, b)=>{return b[1] - a[1]});
                    max_percentage_indexs.sort((a, b)=>{return a[1] - b[1]});

                    const max_latency_order = [];
                    const max_issue_order   = [];
                    const max_percentage_order = [];
                    for(let i = 0; i < max_latency_indexs.length; ++i){
                        max_latency_order.push(max_latency_indexs[i][0]);
                        max_issue_order.push(max_issue_indexs[i][0]);
                        max_percentage_order.push(max_percentage_indexs[i][0]);
                    }
                    this.compare_view_plane.max_latency_groups_order    = max_latency_order;
                    this.compare_view_plane.max_issue_groups_order      = max_issue_order;
                    this.compare_view_plane.max_percentage_groups_order = max_percentage_order;
                    
                    ///////////////////////////////////////////////////////////////////////////////////////////////
                    const perlayer_max_issue_indexs   = [];
                    const perlayer_max_latency_indexs = [];
                    const perlayer_default_groups_order = [];
                    const perlayer_max_percentage_indexs = [];
                    for(let i = 0; i < app.nodes.length; ++i){
                        perlayer_default_groups_order.push(i);

                        const node = app.nodes[i];
                        const profile_a = node.raw.profiles || [];
                        const profile_b = node.raw.profiles_compared || [];
                        const max_num_items = Math.max(profile_a.length, profile_b.length);
                        let profile_a_latency_total = 0;
                        let profile_b_latency_total = 0;
                        for(let j = 0; j < max_num_items; ++j){
                            let profile_a_item = profile_a[j];
                            let profile_b_item = profile_b[j];
                            profile_a_latency_total += profile_a_item && profile_a_item.Profile ? profile_a_item.Profile.averageMs : 0;
                            profile_b_latency_total += profile_b_item && profile_b_item.Profile ? profile_b_item.Profile.averageMs : 0;
                        }
                        perlayer_max_latency_indexs.push([i, profile_a_latency_total]);
                        // max_issue_indexs.push([i, profile_a_latency_total + (profile_a_latency_total > profile_b_latency_total ? 10000 : 0)]);
                        perlayer_max_issue_indexs.push([i, profile_a_latency_total - profile_b_latency_total]);
                        perlayer_max_percentage_indexs.push([i, (profile_b_latency_total - profile_a_latency_total) / profile_a_latency_total]);
                    }
                
                    this.perlayer_compare_view_plane.default_groups_order = perlayer_default_groups_order;
                    perlayer_max_latency_indexs.sort((a, b)=>{return b[1] - a[1]});
                    perlayer_max_issue_indexs.sort((a, b)=>{return b[1] - a[1]});
                    perlayer_max_percentage_indexs.sort((a, b)=>{return a[1] - b[1]});

                    const perlayer_max_latency_order = [];
                    const perlayer_max_issue_order   = [];
                    const perlayer_max_percentage_order = [];
                    for(let i = 0; i < perlayer_max_latency_indexs.length; ++i){
                        perlayer_max_latency_order.push(perlayer_max_latency_indexs[i][0]);
                        perlayer_max_issue_order.push(perlayer_max_issue_indexs[i][0]);
                        perlayer_max_percentage_order.push(perlayer_max_percentage_indexs[i][0]);
                    }
                    this.perlayer_compare_view_plane.max_latency_groups_order    = perlayer_max_latency_order;
                    this.perlayer_compare_view_plane.max_issue_groups_order      = perlayer_max_issue_order;
                    this.perlayer_compare_view_plane.max_percentage_groups_order = perlayer_max_percentage_order;
                };

                this.with_profile_data = app.graph.layerinfo != null;
                this.have_logfile = app.graph.buildlog_meta != null;
                if(this.with_profile_data){
                    const profile_default_order = [];
                    const profile_latency_indices = [];
                    const profile_latency_order = [];
                    for(let i = 0; i < app.graph.layerinfo.Layers.length; ++i){
                        profile_default_order.push(i);

                        const layer = app.graph.layerinfo.Layers[i];
                        const profile = app.graph.profile != null ? app.graph.profile[layer.Name] : null;
                        profile_latency_indices.push([i, profile ? profile.averageMs : 0]);
                    }
                    profile_latency_indices.sort((a, b)=>{return b[1] - a[1]});

                    for(const item of profile_latency_indices)
                        profile_latency_order.push(item[0]);

                    this.profile_view_plane.default_order = profile_default_order;
                    this.profile_view_plane.current_order = Object.assign(profile_latency_order, []);
                    this.profile_view_plane.max_latency_order = profile_latency_order;
                }
            },
            compare_table_current_change(current_row){
                this.compare_view_plane.current_row = current_row;
            },
            perlayer_compare_table_current_change(current_row){
                this.perlayer_compare_view_plane.current_row = current_row;
            },
            update_compare_view_table_data(kernel_match, order=null){
                const profiles_show = [];
                const merge_rows    = {};
                const merge_rows_mask = [];
                let total_diff_latency = 0;
                let total_profile_latency = 0;
                let total_compared_latency = 0;
                for(let i = 0; i < kernel_match.length; ++i){
                    const group = order ? kernel_match[order[i]] : kernel_match[i];
                    const ThorU = group.ThorU;
                    const OrinX = group.OrinX;
                    const max_num_items = Math.max(ThorU.length, OrinX.length);
                    const row_start_id = profiles_show.length;
                    merge_rows[row_start_id] = max_num_items;
                    let ThorU_latency_total = 0;
                    let OrinX_latency_total = 0;
                    for(let j = 0; j < max_num_items; ++j){
                        let orinx_item = OrinX[j];
                        let thoru_item = ThorU[j];
                        ThorU_latency_total += thoru_item && thoru_item.profile && thoru_item.profile.averageMs != null ? thoru_item.profile.averageMs : 0;
                        OrinX_latency_total += orinx_item && orinx_item.profile && orinx_item.profile.averageMs != null ? orinx_item.profile.averageMs : 0;
                    }
                    for(let j = 0; j < max_num_items; ++j){
                        let orinx_item = OrinX[j];
                        let thoru_item = ThorU[j];
                        const thoru_onnx_layer_op_types = [];
                        const orinxu_onnx_layer_op_types = [];
                        for(const name of (thoru_item ? thoru_item.ONNXNames : [])){
                            if(name in app.node_name_lookup_table)
                                thoru_onnx_layer_op_types.push(app.node_name_lookup_table[name].raw.optype)
                            else{
                                thoru_onnx_layer_op_types.push("Unknow");
                            }
                        }

                        for(const name of (orinx_item ? orinx_item.ONNXNames : [])){
                            if(name in app.node_name_lookup_table)
                                orinxu_onnx_layer_op_types.push(app.node_name_lookup_table[name].raw.optype)
                            else{
                                orinxu_onnx_layer_op_types.push("Unknow");
                            }
                        }

                        const current_row_index = profiles_show.length;
                        // if(old_select_row && group.index == old_select_row.group_index && j == old_select_row.inner_index){
                        //     matched_select_row = current_row_index;
                        // }
                        merge_rows_mask.push(j == 0);
                        profiles_show.push({
                            index: current_row_index,
                            group_index: group.index,
                            inner_index: j,
                            name: thoru_item ? thoru_item.Name : "",
                            profile_id: thoru_item && thoru_item.profile ? app.profile_name_to_idd[thoru_item.profile.name] : null,
                            latency: thoru_item && thoru_item.profile && thoru_item.profile.averageMs != null ? Math.round(thoru_item.profile.averageMs * 10000) / 10000 : null,
                            onnx_layers: thoru_item ? thoru_item.ONNXNames : [],
                            onnx_layers_string: thoru_onnx_layer_op_types.join(", "),
                            layer_type: thoru_item ? thoru_item.LayerType : "",
                            tactic_name: thoru_item ? thoru_item.TacticName : "",
                            latency_total: Math.round(ThorU_latency_total * 10000) / 10000,
                            compared_name: orinx_item ? orinx_item.Name : "",
                            compared_latency: orinx_item && orinx_item.profile && orinx_item.profile.averageMs != null ? Math.round(orinx_item.profile.averageMs * 10000) / 10000 : null,
                            compared_onnx_layers: orinx_item ? orinx_item.ONNXNames : [],
                            compared_onnx_layers_string: orinxu_onnx_layer_op_types.join(", "),
                            compared_layer_type: orinx_item ? orinx_item.LayerType : "",
                            compared_tactic_name: orinx_item ? orinx_item.TacticName : "",
                            compared_latency_total: Math.round(OrinX_latency_total * 10000) / 10000,
                            latency_percentage: Math.round((ThorU_latency_total == 0 ? 0 : (OrinX_latency_total - ThorU_latency_total) / ThorU_latency_total * 100)) + "%",
                            diff_latency: Math.round((ThorU_latency_total - OrinX_latency_total) * 10000) / 10000
                        });
                    }
                    total_profile_latency += ThorU_latency_total;
                    total_compared_latency += OrinX_latency_total;

                    if(this.compare_view_plane.total_latency_policy_remove_empty_group && OrinX.length == 0){
                        // skip to compute the latency.
                    }else{
                        total_diff_latency += Math.max(ThorU_latency_total - OrinX_latency_total, 0);
                    }
                }
                this.compare_view_plane.merge_rows = merge_rows;
                this.compare_view_plane.merge_rows_mask = merge_rows_mask;
                this.compare_view_plane.profiles = profiles_show;
                this.compare_view_plane.total_diff_latency = total_diff_latency;
                this.compare_view_plane.total_profile_latency  = total_profile_latency;
                this.compare_view_plane.total_compared_latency = total_compared_latency;

                if(order)
                    this.compare_view_plane.current_groups_order = order;

                // if(matched_select_row != -1){
                //     this.$nextTick(()=>{
                //         const expandedRows = this.$refs.compare_view_table.bodyWrapper.querySelectorAll(".el-table__expanded-cell");
                //         const theTableRows = this.$refs.compare_view_table.bodyWrapper.querySelectorAll(".el-table__body tbody .el-table__row");
                //         let totalHeight = 0;
                //         for (let i = 0; i < Math.max(0, matched_select_row - 2); i++) {
                //             totalHeight += theTableRows[i].offsetHeight;
                //             if (expandedRows[i]) {
                //                 totalHeight += expandedRows[i].offsetHeight;
                //             }
                //         }
                //         this.$refs.compare_view_table.bodyWrapper.scrollTop = totalHeight;
                //         this.$refs.compare_view_table.setCurrentRow(this.compare_view_plane.profiles[matched_select_row]);
                //     });
                // }
            },
            update_perlayer_compare_view_table_data(nodes, order=null){
                const profiles_show = [];
                const merge_rows    = {};
                const merge_rows_mask = [];
                let total_diff_latency = 0;
                let total_profile_latency = 0;
                let total_compared_latency = 0;
                for(let i = 0; i < nodes.length; ++i){
                    const node = order ? nodes[order[i]] : nodes[i];
                    const profile_a = node.raw.profiles || [];
                    const profile_b = node.raw.profiles_compared || [];
                    const max_num_items = Math.max(1, Math.max(profile_a.length, profile_b.length));
                    const row_start_id = profiles_show.length;
                    merge_rows[row_start_id] = max_num_items;
                    let profile_a_latency_total = 0;
                    let profile_b_latency_total = 0;
                    for(let j = 0; j < max_num_items; ++j){
                        let profile_b_item = profile_b[j];
                        let profile_a_item = profile_a[j];
                        profile_a_latency_total += profile_a_item && profile_a_item.Profile && profile_a_item.Profile.averageMs != null ? profile_a_item.Profile.averageMs : 0;
                        profile_b_latency_total += profile_b_item && profile_b_item.Profile && profile_b_item.Profile.averageMs != null ? profile_b_item.Profile.averageMs : 0;
                    }
                    for(let j = 0; j < max_num_items; ++j){
                        let profile_b_item = profile_b[j];
                        let profile_a_item = profile_a[j];
                        const profile_a_onnx_layer_op_types = [];
                        const profile_b_onnx_layer_op_types = [];
                        for(const name of (profile_a_item ? profile_a_item.ONNXNames : [])){
                            if(name in app.node_name_lookup_table)
                                profile_a_onnx_layer_op_types.push(app.node_name_lookup_table[name].raw.optype)
                            else{
                                profile_a_onnx_layer_op_types.push("Unknow");
                            }
                        }

                        for(const name of (profile_b_item ? profile_b_item.ONNXNames : [])){
                            if(name in app.node_name_lookup_table)
                                profile_b_onnx_layer_op_types.push(app.node_name_lookup_table[name].raw.optype)
                            else{
                                profile_b_onnx_layer_op_types.push("Unknow");
                            }
                        }

                        const current_row_index = profiles_show.length;
                        merge_rows_mask.push(j == 0);
                        profiles_show.push({
                            index: current_row_index,
                            node_name: node.raw.name,
                            node_optype: node.raw.optype,
                            group_index: node.raw.idd,
                            inner_index: j,
                            name: profile_a_item ? profile_a_item.Name : "",
                            profile_id: profile_a_item && profile_a_item.Profile ? app.profile_name_to_idd[profile_a_item.Profile.name] : null,
                            latency: profile_a_item && profile_a_item.Profile && profile_a_item.Profile.averageMs != null ? Math.round(profile_a_item.Profile.averageMs * 10000) / 10000 : null,
                            onnx_layers: profile_a_item ? profile_a_item.ONNXNames : [],
                            onnx_layers_string: profile_a_onnx_layer_op_types.join(", "),
                            layer_type: profile_a_item ? profile_a_item.LayerType : "",
                            tactic_name: profile_a_item ? profile_a_item.TacticName : "",
                            latency_total: Math.round(profile_a_latency_total * 10000) / 10000,
                            compared_name: profile_b_item ? profile_b_item.Name : "",
                            compared_latency: profile_b_item && profile_b_item.Profile && profile_b_item.Profile.averageMs != null ? Math.round(profile_b_item.Profile.averageMs * 10000) / 10000 : null,
                            compared_onnx_layers: profile_b_item ? profile_b_item.ONNXNames : [],
                            compared_onnx_layers_string: profile_b_onnx_layer_op_types.join(", "),
                            compared_layer_type: profile_b_item ? profile_b_item.LayerType : "",
                            compared_tactic_name: profile_b_item ? profile_b_item.TacticName : "",
                            compared_latency_total: Math.round(profile_b_latency_total * 10000) / 10000,
                            latency_percentage: Math.round((profile_a_latency_total == 0 ? 0 : (profile_b_latency_total - profile_a_latency_total) / profile_a_latency_total * 100)) + "%",
                            diff_latency: Math.round((profile_a_latency_total - profile_b_latency_total) * 10000) / 10000
                        });
                    }
                    total_profile_latency += profile_a_latency_total;
                    total_compared_latency += profile_b_latency_total;

                    if(this.perlayer_compare_view_plane.total_latency_policy_remove_empty_group && profile_b.length == 0){
                        // skip to compute the latency.
                    }else{
                        total_diff_latency += Math.max(profile_a_latency_total - profile_b_latency_total, 0);
                    }
                }
                this.perlayer_compare_view_plane.merge_rows = merge_rows;
                this.perlayer_compare_view_plane.merge_rows_mask = merge_rows_mask;
                this.perlayer_compare_view_plane.profiles = profiles_show;
                this.perlayer_compare_view_plane.total_diff_latency = total_diff_latency;
                this.perlayer_compare_view_plane.total_profile_latency  = total_profile_latency;
                this.perlayer_compare_view_plane.total_compared_latency = total_compared_latency;

                if(order)
                    this.perlayer_compare_view_plane.current_groups_order = order;
            },
            table_scroll_to(table, row_index){
                const expandedRows = table.bodyWrapper.querySelectorAll(".el-table__expanded-cell");
                const theTableRows = table.bodyWrapper.querySelectorAll(".el-table__body tbody .el-table__row");
                let totalHeight = 0;
                for (let i = 0; i < Math.max(0, row_index - 2); i++) {
                    totalHeight += theTableRows[i].getBoundingClientRect().height;
                    if (expandedRows[i]) {
                        totalHeight += expandedRows[i].getBoundingClientRect().height;
                    }
                }
                table.bodyWrapper.scrollTop = totalHeight;
                table.setCurrentRow(table.tableData[row_index]);
            },
            // compared_view_table_scroll_to(row_index){
            //     const expandedRows = this.$refs.compare_view_table.bodyWrapper.querySelectorAll(".el-table__expanded-cell");
            //     const theTableRows = this.$refs.compare_view_table.bodyWrapper.querySelectorAll(".el-table__body tbody .el-table__row");
            //     let totalHeight = 0;
            //     for (let i = 0; i < Math.max(0, row_index - 2); i++) {
            //         totalHeight += theTableRows[i].getBoundingClientRect().height;
            //         if (expandedRows[i]) {
            //             totalHeight += expandedRows[i].getBoundingClientRect().height;
            //         }
            //     }
            //     this.$refs.compare_view_table.bodyWrapper.scrollTop = totalHeight;
            //     this.$refs.compare_view_table.setCurrentRow(this.$refs.compare_view_table.tableData[row_index]);
            // },
            // profile_view_table_scroll_to(row_index){
            //     const expandedRows = this.$refs.profile_view_table.bodyWrapper.querySelectorAll(".el-table__expanded-cell");
            //     const theTableRows = this.$refs.profile_view_table.bodyWrapper.querySelectorAll(".el-table__body tbody .el-table__row");
            //     let totalHeight = 0;
            //     for (let i = 0; i < Math.max(0, row_index - 2); i++) {
            //         totalHeight += theTableRows[i].getBoundingClientRect().height;
            //         if (expandedRows[i]) {
            //             totalHeight += expandedRows[i].getBoundingClientRect().height;
            //         }
            //     }
            //     this.$refs.profile_view_table.setCurrentRow(this.$refs.profile_view_table.tableData[row_index]);
            //     this.$refs.profile_view_table.bodyWrapper.scrollTop = totalHeight;
            // },
            issuelist_edit_click(){
                if(this.issuelist_view_plane.current_issue == null){
                    this.$message({
                        message: 'No issue is selected',
                        center: true,
                        type: "error"
                    });
                    return;
                }
                const issue = this.issuelist_view_plane.current_issue;
                this.new_issue_dlg.associate_nodes = [...app.selected_nodes];
                this.new_issue_dlg.associate_current_node = app.selected_nodes_current;
                this.new_issue_dlg.show = true;
                this.new_issue_dlg.edited_issue_id = issue.issue_id;
                this.new_issue_dlg.keywords = issue.keywords;
                this.new_issue_dlg.description = issue.description;
                this.new_issue_dlg.current_issue = issue;
                this.new_issue_dlg.mode = "edit";
                this.$nextTick(()=>{
                    this.$refs.new_issue_dlg_keywords.focus();
                    this.$refs.new_issue_dlg_keywords.select();
                });
            },
            issuelist_delete_click(){
                if(this.issuelist_view_plane.current_issue == null){
                    this.$message({
                        message: 'No issue is selected',
                        center: true,
                        type: "error"
                    });
                    return;
                }
                this.$confirm('Are you sure to delete this issue?')
                .then(_ => {
                    const _this = this;
                    var xhr = new XMLHttpRequest();
                    xhr.open('POST', '/delete_issue/' + this.issuelist_view_plane.current_issue.issue_id, true);
                    xhr.setRequestHeader("Content-Type", "application/json;charset=UTF-8");
                    xhr.onload = function (e) {
                        if (xhr.status == 200) {
                            const resp = JSON.parse(xhr.responseText);
                            if(resp.status != "ok"){
                                _this.$message({
                                    message: 'Failed to create issue, error: ' + resp.message,
                                    center: true,
                                    type: "error"
                                });
                                return;
                            }
                            
                            for(let i = 0; i < _this.issuelist_view_plane.issues.length; ++i){
                                if(_this.issuelist_view_plane.issues[i].issue_id == _this.issuelist_view_plane.current_issue.issue_id){
                                    _this.issuelist_view_plane.issues.splice(i, 1);
                                    break;
                                }
                            }
                        }
                    };
                    xhr.send(); 
                }).catch(()=>{});
            },
            issuelist_refresh_issuelist_click(){
                this.refresh_issuelist();
            },
            // issuelist_table_scroll_to(row_index){
            //     const expandedRows = this.$refs.issuelist_view_plane_table.bodyWrapper.querySelectorAll(".el-table__expanded-cell");
            //     const theTableRows = this.$refs.issuelist_view_plane_table.bodyWrapper.querySelectorAll(".el-table__body tbody .el-table__row");
            //     let totalHeight = 0;
            //     for (let i = 0; i < Math.max(0, row_index - 2); i++) {
            //         totalHeight += theTableRows[i].getBoundingClientRect().height;
            //         if (expandedRows[i]) {
            //             totalHeight += expandedRows[i].getBoundingClientRect().height;
            //         }
            //     }
            //     this.$refs.issuelist_view_plane_table.bodyWrapper.scrollTop = totalHeight;
            //     this.$refs.issuelist_view_plane_table.setCurrentRow(this.issuelist_view_plane.issues[row_index]);
            // },
            issuelist_view_plane_to_bottom(select_issue_id=null){
                if(!this.issuelist_view_plane.show){
                    this.issuelist_view_plane.show = true;
                    this.refresh_issuelist(()=>{
                        if(select_issue_id != null){
                            this.$nextTick(()=>{
                                for(let i = 0; i < this.issuelist_view_plane.issues.length; ++i){
                                    if(this.issuelist_view_plane.issues[i].issue_id == select_issue_id){
                                        // this.issuelist_table_scroll_to(i);
                                        this.table_scroll_to(this.$refs.issuelist_view_table, i);
                                        this.issuelist_issues_table_click(this.issuelist_view_plane.issues[i]);
                                        break;
                                    }
                                }
                            });
                        }
                    });
                    return;
                }
                this.issuelist_view_plane.y_pos_bottom_mode = !this.issuelist_view_plane.y_pos_bottom_mode;
                if(this.issuelist_view_plane.y_pos_bottom_mode){
                    this.issuelist_view_plane.old_top = this.$refs.issuelist_view_plane.$el.offsetTop;
                    this.$refs.issuelist_view_plane.$el.style["top"] = "calc(200%)";
                }else{
                    this.$refs.issuelist_view_plane.$el.style["top"] = this.issuelist_view_plane.old_top;
                }
            },
            open_compare_view_plane(){
                if(!this.compare_view_plane.show){
                    this.compare_view_maxissue_click();
                    this.compare_view_plane.show = true;
                    return;
                }
                this.compare_view_plane_to_bottom();
            },
            compare_table_rows_span_method({row, column, rowIndex, columnIndex}){
                if(columnIndex == 4 || columnIndex == 5 || columnIndex == 6 || columnIndex == 7){
                    if(this.compare_view_plane.merge_rows_mask[rowIndex]){
                        if(this.compare_view_plane.merge_rows[rowIndex] > 0)
                            return {rowspan: this.compare_view_plane.merge_rows[rowIndex], colspan: 1};
                    }
                    return {rowspan: 0, colspan: 0};
                }
            },
            perlayer_compare_table_rows_span_method({row, column, rowIndex, columnIndex}){
                if(columnIndex == 0 || columnIndex == 1 || columnIndex == 6 || columnIndex == 7 || columnIndex == 8 || columnIndex == 9){
                    if(this.perlayer_compare_view_plane.merge_rows_mask[rowIndex]){
                        if(this.perlayer_compare_view_plane.merge_rows[rowIndex] > 0)
                            return {rowspan: this.perlayer_compare_view_plane.merge_rows[rowIndex], colspan: 1};
                    }
                    return {rowspan: 0, colspan: 0};
                }
            },
            show_model_info_dlg(close_node_info_dlg=false, close_view_info_dlg=false){
                let inputs_processed = [];
                for(const input of app.inputs){
                    inputs_processed.push({
                        name: input.raw.name,
                        info: app.format_shape(input.raw, true, "shape_left"),
                        node_id: input.raw.idd
                    })
                }
                
                let outputs_processed = [];
                for(const output of app.outputs){
                    outputs_processed.push({
                        name: output.raw.name,
                        info: app.format_shape(output.raw, true, "shape_left"),
                        node_id: output.raw.idd
                    })
                }

                this.model_info_dlg.input = inputs_processed;
                this.model_info_dlg.output = outputs_processed;
                this.model_info_dlg.show = true;
                if(close_view_info_dlg){
                    this.viewinfo_view_plane.show = false;
                }

                if(close_node_info_dlg){
                    this.node_info_dlg.show = false;
                }
            },
            product(shape){
                if(shape){
                    let total_element = 1;
                    for(const d of shape)
                        total_element *= d;
                    return total_element;
                }else{
                    return 0;
                }
            },
            compare_table_scroll_to_profile(profile_id){
                if(this.with_kernel_match_report && this.compare_view_plane.show){
                    for(let i = 0; i < this.$refs.compare_view_table.tableData.length; ++i){
                        const row = this.$refs.compare_view_table.tableData[i];
                        if(row.profile_id == profile_id){
                            // this.compared_view_table_scroll_to(i);
                            this.table_scroll_to(this.$refs.compare_view_table, i);
                            return;
                        }
                    }
                    this.$refs.compare_view_table.setCurrentRow(null);
                }
            },
            perlayer_compare_table_scroll_to_node(node, profile_id){
                if(this.with_kernel_match_report && this.perlayer_compare_view_plane.show){
                    for(let i = 0; i < this.$refs.perlayer_compare_view_table.tableData.length; ++i){
                        const row = this.$refs.perlayer_compare_view_table.tableData[i];
                        if(row.group_index == node.raw.idd && (row.profile_id == null || profile_id == null || row.profile_id == profile_id)){
                            this.table_scroll_to(this.$refs.perlayer_compare_view_table, i);
                            return;
                        }
                    }
                    this.$refs.perlayer_compare_view_table.setCurrentRow(null);
                }
            },
            profile_table_scroll_to_profile(profile_id){
                if(this.with_profile_report && this.profile_view_plane.show){
                    for(let i = 0; i < this.$refs.profile_view_table.tableData.length; ++i){
                        const row = this.$refs.profile_view_table.tableData[i];
                        if(row.idd == profile_id){
                            // this.profile_view_table_scroll_to(i);
                            this.table_scroll_to(this.$refs.profile_view_table, i);
                            return;
                        }
                    }
                    this.$refs.profile_view_table.setCurrentRow(null);
                }
            },
            node_switch_focused_profile(node, current_profile){
                if(node.raw.profiles.length == 0) return;
                const select_nodes = [];
                for(const node_name of current_profile.ONNXNames){
                    if(node_name in app.node_name_lookup_table)
                        select_nodes.push(app.node_name_lookup_table[node_name].raw.idd);
                }

                if(select_nodes.length > 0){
                    app.select_nodes({clear: true, render_style: "major", current_node_id: node.raw.idd}, ...select_nodes);
                }
            },
            scroll_to_related_profile_and_compare_table(node){
                if(node == null){
                    if(this.with_kernel_match_report && this.compare_view_plane.show){
                        this.$refs.compare_view_table.setCurrentRow(null);
                    }

                    if(this.with_kernel_match_report && this.perlayer_compare_view_plane.show){
                        this.$refs.perlayer_compare_view_table.setCurrentRow(null);
                    }

                    if(this.with_profile_report && this.profile_view_plane.show){
                        this.$refs.profile_view_table.setCurrentRow(null);
                    }
                    return;
                }

                if(!this.with_kernel_match_report && !this.with_profile_report)
                    return;
                
                let current_profile = null;
                if(node.raw.profiles.length > 0){
                    if(node.current_focus_profile == null){
                        node.current_focus_profile = 0;
                    }else{
                        node.current_focus_profile = (node.current_focus_profile + 1) % node.raw.profiles.length;
                    }
                    current_profile = node.raw.profiles[node.current_focus_profile];
                }
                
                if(current_profile == null) {
                    this.perlayer_compare_table_scroll_to_node(node, null);
                    return;
                }
                
                const prefer_perf_mode = this.compare_view_plane.show && !this.compare_view_plane.y_pos_bottom_mode || this.perlayer_compare_view_plane.show && !this.perlayer_compare_view_plane.y_pos_bottom_mode || this.profile_view_plane.show && !this.profile_view_plane.y_pos_bottom_mode;
                if(prefer_perf_mode)
                    this.node_switch_focused_profile(node, current_profile);

                this.compare_table_scroll_to_profile(current_profile.Profile.idd);
                this.profile_table_scroll_to_profile(current_profile.Profile.idd);
                this.perlayer_compare_table_scroll_to_node(node, current_profile.Profile.idd);
            },
            expend_profile_details(profile){
                profile.expend = !profile.expend;
            },
            show_tensor_info_dlg(...tensors){
                if(tensors.length == 0) return;

                const producers = [];
                const consumers = [];
                for(const tensor of tensors){
                    if(producers.find((x)=>{return x.node_idd == tensor.from.raw.idd}) == null){
                        producers.push({
                            node_idd: tensor.from.raw.idd,
                            node_name: tensor.from.raw.name,
                            node_optype: tensor.from.raw.optype
                        });
                    }
                    if(consumers.find((x)=>{return x.node_idd == tensor.to.raw.idd}) == null){
                        consumers.push({
                            node_idd: tensor.to.raw.idd,
                            node_name: tensor.to.raw.name,
                            node_optype: tensor.to.raw.optype
                        });
                    }
                }
                this.tensor_info_dlg.current_tensor = tensors[0];
                this.tensor_info_dlg.current_tensor_id = tensors[0].idd;
                this.tensor_info_dlg.producers = producers;
                this.tensor_info_dlg.consumers = consumers;
                this.tensor_info_dlg.show = true;
            },
            show_node_info_dlg(node){
                this.tensor_info_dlg.show = false;
                const metadata = app.metadata[node.raw.optype];
                let current_inputs_value = [];
                const specific_full_expend = node.raw.optype in {Reshape:1, Split:1, Resize:1, Slice:1};
                for(let i = 0; i < (node.raw.input ? node.raw.input.length : 0); ++i){
                    let label = "";
                    if(metadata && i < metadata.inputs.length){
                        label = metadata.inputs[i].name + ":";
                    }else{
                        if(metadata && metadata.inputs.length > 0){
                            label = "";
                        }else{
                            label = "input" + i + ":";
                        }
                    }

                    if(node.raw.input[i] in app.initializer_by_name){
                        const init = app.initializer_by_name[node.raw.input[i]];
                        let expend_value = init.name;
                        if(specific_full_expend){
                            expend_value = "[" + init.data_view.join(", ") + "]";
                        }else if(init.shape.length == 0 || init.shape.length == 1 && init.shape[0] == 1){
                            expend_value = init.data_view[0];
                        }
                        current_inputs_value.push({
                            category: "initializer",
                            shape: init.shape,
                            dtype: init.dtype,
                            name: init.name,
                            expend: false,
                            total_element: this.product(init.shape),
                            data_view: init.data_view,
                            title: expend_value,
                            reference_nodes: [],
                            reference_nodes_optype: [],
                            label: label
                        })
                    }else{
                        const node_ids = [];
                        const node_optypes = [];
                        if(node.raw.input[i] in app.nodes_by_output){
                            for(const item of app.nodes_by_output[node.raw.input[i]]){
                                node_ids.push(item.raw.idd);
                                node_optypes.push(item.raw.optype);
                            }
                        }

                        const shape_info = app.tensor_info_map[node.raw.input[i]];
                        current_inputs_value.push({
                            category: "variable",
                            shape: shape_info ? shape_info.shape : [],
                            dtype: shape_info ? shape_info.dtype : "unknow",
                            name: node.raw.input[i],
                            total_element: this.product(shape_info ? shape_info.shape : null),
                            data_view: [],
                            expend: false,
                            title: node.raw.input[i],
                            reference_nodes: node_ids,
                            reference_nodes_optype: node_optypes,
                            label: label
                        })
                    }
                }
                
                let current_outputs_value = [];
                for(let i = 0; i < (node.raw.output ? node.raw.output.length : 0); ++i){
                    let label = "";
                    if(metadata && i < metadata.outputs.length){
                        label = metadata.outputs[i].name;
                    }else{
                        if(metadata && metadata.outputs.length > 0){
                            label = "";
                        }else{
                            label = "output" + i + ":";
                        }
                    }
                    
                    const node_ids = [];
                    const node_optypes = [];
                    if(node.raw.output[i] in app.nodes_by_input){
                        for(const item of app.nodes_by_input[node.raw.output[i]]){
                            node_ids.push(item.raw.idd);
                            node_optypes.push(item.raw.optype);
                        }
                    }
                    
                    const shape_info = app.tensor_info_map[node.raw.output[i]];
                    current_outputs_value.push({
                        category: "variable",
                        shape: shape_info ? shape_info.shape : [],
                        dtype: shape_info ? shape_info.dtype : "unknow",
                        total_element: this.product(shape_info ? shape_info.shape : null),
                        name: node.raw.output[i],
                        title: node.raw.output[i],
                        label: label,
                        expend: false,
                        data_view: [],
                        reference_nodes: node_ids,
                        reference_nodes_optype: node_optypes,
                    });
                }

                const process_profiles = (profiles)=>{
                    const result = [];
                    for(const profile of profiles){
                        const new_profile = Object.assign({}, profile);
                        new_profile.details_title = JSON.stringify(profile, null, 2); //"Inputs: " + profile.Inputs.map((x)=>{return x["Format/Datatype"]}).join(",") + "  Outputs: " + profile.Outputs.map((x)=>{return x["Format/Datatype"]}).join(",");
                        new_profile.expend = false;
                        result.push(new_profile);
                    }
                    return result;
                };

                this.node_info_dlg.current_inputs_value  = current_inputs_value;
                this.node_info_dlg.current_outputs_value = current_outputs_value;
                this.node_info_dlg.current_profile_list = process_profiles(node.raw.profiles);
                this.node_info_dlg.current_profile_compared_list = Object.assign([], process_profiles(node.raw.profiles_compared));
                this.node_info_dlg.current_node = node;
                this.node_info_dlg.current_node_id = node.raw.idd;
                this.node_info_dlg.show = true;
            },
            expend_node_tensor(tensor){
                tensor.expend = !tensor.expend;
            },
            download_current_selected_nodes_as_a_model(){
                if(this.app.selected_nodes.length == 0){
                    this.$message({
                        message: 'Please select the node first to create a subgraph.',
                        center: true,
                        type: "warning"
                    });
                    return;
                }
                window.open("/subgraph/" + app.view_id + "?layerids=" + this.app.selected_nodes.map((a)=>{return a.raw.idd;}).join(","), "_blank");
            },
            open_perlayer_compare_view(){
                if(!this.perlayer_compare_view_plane.show){
                    // this.perlayer_compare_view_maxissue_click();
                    this.update_perlayer_compare_view_table_data(app.nodes, this.perlayer_compare_view_plane.max_issue_groups_order);
                    this.perlayer_compare_view_plane.show = true;
                    return;
                }
                this.perlayer_compare_view_plane_to_bottom();
            },
            open_fork_view(){
                window.open("/?fork=" + app.view_id, "_blank");
            },
            health_check_issue_click(row){
                if(row == null) return;
                if(row.location_type == "tensor_locate_to_node" || row.location_type == "node" || row.location_type == "subgraph"){
                    app.select_nodes({clear: true, current_node_id:row.location[0]}, ...row.location);
                    app.scroll_to(row.location[0], "smooth", "center");
                }
            },
            open_health_check_view(){
                if(this.health_check_dlg.show){
                    this.health_check_dlg_to_bottom();
                    return;
                }

                this.health_check_dlg.show = true;
                this.health_check_dlg.loading = true;
                const _this = this;
                var xhr = new XMLHttpRequest();
                xhr.open('POST', '/health_check/' + app.view_id, true);
                xhr.setRequestHeader("Content-Type", "application/json;charset=UTF-8");
                xhr.onload = function (e) {
                    _this.health_check_dlg.loading = false;
                    if (xhr.status == 200) {
                        const resp = JSON.parse(xhr.responseText);
                        if(resp.status != "ok"){
                            _this.$message({
                                message: 'Failed to load health check issues, error: ' + resp.message,
                                center: true,
                                type: "error"
                            });
                            return;
                        }
                        _this.health_check_dlg.issues = resp.data.issues;
                    }
                };
                xhr.send();
            },
            function_button_click(func){
                const funcs = {
                    "open-search-view": this.open_search_plane,
                    "open-bird-eye-view": ()=>{this.bird_eye_view.show = !this.bird_eye_view.show;},
                    "open-select-list-view": ()=>{this.select_list_dlg.show = !this.select_list_dlg.show;},
                    "open-profile-view": ()=>{this.open_profile_view_plane();},
                    "open-compare-view": this.open_compare_view_plane,
                    "open-perlayer-compare-view": this.open_perlayer_compare_view,
                    "open-fork-view": this.open_fork_view,
                    // "zoom-out-view": app.zoom_out,  // small
                    // "zoom-in-view": app.zoom_in,   // big
                    "open-issuelist-view": this.issuelist_view_plane_to_bottom,
                    "open-coder-view": this.open_coder_view_plane,
                    "open-viewinfo-view": ()=>{this.viewinfo_view_plane.show = !this.viewinfo_view_plane.show;},
                    "download-current-selected-nodes-as-a-model": this.download_current_selected_nodes_as_a_model,
                    "open-health-check-view": this.open_health_check_view
                };
                funcs[func]();
            },
            scroll_to_and_select_onnx_nodes_by_name(current_node_id, onnx_node_names, render_style = "major"){
                const select_nodes = [];
                for(const node_name of onnx_node_names){
                    if(node_name in app.node_name_lookup_table)
                        select_nodes.push(app.node_name_lookup_table[node_name].raw.idd);
                }

                app.select_nodes({clear: true, render_style: render_style, current_node_id: current_node_id}, ...select_nodes);
                if(current_node_id != null)
                    app.scroll_to(current_node_id, "smooth", "center");
            },
            show_node_info_dlg_with_profile(node, profile_id){
                if(node != null && this.node_info_dlg.show){
                    for(let i = 0; i < node.raw.profiles.length; ++i){
                        const profile = node.raw.profiles[i];
                        if(profile.Profile.idd == profile_id){
                            node.current_focus_profile = i;
                            break;
                        }
                    }
                    this.show_node_info_dlg(node);
                }
            },
            profile_table_row_change(item, column, event){
                event.preventDefault();
                event.stopPropagation();
                if(item){
                    if(item.onnx_layers.length > 0){
                        let current_node = null;
                        for(const onnx_layer_name of item.onnx_layers){
                            current_node = app.node_name_lookup_table[onnx_layer_name];
                            if(current_node != null) break;
                        }
                        this.scroll_to_and_select_onnx_nodes_by_name(current_node ? current_node.raw.idd : null, item.onnx_layers);
                        this.show_node_info_dlg_with_profile(current_node, item.idd);
                    }
                    this.compare_table_scroll_to_profile(item.idd);
                }
            },
            compare_table_row_click_change(item, column, event){
                if(item && column.index != 5 && column.index != 6){
                    const onnx_layers = column.index >= 7 ? item.compared_onnx_layers : item.onnx_layers;
                    let current_node = null;
                    for(const onnx_layer_name of onnx_layers){
                        current_node = app.node_name_lookup_table[onnx_layer_name];
                        if(current_node != null) break;
                    }
                    this.scroll_to_and_select_onnx_nodes_by_name(current_node ? current_node.raw.idd : null, onnx_layers);

                    if(item.profile_id){
                        this.profile_table_scroll_to_profile(item.profile_id);
                        this.show_node_info_dlg_with_profile(current_node, item.profile_id);
                    }
                }
            },
            perlayer_compare_table_row_click_change(item, column, event){
                if(item && column.index != 7 && column.index != 8){
                    const onnx_layers = column.index >= 9 ? item.compared_onnx_layers : item.onnx_layers;
                    const current_node = app.node_mapping[item.group_index];
                    this.scroll_to_and_select_onnx_nodes_by_name(current_node.raw.idd, onnx_layers);

                    if(item.profile_id){
                        this.profile_table_scroll_to_profile(item.profile_id);
                        this.show_node_info_dlg_with_profile(current_node, item.profile_id);
                    }
                }
            },
            async do_search_by_filter(){
                let result_list = [];
                let filter = this.search_plane.filter;
                const strict = filter.startsWith("\"") && filter.endsWith("\"");
                if(strict){
                    filter = filter.substring(1, filter.length - 1);
                }else{
                    filter = filter.toLowerCase();
                }

                for(const node of app.nodes){
                    let matched = (node.raw.idd + "") == filter;
                    if(node.type == "operation"){
                        if(strict){
                            matched = matched || node.raw.name == filter || node.raw.optype == filter;
                        }else{
                            matched = matched || node.raw.name.toLowerCase().indexOf(filter) != -1 || node.raw.optype.toLowerCase().indexOf(filter) != -1;
                        }
                    }else if(node.type == "input" || node.type == "output"){
                        if(strict){
                            matched = matched || node.raw.name == filter;
                        }else{
                            matched = matched || node.raw.name.toLowerCase().indexOf(filter) != -1;
                        }
                    }
                    if(matched){
                        result_list.push({
                            name: node.raw.name,
                            idd: node.raw.idd,
                            optype: node.raw.optype,
                            itemtype: node.type
                        });
                    }
                }

                for(const edge of app.edges){
                    const name = edge.name == null ? "" : edge.name;
                    let matched = name == filter;
                    if(!matched && !strict){
                        matched = name.toLowerCase().indexOf(filter) != -1;
                    }
                    if(matched){
                        result_list.push({
                            name: edge.name,
                            idd: edge.idd,
                            optype: "Tensor",
                            itemtype: "tensor"
                        });
                    }
                }

                if(result_list.length == 0){
                    this.$message({
                        message: 'No any matched node by keywords: ' + this.search_plane.filter,
                        center: true,
                        type: "warning"
                    });
                }
                this.search_plane.data = result_list;
            },
            scroll_to_tensor(tensor){
                if(tensor != null){
                    app.scroll_to_tensor(tensor, "smooth", "center");
                }
            },
            search_result_click(item, column, event){
                if(item){
                    if(item.itemtype == "tensor"){
                        if(event.ctrlKey || event.metaKey){
                            const edges = app.edges_mapping[item.name];
                            if(edges != null){
                                const node_ids = [];
                                for(const edge of edges){
                                    node_ids.push(edge.to.raw.idd);
                                    node_ids.push(edge.from.raw.idd);
                                }
                                app.select_nodes({clear: true}, ...node_ids);
                                app.select_tensors({clear: true}, ...edges);

                                if(edges.length > 0){
                                    app.scroll_to_tensor(edges[0], "smooth", "center");
                                }
                            }
                        }else{
                            const edge = app.edges_idd_mapping[item.idd];
                            if(edge != null){
                                app.select_nodes({clear: true, current_node_id: edge.from.raw.idd}, edge.to.raw.idd, edge.from.raw.idd);
                                app.select_tensors({clear: true}, edge);
                                app.scroll_to_tensor(edge, "smooth", "center");
                            }
                        }
                    }else{
                        app.select_nodes({clear: true, current_node_id:item.idd}, item.idd);
                        app.scroll_to(item.idd, "smooth", "center");
                        this.show_node(app.node_mapping[item.idd]);
                    }
                }
            },
            select_list_item_click(item, preview){
                if(preview){
                    app.preview_nodes(item.idd);
                }else{
                    app.select_nodes({clear: false}, item.idd);
                }
                app.scroll_to(item.idd, "smooth", "center");
                this.show_node(app.node_mapping[item.idd]);
            },
            update_select_dlg(selected_nodes){
                const nodes = [];
                for(const node of selected_nodes){
                    let show_name = node.raw.idd + " [" + node.raw.optype + "] " + node.raw.name;
                    if(node.type == "input" || node.type == "output"){
                        show_name = node.raw.idd + " [" + node.type + "] " + node.raw.name;
                    }
                    nodes.push({
                        name: show_name,
                        raw_name: node.raw.name,
                        idd: node.raw.idd
                    });
                }
                this.select_list_dlg.current_select_nodes = nodes;
            },
            viewinfo_files_button_click(file){
                window.open("/download/" + file.file_id, "_blank");
            },
            viewinfo_copy_input_output_definitions(){
                let input_names = app.inputs.map((x)=>{return "\"" + x.raw.name + "\"";});
                let input_shapes = app.inputs.map((x)=>{return "[" + x.raw.shape.map((y)=>{return typeof(y) == "string" ? "\"" + y + "\"" : y;}).join(",") + "]";});
                let input_dtypes = app.inputs.map((x)=>{return "\"" + x.raw.dtype + "\"";});
                let output_names = app.outputs.map((x)=>{return "\"" + x.raw.name + "\"";});
                let output_shapes = app.outputs.map((x)=>{return "[" + x.raw.shape.map((y)=>{return typeof(y) == "string" ? "\"" + y + "\"" : y;}).join(",") + "]";});
                let output_dtypes = app.outputs.map((x)=>{return "\"" + x.raw.dtype + "\"";});
                let options = [
                    ["input_names", input_names],
                    ["input_shapes", input_shapes],
                    ["input_dtypes", input_dtypes],
                    ["output_names", output_names],
                    ["output_shapes", output_shapes],
                    ["output_dtypes", output_dtypes]
                ];
                options = options.map((x)=>{return x[0] + " = [" + x[1].join(",") + "]";}).join("\n");
                
                const handler = (event)=>{
                    event.clipboardData.setData("text/plain", options);
                    event.preventDefault();
                    document.removeEventListener('copy', handler, true);
                };
                document.addEventListener('copy', handler, true);
                document.execCommand('copy');
                this.$message({
                    message: 'Options have been copied!',
                    center: true,
                    type: "success"
                });
            },
            viewinfo_copy_proposed_force_to_fp16_io_formats(){
                let input_formats = [];
                for(const input of app.inputs){
                    let format = "";
                    let dtype = input.raw.dtype;
                    dtype = dtype.replace("float", "fp");
                    if(dtype == "fp32") dtype = "fp16";
                    if (input.raw.shape && input.raw.shape.length == 4 && input.raw.shape[1] >= 8 && (dtype == "fp16")) {
                        format = `${dtype}:hwc8`;
                    } else {
                        format = `${dtype}:chw`; 
                    }
                    input_formats.push(format);
                }
                
                let output_formats = [];
                for(const output of app.outputs){
                    let format = "";
                    let dtype = output.raw.dtype;
                    dtype = dtype.replace("float", "fp");
                    if(dtype == "fp32") dtype = "fp16";
                    if (output.raw.shape && output.raw.shape.length == 4 && output.raw.shape[1] >= 8 && (dtype == "fp16")) {
                        format = `${dtype}:hwc8`;
                    } else {
                        format = `${dtype}:chw`; 
                    }
                    output_formats.push(format);
                }

                let options = "--inputIOFormats=" + input_formats.join(",") + " --outputIOFormats=" + output_formats.join(",");
                const handler = (event)=>{
                    event.clipboardData.setData("text/plain", options);
                    event.preventDefault();
                    document.removeEventListener('copy', handler, true);
                };
                document.addEventListener('copy', handler, true);
                document.execCommand('copy');
                this.$message({
                    message: 'Options have been copied!',
                    center: true,
                    type: "success"
                });
            },
            viewinfo_copy_trtexec_commandline(){
                const _this = this;
                var xhr = new XMLHttpRequest();
                xhr.open('GET', '/get_logfile_data/' + app.graph.buildlog_meta.file_id + "/command", true);
                xhr.setRequestHeader("Content-Type", "application/json;charset=UTF-8");
                xhr.onload = function (e) {
                    if (xhr.status == 200) {
                        const resp = JSON.parse(xhr.responseText);
                        if(resp.status != "ok"){
                            _this.$message({
                                message: 'Failed to create issue, error: ' + resp.message,
                                center: true,
                                type: "error"
                            });
                            return;
                        }
                        _this.copy_content(resp.data.cmd, "trtexec commandline has been copied!");
                    }
                };
                xhr.send();
            },
            viewinfo_show_related_trex_view(){
                window.open("/related_trex_view/" + app.view_id, "_blank");
            },
            viewinfo_copy_proposed_io_formats(){
                let input_formats = [];
                for(const input of app.inputs){
                    let format = "";
                    let dtype = input.raw.dtype;
                    dtype = dtype.replace("float", "fp");
                    if (input.raw.shape && input.raw.shape.length == 4 && (dtype == "fp16")) {
                        format = `${dtype}:hwc8`;
                    } else {
                        format = `${dtype}:chw`; 
                    }
                    input_formats.push(format);
                }
                
                let output_formats = [];
                for(const output of app.outputs){
                    let format = "";
                    let dtype = output.raw.dtype;
                    dtype = dtype.replace("float", "fp");
                    if (output.raw.shape && output.raw.shape.length == 4 && (dtype == "fp16")) {
                        format = `${dtype}:hwc8`;
                    } else {
                        format = `${dtype}:chw`; 
                    }
                    output_formats.push(format);
                }

                let options = "--inputIOFormats=" + input_formats.join(",") + " --outputIOFormats=" + output_formats.join(",");
                const handler = (event)=>{
                    event.clipboardData.setData("text/plain", options);
                    event.preventDefault();
                    document.removeEventListener('copy', handler, true);
                };
                document.addEventListener('copy', handler, true);
                document.execCommand('copy');
                this.$message({
                    message: 'Options have been copied!',
                    center: true,
                    type: "success"
                });
            },
            draw_link(screen, edge, px, py){
                const points = edge.points;
                const line = document.createElementNS("http://www.w3.org/2000/svg", "path");
        
                if(points.length > 0){
                    if(points[0].x > edge.from.x + edge.from.width * 0.5)
                        points[0].x = edge.from.x + edge.from.width * 0.5;
                    else if(points[0].x < edge.from.x - edge.from.width * 0.5)
                        points[0].x = edge.from.x - edge.from.width * 0.5;
                }
                
                if(points.length % 3 != 0){
                    const num = 3 - (points.length % 3);
                    for(let i = 0; i <  num; ++i){
                        points.push(points[points.length - 1]);
                    }
                }
        
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
                const copyed_line = line.cloneNode(true);
                
                line.classList.add("g-node-link");
                line.setAttribute("marker-end", "url(#arrow)");
                edge.label_text_element = null;
                edge.line_element = line;

                if(edge.name_show){
                    const text = document.createElementNS("http://www.w3.org/2000/svg", "text");
                    text.setAttribute("transform", "translate(" + (edge.x + px) + "," + (edge.y + py - 10) + ")");
                    text.classList.add("g-node-link-shape");
                    text.innerHTML = "<tspan xml:space=\"preserve\" dy=\"1em\" x=\"1\">" + edge.name_show + "</tspan>";
                    edge.label_text_element = text;
                    screen.insertBefore(text, screen.firstChild);
                }

                copyed_line.classList.add("g-node-link-copyed");
                copyed_line.addEventListener("pointerdown", (e)=>{
                    e.preventDefault();
                    e.stopPropagation();
                    this.edge_click(edge);
                });
        
                screen.insertBefore(copyed_line, screen.firstChild);
                screen.insertBefore(line, screen.firstChild);
            },
            edge_click(edge){
                if(this.subgraph_preview_code_metas != null){
                    if(this.subgraph_preview_code_metas.vars_codeline_mapping[edge.name] != null){
                        this.subgraph_preview_locate_to(this.subgraph_preview_code_metas.vars_codeline_mapping[edge.name]);
                    }
                }

                this.subgraph_preview_clean_focused();
                this.subgraph_preview_dlg.selected_edges = [edge];
                for(const edge of this.subgraph_preview_dlg.selected_edges)
                    edge.line_element.classList.add("g-node-link-highlight");
            },
            draw_node(screen, node, px, py){
                const raw = node.raw;
                const title = document.createElementNS("http://www.w3.org/2000/svg", "text");
                const isIOTensor = node.type == "input" || node.type == "output";
                const className  = isIOTensor ? "io-tensor-node" : "operator-node";
                title.innerHTML = isIOTensor ? app.simplify_title(raw.name) : raw.name;
                title.setAttribute("x", 5);
                title.setAttribute("y", 19);
                title.classList.add("g-node-title");
        
                let left = node.x - node.width  * 0.5 + px;
                let top  = node.y - node.height * 0.5 + 1 + py;
        
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
                    if(raw.optype in node_color_formater){
                        background.style["fill"] = node_color_formater[raw.optype].fill;
                    }
                    // background.style["fill"] = raw.optype in node_color_formater ? node_color_formater[raw.optype].fill : "#000";
                }
        
                node.element = body;
                body.appendChild(background);
                body.appendChild(title);
                screen.appendChild(body);
        
                body.addEventListener("pointerdown", (e)=>{
                    e.preventDefault();
                    e.stopPropagation();

                    if(node.type == "input"){
                        this.edge_click(node.outputs[0]);
                        this.subgraph_preview_dlg.selected_nodes = [node];
                        for(const node of this.subgraph_preview_dlg.selected_nodes)
                            node.element.classList.add("g-node-focused-highlight");
                        return;
                    }else if(node.type == "output"){
                        this.edge_click(node.inputs[0]);
                        this.subgraph_preview_dlg.selected_nodes = [node];
                        for(const node of this.subgraph_preview_dlg.selected_nodes)
                            node.element.classList.add("g-node-focused-highlight");
                        return;
                    }

                    this.subgraph_preview_clean_focused();
                    if(this.subgraph_preview_code_metas != null){
                        if(this.subgraph_preview_code_metas.nodes_codeline_mapping[node.raw.idd] != null){
                            this.subgraph_preview_locate_to(this.subgraph_preview_code_metas.nodes_codeline_mapping[node.raw.idd]);
                        }
                    }
                    this.subgraph_preview_dlg.selected_nodes = [node];
                    for(const node of this.subgraph_preview_dlg.selected_nodes)
                        node.element.classList.add("g-node-focused-highlight");

                    // if(e.ctrlKey || e.metaKey)
                    //     this.show_node(node);
                });
            },
            subgraph_preview_clean_focused(){
                if(this.subgraph_preview_dlg.selected_nodes == null) this.subgraph_preview_dlg.selected_nodes = [];
                if(this.subgraph_preview_dlg.selected_edges == null) this.subgraph_preview_dlg.selected_edges = [];
                for(const node of this.subgraph_preview_dlg.selected_nodes)
                    node.element.classList.remove("g-node-focused-highlight");

                for(const edge of this.subgraph_preview_dlg.selected_edges)
                    edge.line_element.classList.remove("g-node-link-highlight");
            },
            getLineHeight(element) {
                const computedStyle = window.getComputedStyle(element);
                return parseFloat(computedStyle.lineHeight);
            },
            getLineYOffsetByHeight(element, lineIndex) {
                const lineHeight = this.getLineHeight(element);
                return lineHeight * lineIndex - 100;
            },
            subgraph_preview_locate_to(line){
                this.subgraph_preview_clean_focused();
                if(line == null || this.subgraph_preview_codeline_offsets == null) return;

                const start = this.subgraph_preview_codeline_offsets[line];
                const end   = this.subgraph_preview_codeline_offsets[line + 1] - 1;

                const textarea = this.$refs.subgraph_preview_dlg_editor.$el.getElementsByTagName("textarea")[0];
                textarea.focus();
                textarea.selectionStart = start;
                textarea.selectionEnd   = end;

                const y = this.getLineYOffsetByHeight(textarea, line);
                textarea.scrollTo({
                    top: y,
                    left: 0,
                    behavior: "instant"
                });
            },
            subgraph_preview_get_editor_current_selection(){
                const textarea = this.$refs.subgraph_preview_dlg_editor.$el.getElementsByTagName("textarea")[0];
                if(textarea == null || this.subgraph_preview_codeline_offsets == null) return null;

                const selection_start = textarea.selectionStart;
                const selection_end   = textarea.selectionEnd;
                let line = null;
                for(let i = 0; i < this.subgraph_preview_codeline_offsets.length - 1; ++i){
                    const start = this.subgraph_preview_codeline_offsets[i];
                    const end   = this.subgraph_preview_codeline_offsets[i + 1];
                    if(selection_start >= start && selection_start < end){
                        line = i;
                        break;
                    }
                }
                return line;
            },
            subgraph_preview_focus_line(target_line, method="code-line-focused"){
                this.subgraph_preview_clean_focused();
                this.subgraph_codeline_focus_mode = method;
                if(this.subgraph_preview_code_metas == null) return;
                const objs = this.subgraph_preview_code_metas.codeline_to_objs[target_line];
                if(objs == null || objs.length == 0) return;

                const select_nodes = [];
                const select_edges = [];
                for(const obj of objs){
                    if(obj.type == "node"){
                        if(this.subgraph_preview_nodes_mapping[obj.node_idd] != null)
                            select_nodes.push(this.subgraph_preview_nodes_mapping[obj.node_idd]);
                    }else if(obj.type == "variable"){
                        if(this.subgraph_preview_edges_mapping[obj.name] != null){
                            select_edges.push(...this.subgraph_preview_edges_mapping[obj.name]);
                        }
                    }
                }

                this.subgraph_preview_dlg.selected_nodes = select_nodes;
                if(select_nodes.length > 0){
                    for(const node of select_nodes)
                        node.element.classList.add("g-node-focused-highlight");

                    const node = select_nodes[0];
                    const roi = this.$refs.subgraph_preview_div.getBoundingClientRect();
                    this.$refs.subgraph_preview_div.scrollTo({
                        left: node.x - roi.width * 0.5 + node.width * 0.5,
                        top: node.y - roi.height * 0.2,
                        behavior: "smooth"
                    });
                }

                this.subgraph_preview_dlg.selected_edges = select_edges;
                if(select_edges.length > 0){
                    for(const edge of select_edges)
                        edge.line_element.classList.add("g-node-link-highlight");

                    if(select_nodes.length == 0){
                        const edge = select_edges[0];
                        this.$refs.subgraph_preview_div.scrollTo({
                            left: edge.x - 100,
                            top: edge.y - 100,
                            behavior: "smooth"
                        });
                    }
                }
            },
            subgraph_preview_dlg_profiling_tab_click(tab_name=null){
                
            },
            subgraph_preview_dlg_profiling_tab_close(tab_name){
                this.$confirm('Delete profiling tab [' + tab_name + "]?", 'Warning', {
                    confirmButtonText: 'OK',
                    cancelButtonText: 'Cancel',
                    type: 'warning'
                }).then(() => {
                    // this.subgraph_preview_dlg.profiling_tabs.splice(i, 1);
                });
            },
            subgraph_preview_dlg_code_tab_close(tab_name){
                this.$confirm('Delete tab [' + tab_name + "]?", 'Warning', {
                    confirmButtonText: 'OK',
                    cancelButtonText: 'Cancel',
                    type: 'warning'
                }).then(() => {
                    this.query("/delete_subgraph_code/" + app.view_id, {"name": tab_name}, null, ()=>{
                        for(let i = 0; i < this.subgraph_preview_dlg.code_tabs.length; ++i){
                            const tab = this.subgraph_preview_dlg.code_tabs[i];
                            if(tab.name == tab_name){
                                if(tab_name == this.subgraph_preview_dlg.current_code_tab.name){
                                    this.subgraph_preview_dlg.current_code_tab = this.subgraph_preview_dlg.code_tabs[i - 1];
                                }
                                this.subgraph_preview_dlg.code_tabs.splice(i, 1);
                                break;
                            }
                        }
                        this.subgraph_preview_compile_current_code();
                        this.$message({
                            type: 'success',
                            message: 'Delete completed.'
                        });
                    });
                });
            },
            subgraph_preview_dlg_code_tab_click(tab_name=null){
                if(tab_name == null){
                    this.subgraph_preview_compile_current_code();
                    return;
                }
                for(const tab of this.subgraph_preview_dlg.code_tabs){
                    if(tab.name == tab_name){
                        this.subgraph_preview_dlg.current_code_tab = tab;
                        // this.subgraph_preview_dlg.current_profiling_tab = "ONNX View";
                        this.refresh_profiling_content();
                        this.subgraph_preview_compile_current_code();
                        break;
                    }
                }
            },
            subgraph_preview_dlg_editor_on_click(e){
                this.subgraph_preview_editor_focus_current(this.subgraph_codeline_focus_mode);
            },
            subgraph_preview_editor_focus_current(method="code-line-focused"){
                const target_line = this.subgraph_preview_get_editor_current_selection();
                if(target_line == null) return;
                this.subgraph_preview_focus_line(target_line, method);
            },
            subgraph_preview_compile_current_code(callback=null){
                const lines = this.subgraph_preview_dlg.current_code_tab.code.split("\n");
                this.setup_subgraph_view(lines, this.subgraph_preview_dlg.current_code_tab.name, callback);
            },
            subgraph_preview_dlg_on_code_change(e){
                this.$nextTick(()=>{
                    this.subgraph_preview_clean_focused();
                    if(this.subgraph_preview_dlg_setup_subgraph_timer != null)
                        clearTimeout(this.subgraph_preview_dlg_setup_subgraph_timer);

                    this.subgraph_preview_dlg_setup_subgraph_timer = setTimeout(()=>{
                        clearTimeout(this.subgraph_preview_dlg_setup_subgraph_timer);
                        this.subgraph_preview_compile_current_code(()=>{
                            this.subgraph_preview_editor_focus_current(this.subgraph_codeline_focus_mode);
                        });
                    }, 100);
                });
            },
            render_subgraph_view(graph, code_metas){
                for(const node of graph.node){
                    for(const attr of node.attrs){
                        attr.value_show = app.convert_to_display_string(attr, node);
                    }
                }

                const tensor_info_map = {};
                if(graph.tensor_info != null){
                    for(const info of graph.tensor_info){
                        tensor_info_map[info.name] = info;
                    }
                }
                const layout_info = app.network_layout(graph, tensor_info_map, true);
                const screen = this.$refs.subgraph_preview_dlg_container;
                screen.replaceChildren();
                let max_x = 0;
                let max_y = 0;
                const px = 100;
                const py = 100;

                for(const node of layout_info.nodes){
                    node.raw.profiles = [];
                    node.raw.profiles_compared = [];
                    this.draw_node(screen, node, px, py);
                    max_x = Math.max(max_x, node.x + node.width * 0.5);
                    max_y = Math.max(max_y, node.y + node.height * 0.5);
                }

                screen.setAttribute("width", max_x + px * 2 + 100);
                screen.setAttribute("height", max_y + py * 2);

                this.subgraph_preview_code_metas = code_metas;
                this.subgraph_preview_nodes = layout_info.nodes;
                this.subgraph_preview_edges = layout_info.edges;
                // node_mapping: node_mapping, nodes_by_output: nodes_by_output, nodes_by_input: nodes_by_input
                this.subgraph_preview_nodes_mapping = layout_info.node_mapping;
                this.subgraph_preview_dlg.selected_edges = [];
                this.subgraph_preview_dlg.selected_nodes = [];

                const edge_mapping = {};
                for(const edge of layout_info.edges){
                    this.draw_link(screen, edge, px, py);

                    if(edge_mapping[edge.name] == null){
                        edge_mapping[edge.name] = [];
                    }
                    edge_mapping[edge.name].push(edge);
                }
                this.subgraph_preview_edges_mapping = edge_mapping;
            },
            setup_subgraph_view(code_lines, name, callback=null){
                this.subgraph_preview_codeline_offsets = code_lines.map((l)=>{return l.length;});
                this.subgraph_preview_codeline_offsets.splice(0, 0, 0);
                for(let i = 1; i < this.subgraph_preview_codeline_offsets.length; ++i){
                    this.subgraph_preview_codeline_offsets[i] += this.subgraph_preview_codeline_offsets[i-1] + 1;
                }

                this.$refs.subgraph_preview_dlg_running_output.innerHTML = "<span style=\"color:#0f0; font-weight: bold\">Compiling...</span>";
                const _this = this;
                var xhr = new XMLHttpRequest();
                xhr.open('POST', '/execute_graph_code/' + app.view_id, true);
                xhr.setRequestHeader("Content-Type", "application/json;charset=UTF-8");
                xhr.onload = function (e) {
                    if (xhr.status == 200) {
                        const resp = JSON.parse(xhr.responseText);
                        if(resp.status != "ok"){
                            _this.$message({
                                message: 'Failed to create issue, error: ' + resp.message,
                                center: true,
                                type: "error"
                            });
                            return;
                        }
                        if(resp.data.running_status == "success"){
                            _this.$refs.subgraph_preview_dlg_running_output.innerText = resp.data.console_output;
                            _this.render_subgraph_view(resp.data.model, resp.data.code_metas);
                        }else{
                            _this.$refs.subgraph_preview_dlg_running_output.innerHTML = "<span class=\"text-error\">Error: " + resp.data.message + "</span><br/><br/><pre>" + resp.data.traceback.replace("\n", "<br/>") + "</pre><br/><br/>" + resp.data.console_output.replace("\n", "<br/>");
                        }
                    }
                    if(callback) callback();
                };
                xhr.send(JSON.stringify({
                    code: code_lines.join("\n"),
                    load_onnx: this.subgraph_preview_dlg.load_onnx,
                    name: name
                }));
            },
            subgraph_preview_download_current_profiling_file(){
                let file_name = this.subgraph_preview_dlg.current_profiling_tab;
                window.open("/get_profiling_file/" + app.view_id + "/" + this.subgraph_preview_dlg.current_code_tab.name + "/" + file_name);
            },
            subgraph_preview_dlg_input_profiling_script_confirm(){
                if(this.subgraph_preview_dlg.profiling_script.code.trim() == ""){
                    this.$message({
                        message: 'Please input the profiling script code',
                        center: true,
                        type: "error"
                    });
                    this.$refs.subgraph_preview_dlg_input_profiling_script.focus();
                    return;
                }

                this.subgraph_preview_dlg.profiling_script.show = false;
                const profiling_script = this.subgraph_preview_dlg.profiling_script.code.trim();
                localStorage.setItem("profiling_script.code", profiling_script);

                const start_time = new Date().getTime();
                this.subgraph_preview_dlg.current_code_tab.profiling_list = [];
                this.subgraph_preview_dlg.current_profiling_tab = "ONNX View";
                this.loadingInstance = window.ELEMENT.Loading.service({ fullscreen: true });
                this.query("/run_profiling_pipeline/" + app.view_id, {
                    "code": this.subgraph_preview_dlg.current_code_tab.code,
                    "name": this.subgraph_preview_dlg.current_code_tab.name,
                    "profiling_script": profiling_script
                }, null, (resp)=>{
                    const update_state = ()=>{
                        this.query("/get_profiling_pipeline_status/" + app.view_id, {"name": this.subgraph_preview_dlg.current_code_tab.name}, null, (resp)=>{
                            if(resp.status == "running" || resp.status == "error"){
                                const elapsed_time = new Date().getTime() - start_time;
                                this.loadingInstance.setText(resp.message + " (" + (elapsed_time / 1000).toFixed(2) + " seconds)");
                            }

                            if(resp.status == "running"){
                                setTimeout(update_state, 100);
                            }else if(resp.status == "error"){
                                this.loadingInstance.close();
                                this.$message({message: "Pipeline failed: " + resp.message, center: true, showClose: true, type: "error"});
                            }else{
                                this.loadingInstance.close();
                                this.subgraph_preview_dlg.current_code_tab.profiling_list = resp.files;
                                this.refresh_profiling_content();
                                this.$message({message: "Profiling completed.", center: true, duration: 1000, showClose: true, type: "success"});
                            }
                        });
                    };
                    setTimeout(update_state, 100);
                });
            },
            subgraph_preview_dlg_input_name_confirm(){
                this.query("/upload_subgraph_code/" + app.view_id, {
                    name: this.subgraph_preview_dlg.input_name,
                    code: this.subgraph_preview_dlg.current_code_tab.code
                }, ()=>{
                    this.subgraph_preview_dlg.input_name_dialog_show = false;
                }, (resp)=>{
                    if(this.subgraph_preview_dlg.input_name != this.subgraph_preview_dlg.current_code_tab.name){
                        this.subgraph_preview_dlg.code_tabs.push({
                            "name": this.subgraph_preview_dlg.input_name,
                            "code": this.subgraph_preview_dlg.current_code_tab.code,
                            "profiling_list": [],
                            "update_time": null,
                            "create_time": null
                        })
                    }
                    this.subgraph_preview_dlg_code_tab_click(this.subgraph_preview_dlg.input_name);
                    this.$message({message: "Code has been saved.", center: true, type: "success"});
                });
            },
            subgraph_preview_current_save(){
                this.subgraph_preview_dlg.input_name = this.subgraph_preview_dlg.current_code_tab.name;
                this.subgraph_preview_dlg.input_name_dialog_show = true;
                this.$nextTick(()=>{
                    this.$refs.subgraph_preview_dlg_input_name.focus();
                    this.$refs.subgraph_preview_dlg_input_name.select();
                });
            },
            subgraph_preview_current_profiling(){
                if(this.subgraph_preview_dlg.profiling_script.code == null || this.subgraph_preview_dlg.profiling_script.code == ""){
                    let code = localStorage.getItem("profiling_script.code");
                    if(code == null){
                        code = [
                            "# WORKDIR /tmp/.__profiling_{TIME_NOW}",
                            "# HOSTIP 10.10.10.10",
                            "# HOSTUSER nvidia",
                            "# HOSTPASSWD nvidia",
                            "# TREX_VIEW outputs/layer_info.json outputs/profile.json",
                            "",
                            "export LD_LIBRARY_PATH=/usr/src/tensorrt/lib:${LD_LIBRARY_PATH}",
                            "",
                            "/usr/src/tensorrt/bin/trtexec --onnx=inputs/model.onnx --profilingVerbosity=detailed --fp16 --int8 --separateProfileRun \\",
                            "    --exportLayerInfo=outputs/layer_info.json --verbose --exportProfile=outputs/profile.json > outputs/build.log 2>&1"
                        ].join("\n");
                        localStorage.setItem("profiling_script.code", code);
                    }
                    this.subgraph_preview_dlg.profiling_script.code = code;
                }
                this.subgraph_preview_dlg.profiling_script.show = true;

                this.$nextTick(()=>{
                    this.$refs.subgraph_preview_dlg_input_profiling_script.focus();
                });
            },
            refresh_profiling_content(){
                for(const file of this.subgraph_preview_dlg.current_code_tab.profiling_list){
                    fetch(
                        "/get_profiling_file/" + app.view_id + "/" + this.subgraph_preview_dlg.current_code_tab.name + "/" + file.name + "?t=" + (new Date().getTime()),
                        {method: "GET"}
                    ).then(async (resp)=>{
                        const element = this.$refs.subgraph_preview_dlg_profiling_tabs.$el.querySelector(".profiling-content-view[name=\"" + file.name + "\"]");
                        if(file.type == "plain-text"){
                            element.innerHTML = "<pre>" + await resp.text() + "</pre>";
                            element.classList.remove("profiling-content-plain-text");
                        }else if(file.type == "svg"){
                            element.classList.add("profiling-content-svgview");
                            element.innerHTML = await resp.text();
                            const svg = element.firstElementChild;
                            svg.setAttribute("width", svg.width.animVal.value * 0.7);
                            svg.setAttribute("height", svg.height.animVal.value * 0.7);
                        }
                    });
                }
            },
            subgraph_preview_attached_file_remove(row){
                this.query("/remove_profiling_file/" + app.view_id, {
                    profile_name: this.subgraph_preview_dlg.current_code_tab.name,
                    file_name: row.name
                }, null, (resp)=>{
                    this.subgraph_preview_dlg.current_code_tab.attached_files = resp.files;
                });
            },
            subgraph_preview_attached_file_upload(){
                this.$refs.file_broswer.accept = "*";
                this.$refs.file_broswer.value  = null;
                this.subgraph_preview_dlg.attached_files.progress = 0;
                this.subgraph_preview_dlg.attached_files.progress_show = false;
                this.$refs.file_broswer.click();

                const _this = this;
                this.$refs.file_broswer.onchange = (e)=>{
                    this.$refs.file_broswer.onchange = null;
                    this.subgraph_preview_dlg.attached_files.progress_show = true;

                    const file = e.target.files[0];
                    const file_name = file.name;
                    var data = new FormData();
                    data.append('file', e.target.files[0]);
                    
                    var xhr = new XMLHttpRequest();
                    xhr.open('POST', '/upload_profiling_file/' + app.view_id, true);
                    xhr.setRequestHeader("profile_name", this.subgraph_preview_dlg.current_code_tab.name);
                    xhr.setRequestHeader("file_name", encodeURIComponent(file_name));
                    xhr.onload = function (e) {
                        if (xhr.status === 200) {
                            const resp = JSON.parse(xhr.responseText);
                            if(resp.status != "ok"){
                                progress.show = false;
                                _this.$message({
                                    message: 'Failed to upload file: ' + file.name + ", error: " + resp.message,
                                    center: true,
                                    type: "error"
                                });
                                return;
                            }
                            _this.subgraph_preview_dlg.attached_files.progress_show = false;
                            _this.subgraph_preview_dlg.current_code_tab.attached_files = resp.data.files;
                        }else{
                            _this.$message({
                                message: 'Failed to upload file: ' + file.name + ", code = " + xhr.status,
                                center: true,
                                type: "error"
                            });
                        }
                    };
                    xhr.upload.onprogress = function(e){
                        if (e.lengthComputable) {
                            var percent_complete = Math.round((e.loaded / e.total) * 100);  
                            _this.subgraph_preview_dlg.attached_files.progress = percent_complete;
                        } 
                    }
                    xhr.send(data);
                };
            },
            subgraph_preview_attached_file_list(){
                this.subgraph_preview_dlg.attached_files.show = true;
            },
            show_subgraph_preview(code_lines){
                this.loadingInstance = window.ELEMENT.Loading.service({ fullscreen: true });
                this.subgraph_preview_dlg.show = true;
                this.query("/get_subgraph_metas/" + app.view_id, null, null, (resp)=>{
                    const tabs = [];
                    let have_current_code = false;
                    for(const key in resp.meta){
                        const content = resp.meta[key];
                        if(key == "current"){
                            if(code_lines.length > 0)
                                content.code = code_lines.join("\n");
                            have_current_code = true;
                        }

                        tabs.push({
                            "name": key, "code": content.code, 
                            "create_time": content.create_time, 
                            "profiling_list": content.profiling_files == null ? [] : content.profiling_files, 
                            "update_time": content.update_time,
                            "attached_files": content.attached_files
                        });
                    }

                    if(!have_current_code){
                        tabs.push({
                            "name": "current", 
                            "code": code_lines.join("\n"), 
                            "create_time": new Date().toISOString(),
                            "profiling_list": [],
                            "update_time": new Date().toISOString(),
                            "attached_files": []
                        });
                    }

                    tabs.sort((a, b)=>{return (a.name == "current" ? 0 : 1) - (b.name == "current" ? 0 : 1);});
                    this.subgraph_preview_dlg.code_tabs = tabs;
                    this.subgraph_preview_dlg_code_tab_click("current");
                }, ()=>{this.loadingInstance.close();});
            },
            switch_theme(){
                if(this.global_theme == "light"){
                    this.global_theme = "dark";
                }else{
                    this.global_theme = "light";
                }
                this.set_theme(this.global_theme);
            },
            switch_highlight_precision(){
                if(this.highlight_precision == "strongly"){
                    this.highlight_precision = "weakly";
                }else{
                    this.highlight_precision = "strongly";
                }

                for(const node of app.nodes){
                    node.element.classList.remove("g-node-precision-strongly");
                    node.element.classList.remove("g-node-precision-strongly-fp16");
                    node.element.classList.remove("g-node-precision-strongly-int8");
                    node.element.classList.remove("g-node-precision-strongly-fp32");
                }

                if(this.highlight_precision == "strongly"){
                    for(const node of app.nodes){
                        for(const input of node.raw.input){
                            if(input in app.initializer_by_name){
                                const initializer = app.initializer_by_name[input];
                                if(initializer.dtype == "float16"){
                                    node.element.classList.add("g-node-precision-strongly");
                                    node.element.classList.add("g-node-precision-strongly-fp16");
                                    break;
                                }else if(initializer.dtype == "int8"){
                                    node.element.classList.add("g-node-precision-strongly");
                                    node.element.classList.add("g-node-precision-strongly-int8");
                                    break;
                                }else if(initializer.dtype == "float32"){
                                    node.element.classList.add("g-node-precision-strongly");
                                    node.element.classList.add("g-node-precision-strongly-fp32");
                                    break;
                                }
                            }
                        }
                    }
                }
            }
        }
        });
        this.app = app;
    }
    
    viewinfo_add_file_meta_if_exists(file_meta, style){
        if(file_meta == null) return;
        this.viewinfo_view_plane.files.push({
            name: file_meta.name,
            file_id: file_meta.file_id,
            size_bytes: file_meta.size_bytes,
            style: style
        });
    }

    set_theme(theme_name){
        document.body.classList.remove("onnx-dark-theme");
        document.body.classList.remove("onnx-light-theme");

        if(theme_name == "light"){
            document.body.classList.add("onnx-light-theme");
        }else{
            document.body.classList.add("onnx-dark-theme");
        }
        localStorage.setItem("global_theme", this.global_theme);
    }

    initialize(){
        const storage_theme = localStorage.getItem("global_theme");
        if(storage_theme != null){
            this.global_theme = storage_theme;
            this.set_theme(this.global_theme);
        }else{
            if(document.body.classList.contains("onnx-light-theme")){
                this.global_theme = "light";
            }else{
                this.global_theme = "dark";
            }
        }

        this.init_table_order();
        const graph = this.app.graph;
        this.selected_nodes_summary.show = graph.profile != null || graph.profile_compared != null;
        this.$nextTick(()=>{
            document.getElementById("select-nodes-info").style["display"] = "unset";
        });

        this.viewinfo_view_plane.basic_info = {
            model_name: graph.model_meta.name,
            size_bytes: graph.model_meta.size_bytes,
            virtual_folder: graph.model_meta.virtual_folder,
            description: graph.model_meta.description,
            create_time: graph.model_meta.create_time,
            file_id: graph.model_meta.file_id
        };

        this.viewinfo_view_plane.performance_summary = graph.view_meta.metadata.performance_summary;
        this.viewinfo_view_plane.performance_summary_compared = graph.view_meta.metadata.performance_summary_compared;

        let total_latency_of_all_nodes = 0;
        if(this.with_profile_report){
            for(const name of Object.keys(graph.profile)){
                const profile = graph.profile[name];
                total_latency_of_all_nodes += profile.averageMs ? profile.averageMs : 0;
            }

            // { headerName: 'Issue-Latency\n(cycle)', valueGetter: p => parseFloat(p.data['smsp__average_warp_latency_per_inst_issued.ratio']), valueFormatter: p => p.value.toFixed(1) },
            // { headerName: 'Long-SB\n(cycle)', valueGetter: p => parseFloat(p.data['smsp__average_warps_issue_stalled_long_scoreboard_per_issue_active.ratio']), valueFormatter: p => p.value.toFixed(1) },
            // { headerName: 'Math\n(cycle)', valueGetter: p => parseFloat(p.data['smsp__average_warps_issue_stalled_math_pipe_throttle_per_issue_active.ratio']), valueFormatter: p => p.value.toFixed(1) },
            // { headerName: 'Wait\n(cycle)', valueGetter: p => parseFloat(p.data['smsp__average_warps_issue_stalled_wait_per_issue_active.ratio']), valueFormatter: p => p.value.toFixed(1) },
            // { headerName: 'MIO\n(cycle)', valueGetter: p => parseFloat(p.data['smsp__average_warps_issue_stalled_mio_throttle_per_issue_active.ratio']), valueFormatter: p => p.value.toFixed(1) },
            // { headerName: 'Barrier\n(cycle)', valueGetter: p => parseFloat(p.data['smsp__average_warps_issue_stalled_barrier_per_issue_active.ratio']), valueFormatter: p => p.value.toFixed(1) },

            if(graph.trtperf != null){

                var ddr_throughput_getter, ddr_rw_getter, ddr_r_getter, ddr_w_getter;
                if (graph.trtperf.columns.includes('gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed')) {
                    ddr_throughput_getter = p => parseFloat(p['gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed']);
                    ddr_rw_getter = p => (parseFloat(p['dram__bytes_read.sum']) + parseFloat(p['dram__bytes_write.sum'])) / 1048576;
                    ddr_r_getter = p => parseFloat(p['dram__bytes_read.sum']) / 1048576;
                    ddr_w_getter = p => parseFloat(p['dram__bytes_write.sum']) / 1048576;
                }
                else {
                    ddr_throughput_getter = p => parseFloat(p['lts__d_sectors_fill_sysmem.avg.pct_of_peak_sustained_elapsed']) + parseFloat(p['lts__t_sectors_aperture_sysmem_op_write.avg.pct_of_peak_sustained_elapsed']);
                    ddr_rw_getter = p => (parseFloat(p['lts__d_sectors_fill_sysmem.sum']) + parseFloat(p['lts__t_sectors_aperture_sysmem_op_write.sum'])) * 32 / 1048576;
                    ddr_r_getter = p => parseFloat(p['lts__d_sectors_fill_sysmem.sum']) * 32 / 1048576;
                    ddr_w_getter = p => parseFloat(p['lts__t_sectors_aperture_sysmem_op_write.sum']) * 32 / 1048576;
                }

                const tofixed_value = (d, n)=>{const factor = Math.pow(10, n); return parseInt(d * factor) / factor;}
                var header = [
                    { headerName: 'SM(%)', propName: "SM", valueGetter: p => parseFloat(p['sm__throughput.avg.pct_of_peak_sustained_elapsed']), valueFormatter: p => tofixed_value(p, 1) },
                    { headerName: 'Tensor-Core(%)', propName: "TensorCore", valueGetter: p => parseFloat(p['sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_active']), valueFormatter: p => tofixed_value(p, 1) },
                    { headerName: 'Memory(%)', propName: "Memory", valueGetter: p => parseFloat(p['gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed']), valueFormatter: p => tofixed_value(p, 1) },
                    { headerName: 'L2-Hit(%)', propName: "L2HitRate", valueGetter: p => parseFloat(p['lts__t_sector_hit_rate.pct']), valueFormatter: p => tofixed_value(p, 1) },
                    { headerName: 'DDR-RW(MB)', propName: "DDRRW", valueGetter: ddr_rw_getter, valueFormatter: p => tofixed_value(p, 1) },
                    { headerName: 'DDR(%)', propName: "DDR", valueGetter: ddr_throughput_getter, valueFormatter: p => tofixed_value(p, 1) },
                    { headerName: 'L2(%)', propName: "L2", valueGetter: p => parseFloat(p['lts__throughput.avg.pct_of_peak_sustained_elapsed']), valueFormatter: p => tofixed_value(p, 1) },
                    { headerName: 'L2-R(%)', propName: "L2R", valueGetter: p => parseFloat(p['lts__t_sectors_srcunit_tex_op_read.avg.pct_of_peak_sustained_elapsed']), valueFormatter: p => tofixed_value(p, 1) },
                    { headerName: 'L2-W(%)', propName: "L2W", valueGetter: p => parseFloat(p['lts__t_sectors_srcunit_tex_op_write.avg.pct_of_peak_sustained_elapsed']), valueFormatter: p => tofixed_value(p, 1) },
                    { headerName: 'DDR-R(MB)', propName: "DDRR", valueGetter: ddr_r_getter, valueFormatter: p => tofixed_value(p, 1) },
                    { headerName: 'DDR-W(MB)', propName: "DDRW", valueGetter: ddr_w_getter, valueFormatter: p => tofixed_value(p, 1) },
                    { headerName: 'Issue-Latency(cycle)', propName: "IssueLatency", valueGetter: p => parseFloat(p['smsp__average_warp_latency_per_inst_issued.ratio']), valueFormatter: p => tofixed_value(p, 1) },
                    { headerName: 'Long-SB(cycle)', propName: "LongSB", valueGetter: p => parseFloat(p['smsp__average_warps_issue_stalled_long_scoreboard_per_issue_active.ratio']), valueFormatter: p => tofixed_value(p, 1) },
                    { headerName: 'Math(cycle)', propName: "Math", valueGetter: p => parseFloat(p['smsp__average_warps_issue_stalled_math_pipe_throttle_per_issue_active.ratio']), valueFormatter: p => tofixed_value(p, 1) },
                    { headerName: 'Wait(cycle)', propName: "Wait", valueGetter: p => parseFloat(p['smsp__average_warps_issue_stalled_wait_per_issue_active.ratio']), valueFormatter: p => tofixed_value(p, 1) },
                    { headerName: 'MIO(cycle)', propName: "MIO", valueGetter: p => parseFloat(p['smsp__average_warps_issue_stalled_mio_throttle_per_issue_active.ratio']), valueFormatter: p => tofixed_value(p, 1) },
                    { headerName: 'Barrier(cycle)', propName: "Barrier", valueGetter: p => parseFloat(p['smsp__average_warps_issue_stalled_barrier_per_issue_active.ratio']), valueFormatter: p => tofixed_value(p, 1) },
                    { headerName: 'Grid(XYZ)', propName: "Grid", valueGetter: p => [p.GridDimX, p.GridDimY, p.GridDimZ], valueFormatter: p=>p.join(", ") },
                    { headerName: 'Block(XYZ)', propName: "Block", valueGetter: p => [p.BlockDimX, p.BlockDimY, p.BlockDimZ], valueFormatter: p=>p.join(", ") },
                    { headerName: 'SMem(Bytes)', propName: "SMem", valueGetter: p => parseInt(p.SharedMemBytes) },
                    { headerName: 'T(ms)', propName: "GPUDuration", valueGetter: p => parseFloat(p['gpu__time_duration.sum']) / 1e6, valueFormatter: p => tofixed_value(p, 3) },
                    { headerName: 'Stream', propName: "Stream", valueGetter: p => p.Stream },
                ]

                const columns = [];
                for(const item of header){
                    columns.push({
                        prop_name: item.propName,
                        show_name: item.headerName,
                        width: 110,
                        value_getter: (data)=>{
                            let value = item.valueGetter(data);
                            if(item.valueFormatter){
                                value = item.valueFormatter(value);
                            }
                            return value;
                        }
                    })
                }
                this.profile_view_plane.trtperf_columns = columns;

                for(const row of graph.trtperf.data){
                    const name = row[1];
                    if(!(name in graph.profile)) continue;
                    const profile = graph.profile[name];
                    const record = {};
                    for(let icol = 0; icol < graph.trtperf.columns.length; ++icol){
                        const col = graph.trtperf.columns[icol];
                        record[col] = row[icol];
                    }

                    for(const col of columns){
                        profile[col.prop_name] = col.value_getter(record); //row[col.column_index];
                    }
                }
            }
        }

        const nodes_grouped_by_optype = {};
        for(const node of this.app.nodes){
            if(node.type != "operation") continue;
            if(!(node.raw.optype in nodes_grouped_by_optype))
                nodes_grouped_by_optype[node.raw.optype] = [];

            nodes_grouped_by_optype[node.raw.optype].push(node);
        }
        
        let layers_summary = [];
        for(const key of Object.keys(nodes_grouped_by_optype)){
            const nodes = nodes_grouped_by_optype[key];
            const profiles_mapping = {};
            for(const node of nodes){
                for(const profile of node.raw.profiles)
                    profiles_mapping[profile.Name] = profile;
            }
            
            let total_latency = 0;
            for(const key of Object.keys(profiles_mapping)){
                const profile = profiles_mapping[key];
                if(profile.Profile == null) continue;
                total_latency += profile.Profile.averageMs;
            }
            
            const compared_profiles_mapping = {};
            for(const node of nodes){
                for(const profile of node.raw.profiles_compared)
                    compared_profiles_mapping[profile.Name] = profile;
            }
            
            let compared_total_latency = 0;
            for(const key of Object.keys(compared_profiles_mapping)){
                const profile = compared_profiles_mapping[key];
                if(profile.Profile == null) continue;
                compared_total_latency += profile.Profile.averageMs;
            }

            layers_summary.push({
                layer_type: key,
                layer_count: nodes.length,
                total_latency: this.with_profile_report ? Math.round(total_latency * 10000) / 10000 : "No Data",
                total_latency_percent: this.with_profile_report ? Math.round(total_latency / total_latency_of_all_nodes * 100 * 10) / 10 + "%" : "No Data",
                total_kernels: Object.keys(profiles_mapping).length,
                compared_total_latency: this.with_kernel_match_report ? Math.round(compared_total_latency * 10000) / 10000 : "No Data",
                compared_total_latency_percent: this.with_kernel_match_report ? Math.round(compared_total_latency / total_latency_of_all_nodes * 100 * 10) / 10 + "%" : "No Data",
                compared_total_kernels: Object.keys(compared_profiles_mapping).length
            });
        }
        layers_summary.sort((a, b)=>{return b.total_latency - a.total_latency;});

        this.viewinfo_view_plane.latency_total  = Math.round(total_latency_of_all_nodes * 100) / 100;
        this.viewinfo_view_plane.layers_summary = layers_summary;
        this.viewinfo_add_file_meta_if_exists(graph.model_meta, "primary");
        this.viewinfo_add_file_meta_if_exists(graph.layerinfo_meta, "primary");
        this.viewinfo_add_file_meta_if_exists(graph.profile_meta, "primary");
        this.viewinfo_add_file_meta_if_exists(graph.buildlog_meta, "primary");
        this.viewinfo_add_file_meta_if_exists(graph.layerinfo_compared_meta, "default");
        this.viewinfo_add_file_meta_if_exists(graph.profile_compared_meta, "default");
        this.viewinfo_add_file_meta_if_exists(graph.buildlog_compared_meta, "default");
    }
};

class BirdEyeView{
    constructor(app){
        this.app = app;
    }

    mount(canvas_id){
        this.canvas_id = canvas_id;
        this.canvas = document.getElementById(canvas_id);
        this.bounding_size = {
            width: this.canvas.width / 2,
            height: this.canvas.height / 2
        };
        
        const padding = 10;
        this.canvas_context = this.canvas.getContext("2d");
        this.sx = (this.canvas.width - padding * 2)  / this.app.size_config.content_width;
        this.sy = (this.canvas.height - padding * 2) / this.app.size_config.content_height;
        this.scale = Math.min(this.sx, this.sy);
        this.canvas_context.fillStyle = "#888";
        this.canvas_cx = this.canvas.width * 0.5;
        this.canvas_cy = this.canvas.height * 0.5;
        for(const node of this.app.nodes){
            if(node.type != "operation") continue;
            const cx = node.x - this.app.size_config.content_width * 0.5;
            const cy = node.y - this.app.size_config.content_height * 0.5;
            const width  = Math.max(node.width * this.scale, 8);
            const height = Math.max(node.height * this.scale, 5);
            const left = cx * this.scale - width * 0.5 + this.canvas_cx;
            const top  = cy * this.scale - height * 0.5 + this.canvas_cy;
            this.canvas_context.fillRect(
                left, top, 
                width, height
            );
        }

        this.canvas_context.fillStyle = "#00f";
        for(const node of this.app.nodes){
            if(node.type == "operation") continue;
            const cx = node.x - this.app.size_config.content_width * 0.5;
            const cy = node.y - this.app.size_config.content_height * 0.5;
            const scaleup = 10;
            const width = Math.max(node.width * scaleup * this.scale, 20);
            const height = Math.max(node.height * scaleup * this.scale, 10);
            const left = cx * this.scale - width * 0.5 + this.canvas_cx;
            const top  = cy * this.scale - height * 0.5 + this.canvas_cy;
            this.canvas_context.fillRect(
                left, top, 
                width, height
            );
        }
        this.cache_image = this.canvas_context.getImageData(0, 0, this.canvas.width, this.canvas.height);

        this.canvas.addEventListener("pointerdown", (e)=>{
            e.preventDefault();
            e.stopPropagation();
            const x = (e.offsetX * 2 - this.canvas_cx) / this.scale + this.app.size_config.content_width * 0.5;
            const y = (e.offsetY * 2 - this.canvas_cy) / this.scale + this.app.size_config.content_height * 0.5;

            let list_nodes = [];
            for(const node of this.app.nodes){
                const addi_dist = node.type == "input" || node.type == "output" ? 0 : 10000;
                const dist = Math.sqrt((x - node.x) * (x - node.x) + (y - node.y) * (y - node.y));
                if(dist * this.scale < 20)
                    list_nodes.push([dist * this.scale + addi_dist, node]);
            }

            if(list_nodes.length > 0){
                list_nodes = list_nodes.sort((a, b)=>{return a[0] - b[0];}).map((x)=>{return x[1];})
                this.app.scroll_to(list_nodes[0].raw.idd, "instant", "center");
                return;
            }
            window.scrollTo({
                left: x * this.app.screen_scale - window.innerWidth * 0.5,
                top: y * this.app.screen_scale - window.innerHeight * 0.5,
                behavior: "instant"
            })
        });
        
        window.addEventListener("scroll", (e)=>{
            const content_width  = this.app.size_config.content_width;
            const content_height = this.app.size_config.content_height;
            const sx = (this.canvas.width - padding * 2)  / content_width;
            const sy = (this.canvas.height - padding * 2) / content_height;
            const scale = Math.min(sx, sy);
            const x = ((window.scrollX + window.innerWidth * 0.5) / this.app.screen_scale - this.app.size_config.content_width * 0.5) * scale + this.canvas_cx;
            const y = ((window.scrollY + window.innerHeight * 0.5) / this.app.screen_scale - this.app.size_config.content_height * 0.5) * scale + this.canvas_cy;
            this.canvas_context.putImageData(this.cache_image, 0, 0);
            this.canvas_context.fillStyle = "#f00a";
            this.canvas_context.lineWidth = "5px";
            this.canvas_context.beginPath();
            this.canvas_context.arc(x, y, 20, 0, 2 * Math.PI);
            this.canvas_context.fill();
        });

        window.addEventListener("mousewheel", (e)=>{
            if(!(e.metaKey || e.ctrlKey)){
                return;
            }
            e.preventDefault();
            e.stopPropagation();
            this.app.zoom(-0.001 * e.deltaY, e.x, e.y);
            // if(e.deltaY < 0)
            //     this.app.zoom(0.15, e.x, e.y);
            // else
            //     this.app.zoom(-0.15, e.x, e.y);
        }, {passive:false});
    }
};

class Graph{
    constructor(nodes, initializer_map){
        this.nodes_alias = {
            "QuantizeLinear": "Q",
            "DequantizeLinear": "DQ",
            "BatchNormalization": "BN",
        };
        this.initializer_map = initializer_map;
        this.nodes           = nodes;
        this.build_consumer_producer(nodes);
        this.fusion_qdq();
        this.topsort();
    }

    fusion_qdq(){
        const removed_nodes = new Set();
        const new_nodes = [];
        for(let i = 0; i < this.nodes.length; ++i){
            const node = this.nodes[i];
            if(node.raw.optype == "QuantizeLinear"){
                const next = this.consumer_list(node.raw.output[0]);
                if(next.length == 1 && next[0].raw.optype == "DequantizeLinear"){
                    if(node.raw.input[1] == next[0].raw.input[1] && node.raw.input[2] == next[0].raw.input[2]){
                        removed_nodes.add(next[0].raw.idd);
                        removed_nodes.add(node.raw.idd);
                        const new_node = Object.assign({}, node);
                        new_node.raw = Object.assign({}, node.raw);
                        new_node.raw.optype = "QDQ";
                        new_node.raw.input = [...node.raw.input];
                        new_node.raw.output = [...next[0].raw.output];
                        new_nodes.push(new_node);
                    }
                }
            }
        }

        for(let i = 0; i < this.nodes.length; ++i){
            const node = this.nodes[i];
            if(this.nodes_alias[node.raw.optype] != null){
                const new_node = Object.assign({}, node);
                new_node.raw = Object.assign({}, node.raw);
                new_node.raw.optype = this.nodes_alias[node.raw.optype];
                new_node.raw.input = [...node.raw.input];
                new_node.raw.output = [...node.raw.output];
                new_nodes.push(new_node);
                removed_nodes.add(node.raw.idd);
            }
        }

        const keeped_nodes = [];
        for(const node of this.nodes){
            if(removed_nodes.has(node.raw.idd)) continue;
            keeped_nodes.push(node);
        }

        for(const node of new_nodes){
            keeped_nodes.push(node);
        }
        this.nodes = keeped_nodes;
        this.build_consumer_producer(keeped_nodes);
    }

    build_consumer_producer(nodes){
        let producers = [];
        let consumers = [];
        for(const node of nodes){
            if(node.type != "operation") continue;
            for(const inp of node.raw.input){
                if(inp == null || inp == "" || this.initializer_map[inp] != null)
                    continue
                if(consumers[inp] == null)
                    consumers[inp] = [];
                consumers[inp].push(node);
            }
            for(const out of node.raw.output){
                if(producers[out] == null)
                    producers[out] = [];
                producers[out].push(node);
            }
        }

        const inputs  = new Set();
        const outputs = new Set();
        for(const node of nodes){
            for(const inp of node.raw.input){
                if(inp == null || inp == "" || this.initializer_map[inp] != null)
                    continue

                if(producers[inp] == null)
                    inputs.add(inp);
            }
            for(const out of node.raw.output){
                if(consumers[out] == null)
                    outputs.add(out);
            }
        }

        this.inputs    = inputs;
        this.outputs   = outputs;
        this.producers = producers;
        this.consumers = consumers;
    }

    topsort(){
        const ready_mapping = new Set();
        const nodes_ordered = [];
        const require_ready = (tensor)=>{
            if(ready_mapping.has(tensor)) return;
            
            if(this.is_constant(tensor)){
                ready_mapping.add(tensor);
                return;
            }

            const producer = this.producer(tensor);
            if(producer == null){
                ready_mapping.add(tensor);
                return;
            }
            
            for(const inp of producer.raw.input)
                require_ready(inp);

            for(const out of producer.raw.output)
                ready_mapping.add(out);

            if(nodes_ordered.indexOf(producer) == -1)
                nodes_ordered.push(producer);
        };

        for(const out of this.outputs)
            require_ready(out);
        
        this.nodes = nodes_ordered;
    }

    is_iotensor(tensor_name){
        return this.inputs.indexOf(tensor_name) != -1 || this.outputs.indexOf(tensor_name) != -1;
    }

    get_constant(tensor_name){
        return this.initializer_map[tensor_name];
    }

    is_constant(tensor_name){
        if(tensor_name == null || tensor_name == "") return true;
        return this.initializer_map[tensor_name] != null ? true : false;
    }

    producer(tensor_name){
        if(this.producers[tensor_name] == null) return null;
        return this.producers[tensor_name][0];
    }

    consumer_list(tensor_name){
        if(this.consumers[tensor_name] == null) return [];
        return this.consumers[tensor_name];
    }
};

class ONNXRender{
    constructor(view_id){
        this.view_id = view_id;
    }

    measure_node_size(name, simplify=true){
        const title = document.createElementNS("http://www.w3.org/2000/svg", "text");
        title.innerHTML = simplify ? this.simplify_title(name) : name;
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

        if(points.length > 0){
            if(points[0].x + px > edge.from.x + edge.from.width * 0.5)
                points[0].x = edge.from.x + edge.from.width * 0.5 - px;
            else if(points[0].x + px < edge.from.x - edge.from.width * 0.5)
                points[0].x = edge.from.x - edge.from.width * 0.5 - px;

            const last = points[points.length - 1];
            if(last.y + py >= edge.to.y - edge.to.height * 0.5 && last.y + py <= edge.to.y + edge.to.height * 0.5){
                if(last.x + px >= edge.to.x && last.x + px < edge.to.x + edge.to.width * 0.5){
                    last.x = edge.to.x + edge.to.width * 0.5 - px + 5;
                }else if(last.x + px < edge.to.x && last.x + px > edge.to.x - edge.to.width * 0.5){
                    last.x = edge.to.x - edge.to.width * 0.5 - px - 5;
                }
            }
        }
        
        if(points.length % 3 != 0){
            const num = 3 - (points.length % 3);
            for(let i = 0; i <  num; ++i){
                points.push(points[points.length - 1]);
            }
        }

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
        const copyed_line = line.cloneNode(true);
        
        line.classList.add("g-node-link");
        line.setAttribute("marker-end", "url(#arrow)");
        edge.label_text_element = null;
        edge.line_element = line;

        copyed_line.classList.add("g-node-link-copyed");
        copyed_line.addEventListener("pointerdown", (e)=>{
            e.preventDefault();
            e.stopPropagation();

            if(e.ctrlKey || e.metaKey){
                const node_ids = [];
                const edges = this.edges_mapping[edge.name];
                if(edges != null){
                    for(const edge of edges){
                        node_ids.push(edge.to.raw.idd);
                        node_ids.push(edge.from.raw.idd);
                    }
                }else{
                    edges = [];
                }
                this.select_nodes({clear: true}, ...node_ids);
                this.select_tensors({clear: true}, ...edges);
                this.addition_ui.show_tensor_info_dlg(...edges);
            }else{
                this.select_nodes({clear: true, current_node_id: edge.to.raw.idd}, edge.to.raw.idd, edge.from.raw.idd);
                this.select_tensors({clear: true}, edge);
                this.addition_ui.show_tensor_info_dlg(edge);
            }
        });

        this.screen.insertBefore(copyed_line, this.screen.firstChild);
        this.screen.insertBefore(line, this.screen.firstChild);
        if(edge.name_show){
            const text = document.createElementNS("http://www.w3.org/2000/svg", "text");
            text.setAttribute("transform", "translate(" + (edge.x + px) + "," + (edge.y + py - 10) + ")");
            text.classList.add("g-node-link-shape");
            text.innerHTML = "<tspan xml:space=\"preserve\" dy=\"1em\" x=\"1\">" + edge.name_show + "</tspan>";
            edge.label_text_element = text;
            this.screen.insertBefore(text, this.screen.firstChild);
        }
    }

    simplify_title(title){
        if(title == null || title == "") return "";
        const p = title.lastIndexOf("/");
        if(p != -1) return title.substring(p + 1);
        return title;
    }

    draw_node(node){
        const raw = node.raw;
        const title = document.createElementNS("http://www.w3.org/2000/svg", "text");
        const isIOTensor = node.type == "input" || node.type == "output";
        const className  = isIOTensor ? "io-tensor-node" : "operator-node";
        title.innerHTML = isIOTensor ? this.simplify_title(raw.name) : raw.optype;
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
            if(raw.optype in node_color_formater){
                background.style["fill"] = node_color_formater[raw.optype].fill;
            }
            // background.style["fill"] = raw.optype in node_color_formater ? node_color_formater[raw.optype].fill : "#000";
        }

        node.element = body;
        body.appendChild(background);
        body.appendChild(title);
        this.screen.appendChild(body);

        body.addEventListener("pointerdown", (e)=>{
            e.preventDefault();
            e.stopPropagation();

            const clear = !(e.metaKey || e.ctrlKey);
            this.addition_ui.show_node(node);
            this.select_nodes({clear: clear, current_node_id:node.raw.idd}, node.raw.idd);
            this.addition_ui.scroll_to_related_profile_and_compare_table(clear ? node : null);
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

    network_layout(graph, tensor_info, show_layername=false){
        const nodes_with_addition_info = [];
        const nodes_by_input  = {};
        const nodes_by_output = {};
        const edges = [];
        const node_mapping = {};
    
        for(const input of graph.input){
            const box = this.measure_node_size(input.name);
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
            let box  = this.measure_node_size(node.optype);
            if(show_layername){
                box  = this.measure_node_size(node.name);
            }
            const instance = {
                width: box.width,
                height: box.height,
                v: node.idd + "",
                parent: null,
                type: "operation",
                raw: node,
                current_focus_profile: 0
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
            const box = this.measure_node_size(output.name);
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
                const items = nodes_by_input[node.raw.name];
                if(items != null){
                    for(const item of items)
                        edge_outs.push({node: item, name: node.raw.name, name_show: this.format_shape(node.raw, null, "shape_left")});
                }
            }else if(node.type == "output"){
                edge_ins = nodes_by_output[node.raw.name];
                if(edge_ins == null) edge_ins = [];
                edge_outs = [{node: node, name: node.raw.name, name_show: this.format_shape(node.raw, null, "shape_left")}];
            }else if(node.type == "operation"){
                edge_ins = [node];
                edge_outs = [];
                for(const out of node.raw.output){
                    if(!(out in nodes_by_input)) continue;
                    for(const item of nodes_by_input[out]){
                        edge_outs.push({node: item, name: out, name_show: tensor_info != null ? this.format_shape(tensor_info[out], null, "shape_left") : ""});
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
        this.selected_tensors = [];
        this.current_preview_nodes = [];
        this.model_input_mapping = {};
        this.model_output_mapping = {};

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
        
        this.profile_name_to_idd = {};
        if(graph.profile != null && graph.layerinfo != null){
            let remove_idxs = [];
            for(let i = 0; i < graph.layerinfo.Layers.length; ++i){
                const layer = graph.layerinfo.Layers[i];
                if(!(layer.Name in graph.profile)){
                    console.log("Removing layer", layer.Name);
                    remove_idxs.push(i);
                }
            }
            for(let i = remove_idxs.length - 1; i >= 0; --i){
                graph.layerinfo.Layers.splice(remove_idxs[i], 1);
            }
        }

        if(graph.profile){
            for(const profile_name of Object.keys(graph.profile)){
                this.profile_name_to_idd[profile_name] = graph.profile[profile_name].idd;
            }
        }

        const layout_info = this.network_layout(graph, this.tensor_info_map);
        this.nodes = layout_info.nodes;
        this.edges = layout_info.edges;
        this.node_mapping = layout_info.node_mapping;
        this.node_name_lookup_table = {};
        this.nodes_by_output = layout_info.nodes_by_output;
        this.nodes_by_input  = layout_info.nodes_by_input;
        this.edges_mapping = {};  // name -> edges
        this.edges_idd_mapping = {};  // idd -> edge
        let edge_idd = 0;
        for(const edge of this.edges){
            edge.idd = edge_idd++;
            this.edges_idd_mapping[edge.idd] = edge;
            if(this.edges_mapping[edge.name] == null){
                this.edges_mapping[edge.name] = [];
            }
            this.edges_mapping[edge.name].push(edge);
        }

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
                const names = layer.ONNXNames == null ? [] : layer.ONNXNames;
                for(const name of names){
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
            this.model_input_mapping[inp.name] = inp;
        }

        for(const out of graph.output){
            this.outputs.push(this.node_mapping[out.idd]);
            this.model_output_mapping[out.name] = out;
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
    
    get_subgraph_codelines_python(remove_qdq=false){
        const graph = new Graph(this.selected_nodes, this.initializer_by_name);
        const network_def = [];
        const definition = [];
        const vars_mapping = {"": "empty"};
        let var_counter = 0;
        let const_counter = 0;
        let out_counter = 0;
        const qdq_var_mapping = {};
        if(graph.nodes.length == 0) return [];

        const make_var = ()=>{
            var_counter += 1;
            return "var" + var_counter;
        };

        const dtype_mapping = {
            "float32": "fp32",
            "float16": "fp16",
            "int8": "i8",
            "int32": "i32",
            "int64": "i64"
        };

        const make_const = (name, optype)=>{
            const value = graph.get_constant(name);
            const is_layout_op = optype == "Reshape" || optype == "Transpose";
            if(is_layout_op){
                return "[" + value.data_view + "]";
            }

            if(value != null && (value.dtype == "int32" || value.dtype == "int64")){
                if(value.shape.length == 0)
                    return value.data_view[0];
                else if(value.shape.length == 1 && value.shape[0] == 1)
                    return "[" + value.data_view[0] + "]";
            }

            const_counter += 1;
            const const_name = "init" + const_counter;
            const dtype = dtype_mapping[value.dtype] == null ? "\"" + value.dtype + "\"" : dtype_mapping[value.dtype];
            definition.push(const_name + " = Tensor(" + dtype + ", [" + value.shape + "], \"" + name + "\")")
            return const_name;
        };

        const make_out = ()=>{
            out_counter += 1;
            return "out" + out_counter;
        };

        for(const node of graph.nodes){
            if(node.type != "operation") continue;

            const input_variables = [];
            for(const inp of node.raw.input){
                if(inp == null || inp == ""){
                    input_variables.push("\"\"");
                    continue;
                }

                if(graph.is_constant(inp)){
                    if(vars_mapping[inp] == null){
                        vars_mapping[inp] = make_const(inp, node.raw.optype);
                    }
                    input_variables.push(vars_mapping[inp]);
                }else{
                    if(vars_mapping[inp] == null){
                        vars_mapping[inp] = make_var();
                    }
                    input_variables.push(vars_mapping[inp]);
                }
            }

            for(const attr of node.raw.attrs){
                let value = attr.value;
                if(attr.dtype == "float" && value != null && ("" + value).indexOf(".") == -1){
                    value = value + ".0";
                }
                if(attr.dtype == "int_array" || attr.dtype == "float_array"){
                    value = "[" + value.join(",") + "]";
                    if(node.raw.optype == "Conv" || node.raw.optype == "ConvTranspose"){
                        let have_diff_value = false;
                        let val0 = attr.value[0];
                        for(const v of attr.value){
                            if(v != val0){
                                have_diff_value = true;
                                break;
                            }
                        }

                        if(!have_diff_value){
                            value = val0;
                        }

                        if(attr.name == "dilations" && value == 1 || 
                            attr.name == "pads" && value == 0 ||
                            attr.name == "strides" && value == 1){
                            continue;
                        }
                    }
                }else if(attr.dtype == "string"){
                    value = "\"" + value + "\"";
                }else{
                    if(node.raw.optype == "Conv" || node.raw.optype == "ConvTranspose"){
                        if(attr.name == "group" && value == 1)
                            continue;
                    }
                }
                input_variables.push(attr.name + "=" + value);
            }

            const output_variables = [];
            for(const out of node.raw.output){
                if(vars_mapping[out] == null){
                    vars_mapping[out] = make_out();
                }

                if(output_variables.indexOf(vars_mapping[out]) == -1)
                    output_variables.push(vars_mapping[out]);
            }

            if(remove_qdq){
                if((node.raw.optype == "QDQ" || node.raw.optype == "QuantizeLinear" || node.raw.optype == "DequantizeLinear" || node.raw.optype == "Q" || node.raw.optype == "DQ")){
                    qdq_var_mapping[output_variables[0]] = input_variables.length > 0 ? input_variables[0] : "###empty###";
                    continue;
                }else{
                    const splice_indexs = [];
                    for(let i = 0; i < input_variables.length; ++i){
                        const input = input_variables[i];
                        if(qdq_var_mapping[input] != null){
                            if(qdq_var_mapping[input] == "###empty###"){
                                splice_indexs.push(i);
                            }else{
                                input_variables[i] = qdq_var_mapping[input];
                            }
                        }
                    }

                    for(let i = splice_indexs.length - 1; i >= 0; --i){
                        input_variables.splice(splice_indexs[i], 1);
                    }
                }
            }

            const output_string = output_variables.join(", ");
            let input_string  = input_variables.join(", ");
            if(node.raw.optype == "Split"){
                network_def.push(output_string + " = " + node.raw.optype + "(" + input_string + ", outputs=" + output_variables.length + ")");    
            }else if(!knows_keywords.has(node.raw.optype)){
                if(output_variables.length > 1){
                    input_variables.push("outputs=" + output_variables.length);
                }
                if(node.raw.domain != ""){
                    input_variables.push("domain=\"" + node.raw.domain + "\"");
                }
                input_string = input_variables.join(", ");
                network_def.push(output_string + " = layer(\"" + node.raw.optype + "\", " + input_string + ")");
            }else{
                network_def.push(output_string + " = " + node.raw.optype + "(" + input_string + ")");
            }
        }

        definition.push("");
        for(const input of graph.inputs){
            const shape_info = this.tensor_info_map[input];
            // const name = vars_mapping[input];
            const name = input;
            if(shape_info == null){
                definition.push(vars_mapping[input] + " = Input(fp32, [1], \"" + name + "\")");
            }else{
                const shape_info_strings = [];
                for(const s of shape_info.shape){
                    if(typeof(s) == "string")
                        shape_info_strings.push("\"" + s + "\"");
                    else
                        shape_info_strings.push(s);
                }
                const dtype = dtype_mapping[shape_info.dtype] == null ? "\"" + shape_info.dtype + "\"" : dtype_mapping[shape_info.dtype];
                definition.push(vars_mapping[input] + " = Input(" + dtype + ", [" + shape_info_strings.join(",") + "], \"" + name + "\")");
            }
        }

        for(const output of graph.outputs){
            let outname = vars_mapping[output];
            if(remove_qdq){
                if(qdq_var_mapping[outname] != null){
                    outname = qdq_var_mapping[outname];
                }
            }

            const shape_info = this.tensor_info_map[output];
            if(shape_info == null){
                network_def.push("Output(" + outname + ", fp32, [1])");
            }else{
                const shape_info_strings = [];
                for(const s of shape_info.shape){
                    if(typeof(s) == "string")
                        shape_info_strings.push("\"" + s + "\"");
                    else
                        shape_info_strings.push(s);
                }
                const dtype = dtype_mapping[shape_info.dtype] == null ? "\"" + shape_info.dtype + "\"" : dtype_mapping[shape_info.dtype];
                network_def.push("Output(" + outname + ", " + dtype + ", [" + shape_info_strings.join(",") + "], \"" + output + "\")");
            }
        }
        return definition.concat(network_def);
    }

    random_choice(array){
        const i = parseInt(Math.random() * (array.length - 1));
        return array[i];
    }

    show_subgraph_preview(){
        const codelines = this.get_subgraph_codelines_python(false);
        this.addition_ui.show_subgraph_preview(codelines);
    }

    copy_selected_nodes_name_to_clipboard(){
        let code = this.selected_nodes.map((a)=>{return a.raw.name;});
        code = "[" + code.map((a)=>{return "\"" + a + "\"";}).join(", ") + "]";
        this.addition_ui.copy_content(code, "Selected node names have been copied!");
    }

    automatic_connect_nodes_to_model_input(){
        if(this.selected_nodes.length == 0) return;

        const iter_to_model_input = (node, container)=>{
            if(container.has(node.raw.idd)) return;
            container.add(node.raw.idd);
            if(node.type == "input") return;
            for(const input of node.raw.input){
                const producers = this.nodes_by_output[input];
                if(producers == null || producers.length == 0) continue;
                for(const producer of producers){
                    iter_to_model_input(producer, container);
                }
            }
        };

        const nodeids_connected_to_model_input = new Set();
        for(const n of this.selected_nodes) {
            iter_to_model_input(n, nodeids_connected_to_model_input);
        }
        this.select_nodes({clear: true, current_node_id: this.selected_nodes_current ? this.selected_nodes_current.raw.idd : null}, ...nodeids_connected_to_model_input.keys());
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

    clean_selected_tensors(){
        for(const tensor of this.selected_tensors){
            tensor.line_element.classList.remove("g-node-link-selected");
        }
        this.selected_tensors = [];
    }

    select_tensors(option, ...edges){
        for(const tensor of this.selected_tensors){
            tensor.line_element.classList.remove("g-node-link-selected");
        }
        
        if(option.clear){
            this.selected_tensors = [];
        }

        for(const edge of edges){
            this.selected_tensors.push(edge);
            edge.line_element.classList.add("g-node-link-selected");
        }
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
            this.clean_selected_tensors();
        }

        let not_in_ids = true;
        for(const idd of node_ids){
            if(idd == option.current_node_id){
                not_in_ids = false;
            }
        }

        if(not_in_ids){
            node_ids.push(option.current_node_id);
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

        this.addition_ui.selected_nodes_summary.selected.total_nodes = this.selected_nodes.length;
        this.addition_ui.selected_nodes_summary.selected.major_latency = select_nodes_total_latency;
        this.addition_ui.selected_nodes_summary.selected.compared_latency = select_nodes_total_compared_latency;
        this.addition_ui.update_select_dlg(this.selected_nodes);
        let url = [];
        if(selected_node_ids.length > 0)
            url.push("select=" + selected_node_ids.join(","));
        
        if(current_node)
            url.push("current=" + current_node.raw.idd);

        if(option.issue_id != null)
            url.push("issue=" + option.issue_id);
        
        if(url.length > 0){
            location.hash = url.join("&");
        }else{
            history.pushState("", document.title, window.location.pathname
                + window.location.search);
        }
    }

    scroll_to_tensor(tensor, behavior="instant", pos="top"){
        const topgap = pos == "top" ? Math.max(125, window.innerHeight * 0.2) : window.innerHeight * 0.25;
        window.scrollTo({
            left: tensor.x * this.screen_scale - window.innerWidth * 0.5,
            top:  tensor.y * this.screen_scale - topgap,
            behavior: behavior
        });
    }

    scroll_to(node_id, behavior="instant", pos="top"){
        if(!(node_id in this.node_mapping)) return;
        const node   = this.node_mapping[node_id];
        const topgap = pos == "top" ? Math.max(125, window.innerHeight * 0.2) : window.innerHeight * 0.25;
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
                this.addition_ui.selected_nodes_summary.realtime.show = true;
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
                this.addition_ui.selected_nodes_summary.realtime.total_nodes = select_nodes.length;
                this.addition_ui.selected_nodes_summary.realtime.major_latency = select_nodes_total_latency;
                this.addition_ui.selected_nodes_summary.realtime.compared_latency = select_nodes_total_compared_latency;
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
                this.addition_ui.selected_nodes_summary.realtime.show = false;
            }
            this.screen.style["cursor"] = "auto";
            this.screen.releasePointerCapture(e.pointerId);
            screen_pointer.down = false;
        });
    }

    setup_global_keys_event(){
        const _this = this;
        const allow_keys = {
            "f": ()=>{_this.addition_ui.open_search_plane();},
            "e": ()=>{_this.addition_ui.bird_eye_view.show = !_this.addition_ui.bird_eye_view.show;},
            "g": ()=>{_this.addition_ui.select_list_dlg.show = !_this.addition_ui.select_list_dlg.show;},
            "u": ()=>{_this.addition_ui.open_profile_view_plane();},
            "i": ()=>{_this.addition_ui.open_compare_view_plane();},
            "q": ()=>{_this.addition_ui.compare_view_plane_to_bottom();},
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
                    if(e.target.localName == "input" || e.target.localName == "textarea" || e.target.classList.contains("text-editor")) return;
                    e.preventDefault();
                    e.stopPropagation();
                    this.select_nodes({clear: true});
                }else if(key == 'f'){
                    if(e.target.localName == "input" || e.target.localName == "textarea" || e.target.classList.contains("text-editor")) return;
                    e.preventDefault();
                    e.stopPropagation();
                    this.automatic_complete_select_nodes();
                }else if(key == 'g'){
                    if(e.target.localName == "input" || e.target.localName == "textarea" || e.target.classList.contains("text-editor")) return;
                    e.preventDefault();
                    e.stopPropagation();
                    this.automatic_connect_nodes_to_model_input();
                }else if(key == 'y'){
                    if(e.target.localName == "input" || e.target.localName == "textarea" || e.target.classList.contains("text-editor")) return;
                    e.preventDefault();
                    e.stopPropagation();
                    this.copy_selected_nodes_name_to_clipboard();
                }else if(key == 'u'){
                    if(e.target.localName == "input" || e.target.localName == "textarea" || e.target.classList.contains("text-editor")) return;
                    e.preventDefault();
                    e.stopPropagation();
                    this.show_subgraph_preview();
                }else if(key == "escape"){
                    const _this = this;
                    const actions = [
                        [this.addition_ui.node_info_dlg.show, ()=>{_this.addition_ui.node_info_dlg.show = false;}],
                        [this.addition_ui.model_info_dlg.show, ()=>{_this.addition_ui.model_info_dlg.show = false;}],
                        [this.addition_ui.tensor_info_dlg.show, ()=>{_this.addition_ui.tensor_info_dlg.show = false;}],
                        [this.addition_ui.select_list_dlg.show, ()=>{_this.addition_ui.select_list_dlg.show = false;}],
                        [this.addition_ui.search_plane.show, ()=>{_this.addition_ui.search_plane.show = false;}],
                        [this.addition_ui.bird_eye_view.show, ()=>{_this.addition_ui.bird_eye_view.show = false;}],
                        [this.addition_ui.viewinfo_view_plane.show, ()=>{_this.addition_ui.viewinfo_view_plane.show = false;}],
                        [this.addition_ui.issuelist_view_plane.show && !this.addition_ui.issuelist_view_plane.y_pos_bottom_mode, ()=>{_this.addition_ui.issuelist_view_plane_to_bottom();}],
                        [this.addition_ui.profile_view_plane.show && !this.addition_ui.profile_view_plane.y_pos_bottom_mode, ()=>{_this.addition_ui.profile_view_plane_to_bottom();}],
                        [this.addition_ui.compare_view_plane.show && !this.addition_ui.compare_view_plane.y_pos_bottom_mode, ()=>{_this.addition_ui.compare_view_plane_to_bottom();}],
                        [this.addition_ui.perlayer_compare_view_plane.show && !this.addition_ui.perlayer_compare_view_plane.y_pos_bottom_mode, ()=>{_this.addition_ui.perlayer_compare_view_plane_to_bottom();}],
                        [this.addition_ui.coder_view_plane.show && !this.addition_ui.coder_view_plane.y_pos_bottom_mode, ()=>{_this.addition_ui.coder_view_plane_to_bottom();}],
                        [this.addition_ui.health_check_dlg.show && !this.addition_ui.health_check_plane.y_pos_bottom_mode, ()=>{_this.addition_ui.health_check_dlg_to_bottom();}],
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

        if("subgraphview" in hash_param && node_ids.length > 0){
            this.show_subgraph_preview();
        }
    }

    zoom(scale, cx, cy){
        this.screen_scale_events.push([scale, cx, cy]);

        const zoom_implement = ()=>{
            let events = [];
            events = [this.screen_scale_events, this.screen_scale_events = events][0];

            if(events.length == 0) return;
            let new_width = 0;
            let new_height = 0;
            let ax = window.scrollX;
            let ay = window.scrollY;
            const sb = this.screen.getBoundingClientRect();
            const pb = this.screen.parentElement.getBoundingClientRect();
            const diff_x = sb.left - pb.left;
            const diff_y = sb.top - pb.top;

            for(const event of events){
                const [scale, cx, cy] = event;
                const old_scale = this.screen_scale;
                this.screen_scale = Math.max(0.05, Math.min(this.screen_scale + scale, 1.5));
                ax = (cx + ax - diff_x) / old_scale * this.screen_scale - cx + diff_x;
                ay = (cy + ay - diff_y) / old_scale * this.screen_scale - cy + diff_y;
                new_width  = this.size_config.content_width * this.screen_scale;
                new_height = this.size_config.content_height * this.screen_scale;
            }
            this.screen.setAttribute("width", new_width);
            this.screen.setAttribute("height", new_height);
            window.scrollTo({left: ax, top: ay});
        };

        if(this.zoom_timer)
            clearTimeout(this.zoom_timer);

        this.zoom_timer = setTimeout(zoom_implement, 10);
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
        document.title = graph.model_meta.name;
        this.screen_scale = 1.0;
        this.screen_scale_events = [];
        this.render_graph(graph);
        this.addition_ui.with_kernel_match_report = graph.model_meta.with_kernel_match_report;
        this.addition_ui.with_profile_report      = graph.profile != null && graph.layerinfo != null;
        this.addition_ui.initialize();
        this.bev.mount("bev-canvas");
        this.setup_pointer_move_event();
        this.loadingInstance.close();
        this.setup_default_view();
        this.setup_global_keys_event();
    }

    mount(screen_element_id, ui_element_id, graph_url){
        this.addition_ui = new AdditionUI(this, ui_element_id);
        this.bev         = new BirdEyeView(this);
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

const render = new ONNXRender(meta_mapping["view-id"]);
render.mount("screen", "addition-ui", meta_mapping["onnx-json-url"]);