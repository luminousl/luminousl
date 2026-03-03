class UI extends Vue{
    constructor(ui_element_id){
        super({
        el: "#" + ui_element_id,
        data: {
            setup_file_dlg: {
                show: false,
                upload: {
                    show: false,
                    progress: 0,
                    title: "",
                    file_id: null
                },
                target: {},
                tab_value: "FromURL",
                from_file_id: "",
                from_file_url: "",
                from_ssh_path: "",
                from_ssh_ip: "",
                from_ssh_password: "",
                file_name: "",
                folder: "default",
                description: "",
                callback: null,
                projection_code: ""
            },
            setup_path_projection_dlg:{
                show: false,
                projection_code: "",
                callback: null
            },
            create_view_dlg: {
                show: false,
                folder: "default",
                description: "",
                view_name: "unnamed",
                mode: ""
            },
            btn_progress: {
                onnx_profile_layers_from_logfile: {
                    show: false,
                    progress: 0,
                    title: "Generate from Build.log",
                    file_id: null
                },
                onnx_model: {
                    show: false,
                    progress: 0,
                    title: "Model.onnx",
                    file_id: null
                },
                onnx_layerinfo: {
                    show: false,
                    progress: 0,
                    title: "LayerInfo.json",
                    file_id: null
                },
                onnx_profile: {
                    show: false,
                    progress: 0,
                    title: "Profile.json",
                    file_id: null
                },
                trtperf: {
                    show: false,
                    progress: 0,
                    title: "TRTPerf.csv",
                    file_id: null
                },
                onnx_layerinfo_compared: {
                    show: false,
                    progress: 0,
                    title: "LayerInfo.json",
                    file_id: null
                },
                onnx_profile_compared: {
                    show: false,
                    progress: 0,
                    title: "Profile.json",
                    file_id: null
                },
                trtperf_compared: {
                    show: false,
                    progress: 0,
                    title: "TRTPerf.csv",
                    file_id: null
                },
                onnx_profile_layers_from_logfile_compared: {
                    show: false,
                    progress: 0,
                    title: "Generate from Build.log",
                    file_id: null
                },
                trex_layerinfo: {
                    show: false,
                    progress: 0,
                    title: "LayerInfo.json",
                    file_id: null
                },
                trex_profile: {
                    show: false,
                    progress: 0,
                    title: "Profile.json",
                    file_id: null
                },
                trex_profile_layers_from_logfile: {
                    show: false,
                    progress: 0,
                    title: "Generate from Build.log",
                    file_id: null
                },
            },
            with_kernel_match_report: false
        },
        mounted(){
            this.$nextTick(()=>{
                this.initialize();
            });
        },
        methods: {
            parse_arguments(){
                if(!(location.search && location.search.length > 1))
                    return {};
        
                const result = {};
                const vars   = location.search.substring(1).split("&");
                for(const item of vars){
                    const p = item.indexOf("=");
                    if(p != -1){
                        const key   = item.substring(0, p);
                        const value = item.substring(p + 1);
                        result[key] = value;
                    }
                }
                return result;
            },
            initialize(){
                const params = this.parse_arguments();
                if("fork" in params){
                    this.initialize_fork_view(params.fork);
                }
                
                const last_operation_page = localStorage.getItem("last_operation_page");
                if(last_operation_page != null && last_operation_page != ""){
                    this.setup_file_dlg.tab_value = last_operation_page;
                }
                this.setup_file_dlg.from_ssh_ip = localStorage.getItem("create_view_from_ssh_ip");
                this.setup_file_dlg.from_ssh_password = localStorage.getItem("create_view_from_ssh_password");
                this.setup_path_projection_dlg.projection_code = localStorage.getItem("setup_path_projection_code");
                if(this.setup_path_projection_dlg.projection_code == "" || this.setup_path_projection_dlg.projection_code == null){
                    this.setup_path_projection_dlg.projection_code = "return path";
                }
            },
            fill_progress_info(target, file_data){
                if(file_data == null) return;

                target.show = false;
                target.progress = 100;
                target.title = file_data.name.substring(0, Math.min(32, file_data.name.length)) + (file_data.name.length > 32 ? "..." : "");
                target.file_id = file_data.file_id;
            },
            initialize_fork_view(view_id){
                const _this = this;
                var xhr = new XMLHttpRequest();
                xhr.open('POST', '/get_view_info/' + view_id, true);
                xhr.setRequestHeader("Content-Type", "application/json;charset=UTF-8");
                xhr.onload = function (e) {
                    if (xhr.status === 200) {
                        const resp = JSON.parse(xhr.responseText);
                        if(resp.status != "ok"){
                            _this.$message({
                                message: 'Failed to get view info, error: ' + resp.message,
                                center: true,
                                type: "error"
                            });
                            return;
                        }
                        const view = resp.data;
                        _this.create_view_dlg.view_name = view.name;
                        _this.create_view_dlg.folder = view.virtual_folder;
                        if(view.meta_data != null){
                            _this.create_view_dlg.description = view.meta_data.description;
                        }
                        if(view.view_type == "onnx"){
                            _this.fill_progress_info(_this.btn_progress.onnx_model, view.meta_data.model);
                            _this.fill_progress_info(_this.btn_progress.onnx_layerinfo, view.meta_data.layerinfo);
                            _this.fill_progress_info(_this.btn_progress.onnx_profile, view.meta_data.profile);
                            _this.fill_progress_info(_this.btn_progress.trtperf, view.meta_data.trtperf);
                        }else if(view.view_type == "onnx_with_kernel_match"){
                            _this.with_kernel_match_report = true;
                            _this.fill_progress_info(_this.btn_progress.onnx_model, view.meta_data.model);
                            _this.fill_progress_info(_this.btn_progress.onnx_layerinfo, view.meta_data.layerinfo);
                            _this.fill_progress_info(_this.btn_progress.onnx_profile, view.meta_data.profile);
                            _this.fill_progress_info(_this.btn_progress.trtperf, view.meta_data.trtperf);
                            _this.fill_progress_info(_this.btn_progress.onnx_layerinfo_compared, view.meta_data.layerinfo_compared);
                            _this.fill_progress_info(_this.btn_progress.onnx_profile_compared, view.meta_data.profile_compared);
                            _this.fill_progress_info(_this.btn_progress.trtperf_compared, view.meta_data.trtperf_compared);
                        }else if(view.view_type == "trex"){
                            _this.fill_progress_info(_this.btn_progress.trex_layerinfo, view.meta_data.layerinfo);
                            _this.fill_progress_info(_this.btn_progress.trex_profile, view.meta_data.profile);
                        }
                    }
                };
                xhr.send();
            },
            create_file_by_upload(file, progress, callback, file_name="", folder="default", description=""){
                const _this = this;
                var data = new FormData();
                data.append('file', file);
                progress.title = file.name.substring(0, Math.min(16, file.name.length)) + (file.name.length > 16 ? "..." : "");
                progress.progress = 0;
                progress.show = true;
                if(file_name == null || file_name == "")
                    file_name = file.name;
                
                var xhr = new XMLHttpRequest();
                xhr.open('POST', '/create_file_by_upload', true);
                xhr.setRequestHeader("file_name", encodeURIComponent(file_name));
                xhr.setRequestHeader("folder", encodeURIComponent(folder));
                xhr.setRequestHeader("description", encodeURIComponent(description));
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
                        progress.show = false;
                        progress.file_id = resp.data.file.file_id;
                        progress.title = _this.format_progress_title(file.name, resp.data.file.file_id);
                        if(callback != null){
                            const func = callback[0];
                            const args = callback.slice(1);
                            func(resp.data.file, progress, ...args);
                        }
                    }
                };
                xhr.upload.onprogress = function(e){
                    if (e.lengthComputable) {
                        var percent_complete = Math.round((e.loaded / e.total) * 100);  
                        progress.progress = percent_complete;
                    } 
                }
                xhr.send(data);
            },
            setup_file_do_open_file_browser(){
                this.$refs.file_broswer.accept = "*";
                this.$refs.file_broswer.value  = null;
                this.setup_file_dlg.upload.progress = 0;
                this.setup_file_dlg.upload.show = false;
                this.$refs.file_broswer.click();
                this.$refs.file_broswer.onchange = (e)=>{
                    this.create_file_by_upload(e.target.files[0], this.setup_file_dlg.upload, [this.setup_file_on_file_drag_uploaded], this.setup_file_dlg.file_name, this.setup_file_dlg.folder, this.setup_file_dlg.description);
                    this.$refs.file_broswer.onchange = null;
                };
            },
            setup_file_on_file_drag_uploaded(file, progress){
                this.setup_file_dlg.target.title = file.file_name;
                this.setup_file_dlg.target.file_id = file.file_id;
                this.setup_file_dlg.show = false;
                progress.show = false;
            },
            setup_file_ondrop(e, progress, callback=null) {
                e.preventDefault();
                if(e.dataTransfer.files.length == 0)
                    return;

                this.create_file_by_upload(e.dataTransfer.files[0], progress, callback, this.setup_file_dlg.file_name, this.setup_file_dlg.folder, this.setup_file_dlg.description);
            },
            setup_file_dragover(e){
                e.preventDefault();
            },
            open_setup_file_dlg(target, callback=null){
                this.setup_file_dlg.upload.progress = 0;
                this.setup_file_dlg.upload.show = false;
                this.setup_file_dlg.target = target;
                this.setup_file_dlg.show = true;
                this.setup_file_dlg.callback = callback;
                this.setup_file_dlg_focus_input(this.setup_file_dlg.tab_value);
            },
            profile_layers_setup_from_logfile_request_files(file, target, config_func, projection_code=null){
                const _this = this;
                var xhr = new XMLHttpRequest();
                xhr.open('POST', '/generate_profile_layers_from_logfile', true);
                xhr.setRequestHeader("Content-Type", "application/json;charset=UTF-8");
                xhr.onload = function (e) {
                    if (xhr.status === 200) {
                        const resp = JSON.parse(xhr.responseText);
                        if(resp.status != "ok"){
                            target.show = false;
                            _this.$message({
                                message: 'Failed to generate profile layers from logfile, error: ' + resp.message,
                                center: true,
                                type: "error"
                            });
                            return;
                        }

                        const data = resp.data;
                        const layers = data.layers;
                        const profile = data.profile;
                        const onnx = data.onnx;
                        target.show = false;
                        target.file_id = file.file_id;
                        target.title = "Generate from Build.log";
                        if(config_func){
                            if(onnx != null){
                                config_func(layers, profile, onnx);
                            }else{
                                config_func(layers, profile);
                            }
                        }
                    }
                };
                xhr.send(JSON.stringify({build_log_file_id: file.file_id, projection_code: projection_code}));
            },
            config_profile_layers_for_onnx(layers, profile, onnx){
                if(layers != null){
                    this.btn_progress.onnx_layerinfo.show = false;
                    this.btn_progress.onnx_layerinfo.file_id = layers.file_id;
                    this.btn_progress.onnx_layerinfo.title = this.format_progress_title(layers.file_name, layers.file_id);
                }

                if(profile != null){
                    this.btn_progress.onnx_profile.show = false;
                    this.btn_progress.onnx_profile.file_id = profile.file_id;
                    this.btn_progress.onnx_profile.title = this.format_progress_title(profile.file_name, profile.file_id);
                }

                if(onnx != null){
                    this.btn_progress.onnx_model.show = false;
                    this.btn_progress.onnx_model.file_id = onnx.file_id;
                    this.btn_progress.onnx_model.title = this.format_progress_title(onnx.file_name, onnx.file_id);
                }
            },
            config_profile_layers_for_onnx_compared(layers, profile){
                if(layers != null){
                    this.btn_progress.onnx_layerinfo_compared.show = false;
                    this.btn_progress.onnx_layerinfo_compared.file_id = layers.file_id;
                    this.btn_progress.onnx_layerinfo_compared.title = this.format_progress_title(layers.file_name, layers.file_id);
                }

                if(profile != null){
                    this.btn_progress.onnx_profile_compared.show = false;
                    this.btn_progress.onnx_profile_compared.file_id = profile.file_id;
                    this.btn_progress.onnx_profile_compared.title = this.format_progress_title(profile.file_name, profile.file_id);
                }
            },
            config_profile_layers_for_trex(layers, profile){
                if(layers != null){
                    this.btn_progress.trex_layerinfo.show = false;
                    this.btn_progress.trex_layerinfo.file_id = layers.file_id;
                    this.btn_progress.trex_layerinfo.title = this.format_progress_title(layers.file_name, layers.file_id);
                }

                if(profile != null){
                    this.btn_progress.trex_profile.show = false;
                    this.btn_progress.trex_profile.file_id = profile.file_id;
                    this.btn_progress.trex_profile.title = this.format_progress_title(profile.file_name, profile.file_id);
                }
            },
            cancel_setup_path_projection(){
                this.setup_path_projection_dlg.show = false;
                if(this.setup_path_projection_dlg.callback){
                    this.setup_path_projection_dlg.callback(null);
                }
            },
            confirm_setup_path_projection(){
                localStorage.setItem("setup_path_projection_code", this.setup_path_projection_dlg.projection_code);
                this.setup_path_projection_dlg.show = false;
                if(this.setup_path_projection_dlg.callback){
                    this.setup_path_projection_dlg.callback(this.setup_path_projection_dlg.projection_code);
                }
            },
            profile_layers_setup_from_logfile(){
                // this.setup_path_projection_dlg.show = true;
                // this.$nextTick(()=>{
                //     this.$refs.setup_path_projection_dlg_projection_code.focus();
                //     this.$refs.setup_path_projection_dlg_projection_code.select();
                // });

                // this.setup_path_projection_dlg.callback = (projection_code)=>{
                //     this.open_setup_file_dlg(this.btn_progress.onnx_profile_layers_from_logfile, [(data, target)=>{
                //         this.profile_layers_setup_from_logfile_request_files(data, target, this.config_profile_layers_for_onnx, projection_code);
                //     }]);
                // };
                this.open_setup_file_dlg(this.btn_progress.onnx_profile_layers_from_logfile, [this.profile_layers_setup_from_logfile_request_files, this.config_profile_layers_for_onnx]);
            },
            profile_layers_setup_from_logfile_compared(){
                this.open_setup_file_dlg(this.btn_progress.onnx_profile_layers_from_logfile_compared, [this.profile_layers_setup_from_logfile_request_files, this.config_profile_layers_for_onnx_compared]);
            },
            trex_profile_layers_setup_from_logfile(){
                this.open_setup_file_dlg(this.btn_progress.trex_profile_layers_from_logfile, [this.profile_layers_setup_from_logfile_request_files, this.config_profile_layers_for_trex]);
            },
            setup_onnx_model(){
                this.open_setup_file_dlg(this.btn_progress.onnx_model);
            },
            setup_onnx_layerinfo(){
                this.open_setup_file_dlg(this.btn_progress.onnx_layerinfo);
            },
            setup_onnx_profile(){
                this.open_setup_file_dlg(this.btn_progress.onnx_profile);
            },
            setup_trtperf(){
                this.open_setup_file_dlg(this.btn_progress.trtperf);
            },
            setup_onnx_layerinfo_compared(){
                this.open_setup_file_dlg(this.btn_progress.onnx_layerinfo_compared);
            },
            setup_onnx_profile_compared(){
                this.open_setup_file_dlg(this.btn_progress.onnx_profile_compared);
            },
            setup_trtperf_compared(){
                this.open_setup_file_dlg(this.btn_progress.trtperf_compared);
            },
            setup_trex_layerinfo(){
                this.open_setup_file_dlg(this.btn_progress.trex_layerinfo);
            },
            setup_trex_profile(){
                this.open_setup_file_dlg(this.btn_progress.trex_profile);
            },
            async create_a_view_and_open(view, callback=null){
                const _this = this;
                this.loadingInstance = window.ELEMENT.Loading.service({ fullscreen: true });

                var xhr = new XMLHttpRequest();
                xhr.open('POST', '/create_view', true);
                xhr.setRequestHeader("Content-Type", "application/json;charset=UTF-8");
                xhr.onload = function (e) {
                    if(callback) callback();
                    _this.loadingInstance.close();
                    if (xhr.status === 200) {
                        const resp = JSON.parse(xhr.responseText);
                        if(resp.status != "ok"){
                            _this.$message({
                                message: 'Failed to create view, error: ' + resp.message,
                                center: true,
                                type: "error"
                            });
                            return;
                        }
                        window.open("/view/" + resp.data.view_id, "_self");
                    }
                };
                xhr.send(JSON.stringify(view));
            },
            confirm_create_a_view(){
                if(this.create_view_dlg.mode == "onnx_view"){
                    this.create_a_view_and_open({
                        view_type: "onnx",
                        name: this.create_view_dlg.view_name,
                        model: this.btn_progress.onnx_model.file_id,
                        layerinfo: this.btn_progress.onnx_layerinfo.file_id,
                        profile: this.btn_progress.onnx_profile.file_id,
                        trtperf: this.btn_progress.trtperf.file_id,
                        layerinfo_compared: this.btn_progress.onnx_layerinfo_compared.file_id,
                        profile_compared: this.btn_progress.onnx_profile_compared.file_id,
                        trtperf_compared: this.btn_progress.trtperf_compared.file_id,
                        with_kernel_match_report: this.with_kernel_match_report,
                        folder: this.create_view_dlg.folder,
                        description: this.create_view_dlg.description,
                        buildlog: this.btn_progress.onnx_profile_layers_from_logfile.file_id,
                        buildlog_compared: this.btn_progress.onnx_profile_layers_from_logfile_compared.file_id
                    });
                }else if(this.create_view_dlg.mode == "trex_view"){
                    this.create_a_view_and_open({
                        view_type: "trex",
                        name: this.create_view_dlg.view_name,
                        layerinfo: this.btn_progress.trex_layerinfo.file_id,
                        profile: this.btn_progress.trex_profile.file_id,
                        folder: this.create_view_dlg.folder,
                        description: this.create_view_dlg.description,
                        buildlog: this.btn_progress.trex_profile_layers_from_logfile.file_id
                    });
                }
            },
            open_onnx_view(){
                if(this.btn_progress.onnx_model.show || this.btn_progress.onnx_layerinfo.show || this.btn_progress.onnx_profile.show){
                    this.$message({
                        message: 'Please wait for the upload to complete.',
                        center: true,
                        type: "warning"
                    });
                    return;
                }
                if(this.with_kernel_match_report){
                    if(this.btn_progress.onnx_model.file_id == null ||
                        this.btn_progress.onnx_layerinfo.file_id == null ||
                        this.btn_progress.onnx_profile.file_id == null ||
                        this.btn_progress.onnx_layerinfo_compared.file_id == null ||
                        this.btn_progress.onnx_profile_compared.file_id == null
                    ){
                        this.$message({
                            message: 'Missing the required file: onnx_model, onnx_layerinfo, onnx_profile, compared[onnx_layerinfo, profile]',
                            center: true,
                            type: "warning"
                        });
                        return;
                    }
                }else{
                    if(this.btn_progress.onnx_model.file_id == null){
                        this.$message({
                            message: 'Missing the required file: onnx_model',
                            center: true,
                            type: "warning"
                        });
                        return;
                    }
                }
                this.create_view_dlg.mode = "onnx_view";
                this.create_view_dlg.show = true;
            },
            open_trex_view(){
                if(this.btn_progress.trex_layerinfo.show || this.btn_progress.trex_profile.show){
                    this.$message({
                        message: 'Please wait for the upload to complete.',
                        center: true,
                        type: "warning"
                    });
                    return;
                }
                if(this.btn_progress.trex_layerinfo.file_id == null){
                    this.$message({
                        message: 'Missing the required file: layerinfo.json',
                        center: true,
                        type: "warning"
                    });
                    return;
                }
                this.create_view_dlg.mode = "trex_view";
                this.create_view_dlg.show = true;
            },
            format_progress_title(file_name, file_id){
                return file_name.substring(0, Math.min(16, file_name.length)) + (file_name.length > 16 ? "..." : "") + " [" + file_id + "]";
            },
            confirm_from_file_id(){
                if(this.$refs.from_file_id_confirm_btn.$el.classList.contains("btn-disable"))
                    return;

                if(this.setup_file_dlg.description == null || this.setup_file_dlg.description == "" || this.setup_file_dlg.description.startsWith("From File ID: ")){
                    this.setup_file_dlg.description = "From File ID: " + this.setup_file_dlg.from_file_id;
                }

                this.$refs.from_file_id_confirm_btn.$el.classList.add("btn-disable");
                const _this = this;
                const file_id = this.setup_file_dlg.from_file_id;
                var xhr = new XMLHttpRequest();
                xhr.open('POST', '/get_file_info', true);
                xhr.setRequestHeader("Content-Type", "application/json;charset=UTF-8");
                xhr.onload = function (e) {
                    _this.$refs.from_file_id_confirm_btn.$el.classList.remove("btn-disable");

                    if (xhr.status === 200) {
                        const resp = JSON.parse(xhr.responseText);
                        if(resp.status != "ok"){
                            _this.$message({
                                message: 'Failed to create view, error: ' + resp.message,
                                center: true,
                                type: "error"
                            });
                            return;
                        }
                        const file_name = resp.data.file_name;
                        _this.setup_file_dlg.target.title   = _this.format_progress_title(file_name, file_id);
                        _this.setup_file_dlg.target.file_id = file_id;
                        if(_this.create_view_dlg.view_name == "unnamed"){
                            _this.create_view_dlg.view_name = file_name;
                        }
                        if(_this.create_view_dlg.description == ""){
                            _this.create_view_dlg.description = _this.setup_file_dlg.description;
                        }
                        if(_this.create_view_dlg.folder == "default"){
                            _this.create_view_dlg.folder = _this.setup_file_dlg.folder;
                        }
                        _this.setup_file_dlg.show = false;
                        if(_this.setup_file_dlg.callback){
                            const func = _this.setup_file_dlg.callback[0];
                            const args = _this.setup_file_dlg.callback.slice(1);
                            func(resp.data, _this.setup_file_dlg.target, ...args);
                        }
                    }
                };
                xhr.send(JSON.stringify({"file_id": this.setup_file_dlg.from_file_id}));
            },
            confirm_from_url(){
                if(this.$refs.from_url_confirm_btn.$el.classList.contains("btn-disable"))
                    return;

                if(this.setup_file_dlg.from_file_url == null || this.setup_file_dlg.from_file_url == ""){
                    this.$message({
                        message: 'Please input the URL.',
                        center: true,
                        type: "warning"
                    });
                    this.$refs.from_url_input_url.focus();
                    return;
                }

                if(this.setup_file_dlg.description == null || this.setup_file_dlg.description == "" || this.setup_file_dlg.description.startsWith("From URL: ")){
                    this.setup_file_dlg.description = "From URL: " + this.setup_file_dlg.from_file_url;
                }

                this.$refs.from_url_confirm_btn.$el.classList.add("btn-disable");
                const _this = this;
                var xhr = new XMLHttpRequest();
                xhr.open('POST', '/create_file_from_url', true);
                xhr.setRequestHeader("Content-Type", "application/json;charset=UTF-8");
                xhr.onload = function (e) {
                    _this.$refs.from_url_confirm_btn.$el.classList.remove("btn-disable");

                    if (xhr.status === 200) {
                        const resp = JSON.parse(xhr.responseText);
                        if(resp.status != "ok"){
                            _this.$message({
                                message: 'Failed to create view, error: ' + resp.message,
                                center: true,
                                type: "error"
                            });
                            return;
                        }
                        const file_name = resp.data.file_name;
                        const file_id   = resp.data.file_id;
                        _this.setup_file_dlg.target.title   = _this.format_progress_title(file_name, file_id);
                        _this.setup_file_dlg.target.file_id = file_id;
                        if(_this.create_view_dlg.view_name == "unnamed"){
                            _this.create_view_dlg.view_name = file_name;
                        }
                        if(_this.create_view_dlg.description == ""){
                            _this.create_view_dlg.description = _this.setup_file_dlg.description;
                        }
                        if(_this.create_view_dlg.folder == "default"){
                            _this.create_view_dlg.folder = _this.setup_file_dlg.folder;
                        }
                        _this.setup_file_dlg.show = false;
                        if(_this.setup_file_dlg.callback){
                            const func = _this.setup_file_dlg.callback[0];
                            const args = _this.setup_file_dlg.callback.slice(1);
                            func(resp.data, _this.setup_file_dlg.target, ...args);
                        }
                    }
                };
                xhr.send(JSON.stringify({file_url: this.setup_file_dlg.from_file_url, file_name: this.setup_file_dlg.file_name, description: this.setup_file_dlg.description, folder: this.setup_file_dlg.folder}));
            },
            confirm_from_ssh(){
                if(this.$refs.from_ssh_confirm_btn.$el.classList.contains("btn-disable"))
                    return;

                if(this.setup_file_dlg.from_ssh_ip == null || this.setup_file_dlg.from_ssh_ip == "" || 
                    this.setup_file_dlg.from_ssh_path == null || this.setup_file_dlg.from_ssh_path == "" || 
                    this.setup_file_dlg.from_ssh_password == null || this.setup_file_dlg.from_ssh_password == ""){
                    this.$message({
                        message: 'Please input the SSH IP, path and password.',
                        center: true,
                        type: "warning"
                    });
                    return;
                }

                if(this.setup_file_dlg.description == null || this.setup_file_dlg.description == "" || this.setup_file_dlg.description.startsWith("From SSH: ")){
                    this.setup_file_dlg.description = "From SSH: " + this.setup_file_dlg.from_ssh_path;
                }
                
                localStorage.setItem("create_view_from_ssh_ip", this.setup_file_dlg.from_ssh_ip);
                localStorage.setItem("create_view_from_ssh_password", this.setup_file_dlg.from_ssh_password);
                this.$refs.from_ssh_confirm_btn.$el.classList.add("btn-disable");
                const _this = this;
                var xhr = new XMLHttpRequest();
                xhr.open('POST', '/create_file_from_ssh', true);
                xhr.setRequestHeader("Content-Type", "application/json;charset=UTF-8");
                xhr.onload = function (e) {
                    _this.$refs.from_ssh_confirm_btn.$el.classList.remove("btn-disable");

                    if (xhr.status === 200) {
                        const resp = JSON.parse(xhr.responseText);
                        if(resp.status != "ok"){
                            _this.$message({
                                message: 'Failed to create view, error: ' + resp.message,
                                center: true,
                                type: "error"
                            });
                            return;
                        }
                        const file_name = resp.data.file_name;
                        const file_id   = resp.data.file_id;
                        _this.setup_file_dlg.target.title   = _this.format_progress_title(file_name, file_id);
                        _this.setup_file_dlg.target.file_id = file_id;
                        if(_this.create_view_dlg.view_name == "unnamed"){
                            _this.create_view_dlg.view_name = file_name;
                        }
                        if(_this.create_view_dlg.description == ""){
                            _this.create_view_dlg.description = _this.setup_file_dlg.description;
                        }
                        if(_this.create_view_dlg.folder == "default"){
                            _this.create_view_dlg.folder = _this.setup_file_dlg.folder;
                        }
                        _this.setup_file_dlg.show = false;
                        if(_this.setup_file_dlg.callback){
                            const func = _this.setup_file_dlg.callback[0];
                            const args = _this.setup_file_dlg.callback.slice(1);
                            func(resp.data, _this.setup_file_dlg.target, ...args);
                        }
                    }
                };
                xhr.send(JSON.stringify({ip: this.setup_file_dlg.from_ssh_ip, path: this.setup_file_dlg.from_ssh_path, password: this.setup_file_dlg.from_ssh_password, file_name: this.setup_file_dlg.file_name, description: this.setup_file_dlg.description, folder: this.setup_file_dlg.folder}));
            },
            setup_file_dlg_focus_input(tabname){
                if(tabname == "FromFileID"){
                    this.$nextTick(()=>{
                        this.$refs.from_file_input_file_id.select();
                        this.$refs.from_file_input_file_id.focus();
                    });
                }else if(tabname == "FromURL"){
                    this.$nextTick(()=>{
                        this.$refs.from_url_input_url.select();
                        this.$refs.from_url_input_url.focus();
                    });
                }else if(tabname == "FromSSH"){
                    this.$nextTick(()=>{
                        this.$refs.from_ssh_input_path.select();
                        this.$refs.from_ssh_input_path.focus();
                    });
                }
            },
            setup_file_dlg_tabclick(e){
                localStorage.setItem("last_operation_page", e.name);
                this.setup_file_dlg_focus_input(e.name);
            }
        }
        });
    }
};

const ui = new UI("addition-ui");
export default ui;