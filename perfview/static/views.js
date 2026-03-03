class UI extends Vue{
    constructor(ui_element_id){
        super({
        el: "#" + ui_element_id,
        data: {
            search_card:{
                keywords: "",
                search_views: [],
                total: 0,
                pagesize: 20,
                page: 0,
                loading: false
            }
        },
        mounted(){
            this.$nextTick(this.update_search);
        },
        methods: {
            post_request(url, params, callback){
                var _this = this;
                _this.search_card.loading = true;

                var xhr = new XMLHttpRequest();
                xhr.open('POST', url, true);
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
                            _this.search_card.loading = false;
                            return;
                        }
                        callback(resp.data);
                    }
                    _this.search_card.loading = false;
                };
                xhr.send(JSON.stringify(params));
            },
            update_search(){
                this.post_request('/list_views_by_keyword', {
                    keywords: this.search_card.keywords,
                    page: this.search_card.page - 1,
                    pagesize: this.search_card.pagesize
                }, (data)=>{
                    const type_mapping = {
                        onnx_with_kernel_match: "Comparison",
                        onnx: "ONNX",
                        trex: "Trex",
                    };
                    for(const item of data.views){
                        item.view_type = type_mapping[item.view_type];
                    }
                    this.search_card.search_views = data.views;
                    this.search_card.total = data.count;

                    this.$nextTick(()=>{
                        this.$refs.search_card_keywords_input.focus();
                        this.$refs.search_card_keywords_input.select();
                    });
                });
            },
            do_search(){
                this.search_card.page = 1;
                this.update_search();
            },
            search_card_size_change(val){
                this.search_card.page = val;
                this.update_search();
            },
            search_card_current_change(val){
                this.search_card.page = val;
                this.update_search();
            },
            navigate_to_view(row){
                window.open("/view/" + row.idd, "_blank");
            }
        }
        });
    }
};

const ui = new UI("addition-ui");
export default ui;