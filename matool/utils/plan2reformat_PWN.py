import sys
from trex import *
import pandas as pd
from pandas import json_normalize
import plotly.offline as pyo
from bs4 import BeautifulSoup
import shutil
# This dictionary compresses JSON's long format description strings.
regionFormatDict = {
    "Four wide channel vectorized row major Int8 format" : "Int8 NC/4HW4",
    "Four wide channel vectorized row major FP32 format" : "FP32 NC/4HW4",
    "Thirty-two wide channel vectorized row major Int8 format": "Int8 NC/32HW32",
    "Thirty-two wide channel vectorized row major FP32 format": "FP32 NC/32HW32",
    "Thirty-two wide channel vectorized row major FP16 format": "FP16 NC/32HW32",
    "Thirty-two wide channel vectorized row major Int8 format with 3 spatial dimensions": "Int8 NC32DHW",
    "Thirty-two wide channel vectorized row major FP16 format with 3 spatial dimensions": "FP16 NC32DHW",
    "Sixteen wide channel vectorized row major FP16 format": "FP16 NC16HW",
    "Channel major FP16 format where channel % 4 == 0": "FP16 NHWC4",
    "Channel major FP32 format where channel % 4 == 0": "FP32 NHWC4",
    "Channel major Int8 format where channel % 4 == 0": "Int8 NHWC4",
    "Channel major FP16 format where channel % 8 == 0": "FP16 NHWC8",
    "Channel major FP16 format where channel % 16 == 0": "FP16 NHWC16",
    "Channel major FP16 format where channel == 4 and column stride % 32 == 0": "FP16 NHWC4",
    "Channel major INT8 format where channel == 4 and column stride % 32 == 0": "Int8 NHWC4",
    "Channel major INT8 format where column stride % 32 == 0": "Int8 NHWC1",
    "Row major INT8 format where column stride % 64 == 0": "Int8 NCHW",
    "Channel major FP16 format where channel % 8 == 0 with 3 spatial dimensions": "FP16 NDHWC8",
    "Channel major FP16 format where channel == 1 and column stride % 32 == 0": "FP16 NHWC1",
    "Row major FP16 format where column stride % 64 == 0": "FP16",
    "Two wide channel vectorized row major FP16 format": "FP16 NC/2HW2",
    "Row major linear FP32": "FP32 NCHW",
    "Row major linear Int32": "INT32 NCHW",
    "Row major linear FP16 format": "FP16 NCHW",
    "Row major Int8 format": "Int8 NCHW",
    "Channel major FP32 format":"FP32 NHWC",
    "Channel major FP16 format": "FP16 NHWC",
    "Channel major Int8 format": "Int8 NHWC",
    "Row major linear BOOL": "Bool",
    "Unknown format": "Unknown format"
}
reformat_colormap = defaultdict(lambda: UNKNOWN_KEY_COLOR, {
    'INT8->INT8':'red',
    'FP16->FP16':'red',
    'FP32->FP32':'red',
    'INT8->FP16':'green',
    'INT8->FP32':'green',
    'FP16->INT8':'green',
    'FP16->FP32':'green',
    'FP32->INT8':'green',
    'FP32->FP16':'green',
})
categories=[    
    'INT8->INT8',
    'FP16->FP16',
    'FP32->FP32',
    'INT8->FP16',
    'INT8->FP32',
    'FP16->INT8',
    'FP16->FP32',
    'FP32->INT8',
    'FP32->FP16']

def plan2reformat_PWN(plan,save_path,):
    report_html_preparation(save_path)#复制report8到对应文件夹
    slim_df=get_slim_plan_df(plan)
    analyze_reformatting(slim_df,save_path)
    analyze_PWN(slim_df,save_path)

def get_slim_plan_df(plan):
    slim_df=plan.df.loc[:,['Name', 'type','output_precision', 'precision', 'Origin' ,'Inputs','Outputs']]
    slim_df['Input Format'] = slim_df['Inputs'].apply(lambda x: [regionFormatDict.get(item['Format/Datatype'], "Unknown format") for item in x].__str__()[1:-1].replace('\'',''))
    slim_df['Output Format'] = slim_df['Outputs'].apply(lambda x: [regionFormatDict.get(item['Format/Datatype'], "Unknown format") for item in x].__str__()[1:-1].replace('\'',''))
    slim_df=slim_df.loc[:,['Name', 'type', 'Origin' ,'Input Format','Output Format','precision', 'output_precision']]
    return slim_df

def remove_redundancy(string):
    pattern = r'(_\d+|PWN|\(|\)|Unnamed Layer\* \d+|\s|\[|\])'
    result= re.sub(pattern, '', string)
    return result
def analyze_PWN(slim_df,save_path):
    # pattern = r'PWN\([^()]+\)'
    PWN_df=slim_df[slim_df['type']=='PointWise']
    #PWN数量
    PWN_num=len(PWN_df)
    if(PWN_num!=0):
        PWN_df['tag'] = PWN_df['Name'].apply(lambda x: remove_redundancy(x).split(','))
        # print(PWN_df['tag'])
        PWN_df['operators']=PWN_df.apply(lambda x: ', '.join(frozenset(x['tag'])),axis=1)
        grouped=PWN_df.groupby(['operators']).size().reset_index(name='count')

        #绘制reformatting nodes类型统计图
        fig=px_bar(
            df=grouped,
            title="PWN类型统计",
            values_col='count',
            names_col='operators',
            orientation='h',
            show_axis_ticks=(True,True),
            do_show=False
        )
        pyo.plot(fig, filename=save_path + '/graphs/PWN statistical chart.html',auto_open=False)# save_path为对应report下的graphs文档
        
        
        PWN_df=PWN_df.drop(columns=['Origin','precision', 'output_precision','type','tag'])
        #保存PWN列表到html
        table_name = 'PWN列表【点击跳转到对应node】'
        PWN_optimize_path=save_path + '/graphs/PWN dataframe.html'
        PWN_df.to_html(PWN_optimize_path) 
        add_scroll_to_html_table(PWN_optimize_path,save_path+"/structure/engine.svg",table_name)
        add_num_to_PWN_report(PWN_num,save_path+"/reports/report8.html")
    else:#对无PWN情况做出处理
        add_num_to_PWN_report(PWN_num,save_path+"/reports/report8.html")


def analyze_reformatting(slim_df,save_path):
    #reformat dataframe
    reformat_df=slim_df[(slim_df['type']=='Reformat')]
    reformat_optimize_df=slim_df[(slim_df['type']=='Reformat') & (slim_df['output_precision']==slim_df['precision'])]
    #reformat数量+ 同精度转换reformat数量
    reformat_total_num=len(reformat_df)
    reformat_optimize_num=len(reformat_optimize_df)#需对无同精度转换情况做出处理

    #统计转换类型并排序
    grouped = reformat_df.groupby(['precision', 'output_precision']).size().reset_index(name='count')
    grouped['type']=grouped.apply(lambda row:row['precision']+"->"+row['output_precision'],axis=1)
    grouped['type']=grouped['type'].astype(pd.CategoricalDtype(categories=categories,ordered=True))
    grouped=grouped.sort_values(by='type')
    #绘制reformatting nodes类型统计图
    fig=px_bar(
        df=grouped,
        title="reformatting nodes类型统计",
        values_col='count',
        names_col='type',
        orientation='h',
        color='type',
        colormap=reformat_colormap,
        show_axis_ticks=(True,True),
        do_show=False
    )
    pyo.plot(fig, filename=save_path + '/graphs/reformatting nodes statistical chart.html',auto_open=False)

    if(reformat_optimize_num!=0):
        #保存同精度转换reformatting列表到html
        reformat_optimize_df=reformat_optimize_df.drop(columns=['precision', 'output_precision','type'])
        table_name = '输入输出同精度的reformatting算子【点击跳转到对应node】'
        reformat_optimize_path=save_path + '/graphs/reformatting dataframe.html'
        reformat_optimize_df.to_html(reformat_optimize_path) 
        add_scroll_to_html_table(reformat_optimize_path,save_path+"/structure/engine.svg",table_name)
        add_num_to_reformat_report(reformat_total_num,reformat_optimize_num,save_path+"/reports/report8.html")
    else:#对无同精度转换reformatting情况做出处理
        #保存所有reformatting列表到html
        reformat_df=reformat_df.drop(columns=['precision', 'output_precision','type'])
        table_name = 'reformatting算子列表【点击跳转到对应node】'
        reformat_path=save_path + '/graphs/reformatting dataframe.html'
        reformat_df.to_html(reformat_path) 
        add_scroll_to_html_table(reformat_path,save_path+"/structure/engine.svg",table_name)
        add_num_to_reformat_report(reformat_total_num,reformat_optimize_num,save_path+"/reports/report8.html")

#为dataframe添加点击滑动到structure svg对应位置的onclick属性   
def add_scroll_to_html_table(html_file, svg_file,table_name):
    with open(html_file, 'r') as f:
        html_content = f.read()
    # 使用BeautifulSoup解析HTML和SVG
    html_soup = BeautifulSoup(html_content, 'html.parser')

    with open(svg_file, 'r') as f:
        svg_content = f.read()
    svg_soup = BeautifulSoup(svg_content, 'html.parser')
    # g_node_tags = svg_soup.find_all('g',id=lambda x: x and x.startswith("node"))#切换使用a_node查找，以解决node id######问题
    g_a_node_tags= svg_soup.find_all('g',id=lambda x: x and x.startswith("a_node"))
    # 找到并遍历HTML中所有的tr标签  
    tr_tags = html_soup.find_all('tr')
    for tr_tag in tr_tags[1:]:
        # 找到tr下的第一个td内容，即node name
        td_content = tr_tag.find('td').text
        # 在SVG中查找匹配的g标签
        # matching_g_node_tag = next((x for x in g_node_tags if x.text.split('\n')[1] == td_content), None)
        matching_g_a_node_tag = next((x for x in g_a_node_tags if x.next.attrs['xlink:title'].split('\n')[0][5:] == td_content), None)

        if matching_g_a_node_tag:
            # 取出匹配g标签的id   
            g_id = matching_g_a_node_tag['id']
            # 将id添加到tr标签中
            tr_tag['onclick'] = f"parent.scrollToElement('{g_id}')"

    # 创建并添加表名用于提示
    # table_name = '输入输出同精度的reformatting算子【点击跳转到对应node】'
    caption = html_soup.new_tag('caption')
    caption.string = table_name
    html_soup.table.insert(0, caption)
    
    # 将修改后的HTML写回文件
    with open(html_file, 'w') as f:
        f.write(html_soup.prettify())

def add_num_to_reformat_report(total_num,same_precision_num,report_file):
    with open(report_file, 'r') as f:
        report_content = f.read()
    report_soup = BeautifulSoup(report_content, 'html.parser')
    num1_tag = report_soup.find('strong', id='num1')
    num1_tag.string = str(total_num)
    num2_tag = report_soup.find('strong', id='num2')
    num2_tag.string = str(same_precision_num)
    with open(report_file, 'w') as f:
        f.write(report_soup.prettify())

def add_num_to_PWN_report(total_num,report_file):
    with open(report_file, 'r') as f:
        report_content = f.read()
    report_soup = BeautifulSoup(report_content, 'html.parser')
    num1_tag = report_soup.find('strong', id='num3')
    num1_tag.string = str(total_num)
    if(total_num==0):#处理无PWN情况
        ccaption_tag_pic = report_soup.new_tag('caption')
        ccaption_tag_pic.string = "该模型无PWN算子，未生成对应统计图"
        caption_tag_html1 = report_soup.new_tag('caption')
        caption_tag_html1.string = "该模型无PWN算子，未生成对应dataframe"
        iframe_pic = report_soup.find(id='pic_PWN')
        iframe_html1 = report_soup.find(id='PWN_df')
        iframe_pic.replace_with(ccaption_tag_pic)
        iframe_html1.replace_with(caption_tag_html1)
    with open(report_file, 'w') as f:
        f.write(report_soup.prettify())

def report_html_preparation(save_path):
    sample_script_path = os.path.abspath(sys.argv[0])
    sample_script_directory = os.path.dirname(sample_script_path)
    folder2mainfolder = ['reports']
    for file_name in folder2mainfolder:
        sample_script = sample_script_directory + '/html_template/' + file_name
        target_script = save_path + '/' + file_name
        shutil.copytree(sample_script, target_script, dirs_exist_ok=True)
    

