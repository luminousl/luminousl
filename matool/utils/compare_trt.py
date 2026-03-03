#对比优化前后的trt,通过差异节点列表和跳转公共来源节点功能方便快捷地定位差异
import sys
# sys.path.append(r"/workspace/tools/")
import os
from trex import *
import re
import shutil
import pandas as pd
import argparse
from pandas import json_normalize
from bs4 import BeautifulSoup

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

def create_layout_format(list):
    result=[]
    for item in list:
        layout_format=regionFormatDict.get(item['Format/Datatype'],"Unknown format") 
        result.append(layout_format)
    return result

def simplify_dataframe(dataframe):
    dataframe=dataframe.loc[:,['Name', 'subtype' ,'Inputs','Outputs']]
    dataframe['Input Format'] = dataframe['Inputs'].apply(lambda x: [regionFormatDict.get(item['Format/Datatype'], "Unknown format") for item in x].__str__()[1:-1].replace('\'',''))
    dataframe['Output Format'] = dataframe['Outputs'].apply(lambda x: [regionFormatDict.get(item['Format/Datatype'], "Unknown format") for item in x].__str__()[1:-1].replace('\'',''))
    dataframe['Inputs'] = dataframe['Inputs'].apply(lambda x: [item['Name'] for item in x].__str__()[1:-1].replace('\'',''))
    dataframe['Outputs'] = dataframe['Outputs'].apply(lambda x: [item['Name'] for item in x].__str__()[1:-1].replace('\'',''))
    return dataframe

def name_equal_impl(string_a,string_b):
    if '||' in string_a and '||' in string_b:
        names_opt = [re.sub(r'\s+', '', n) for n in string_a.split('||')]
        names_trt = [re.sub(r'\s+', '', n) for n in string_b.split('||')]
        if set(names_opt) == set(names_trt):
            return True
    elif '+' in string_a and '+' in string_b:
        names_opt = [re.sub(r'\s+', '', n) for n in string_a.split('+')]
        names_trt = [re.sub(r'\s+', '', n) for n in string_b.split('+')]
        if set(names_opt) == set(names_trt):
            return True
    elif string_a == string_b:
        return True
    return False

#判断节点是否为同一个，用于找出优化前后共有节点
def name_equal(name_trt, name_opt):
    if name_trt.startswith("Reformatting CopyNode for") and name_opt.startswith("Reformatting CopyNode for"):
        pattern = r"for\s(.+?)\sto\s(.+)"
        match_trt = re.search(pattern, name_trt)
        match_opt = re.search(pattern, name_opt)
        if match_trt and match_opt and match_trt.group(1)==match_opt.group(1):
            node_trt = match_trt.group(2)
            node_opt = match_opt.group(2)
            return name_equal_impl(node_opt,node_trt)
    
    return name_equal_impl(name_trt,name_opt)

#判断两个节点是否相连：一个的输入和另外一个的输出是否有交集
def has_intersection(list1, list2):
    for item in list1:
        if item in list2:
            return True
    return False

def get_intersection(list1,list2):
    for item in list1:
        if item in list2:
            return item
    return None

#循环寻找节点的非共有来源节点
def find_source_nodes(index, df):
    source_nodes = set()
    row=df.loc[index]
    inputs = row['Inputs'].split(', ')
    for index_i, row_i in df.iterrows():
        outputs_i = row_i['Outputs'].split(', ')
        if has_intersection(inputs,outputs_i):
            source_nodes.add(index_i)
            source_nodes.update(find_source_nodes(index_i, df))
    return source_nodes

#得到节点的非公有顶级来源节点
def find_top_source_nodes(index, df):
    all_source_nodes = find_source_nodes(index, df)
    top_source_nodes = set()
    #没有来源节点，自己就是顶级来源节点
    if len(all_source_nodes)==0:
        top_source_nodes.add(index)
        return top_source_nodes
    
    for node_index in all_source_nodes:
        node=df.loc[node_index]
        inputs=node['Inputs'].split(', ')
        is_top=True
        for i in all_source_nodes:
            if has_intersection(inputs,df.loc[i]['Outputs'].split(', ')):
                is_top=False
                break
        if is_top:
            top_source_nodes.add(node_index)
    return top_source_nodes

#寻找另一个模型中对应的共有来源节点
def find_public_source_nodes(top_source_nodes,df,df_public,df_public_otherside,inputs_otherside):
    top_inputs=set()#非共有来源节点的所有输入tensor
    public_source_nodes=set()#共有来源节点set
    otherside_public_source_nodes=set()#对应的另外一个模型的共有节点
    
    for top_source_node in top_source_nodes:
        node=df.loc[top_source_node]
        input_tensors=node['Inputs'].split(', ')
        top_inputs.update(input_tensors)
    
    #得到自身模型的共有来源节点
    for index_i, row_i in df_public.iterrows():
        outputs_i = row_i['Outputs'].split(', ')
        if has_intersection(outputs_i,top_inputs):
            public_source_nodes.add(index_i)
    
    #无共有来源节点的情况
    if len(public_source_nodes)==0:
        if has_intersection(inputs_otherside,top_inputs):#来源为输入的情况，假定一个节点只会有一个公共顶级节点为输入
            otherside_public_source_nodes.add(get_intersection(inputs_otherside,top_inputs))
            return otherside_public_source_nodes
        else:#来源于一个非共有顶级节点,返回空set
            return otherside_public_source_nodes

    #寻找另一个模型对应的节点
    for public_source_node in public_source_nodes:
        for index_i, row_i in df_public_otherside.iterrows():
            if name_equal(df_public.loc[public_source_node]['Name'],row_i['Name']):
                otherside_public_source_nodes.add(row_i['Name'])

    return otherside_public_source_nodes

#为dataframe添加点击滑动到structure svg对应位置的onclick属性   
def add_scroll_to_html_table(html_file_trt,html_file_opt, svg_file_trt,svg_file_opt):
    with open(html_file_trt, 'r') as f:
        html_content_trt = f.read()
    with open(html_file_opt, 'r') as f:
        html_content_opt = f.read()
    # 使用BeautifulSoup解析HTML和SVG
    html_soup_trt = BeautifulSoup(html_content_trt, 'html.parser')
    html_soup_opt = BeautifulSoup(html_content_opt, 'html.parser')

    with open(svg_file_trt, 'r') as f:
        svg_content_trt = f.read()
    with open(svg_file_opt, 'r') as f:
        svg_content_opt = f.read()

    svg_soup_trt = BeautifulSoup(svg_content_trt, 'html.parser')
    svg_soup_opt = BeautifulSoup(svg_content_opt, 'html.parser')
    g_a_node_tags_trt= svg_soup_trt.find_all('g',id=lambda x: x and x.startswith("a_node"))
    g_a_node_tags_opt= svg_soup_opt.find_all('g',id=lambda x: x and x.startswith("a_node"))
    # 找到并遍历HTML中所有的tr标签  
    tr_tags_trt = html_soup_trt.find_all('tr')
    for tr_tag in tr_tags_trt[1:]:
        # 找到tr下的第一个td内容，即node name
        # td_content = tr_tag.find('td').text
        node_name=str(tr_tag).split('\n')[2][4:-5]
        source_name=str(tr_tag).split('\n')[3][4:-5]
        # 在SVG中查找匹配的g标签
        matching_g_a_node_tag_trt = next((x for x in g_a_node_tags_trt if (x.next.attrs['xlink:title'].split('\n')[0][5:] if x.next.attrs['xlink:title'].split('\n')[0].startswith("Name:") else x.next.attrs['xlink:title'].split('\n')[0]) == node_name), None)
        matching_g_a_node_tag_opt = next((x for x in g_a_node_tags_opt if (x.next.attrs['xlink:title'].split('\n')[0][5:] if x.next.attrs['xlink:title'].split('\n')[0].startswith("Name:") else x.next.attrs['xlink:title'].split('\n')[0]) == source_name), None)

        if matching_g_a_node_tag_trt:
            # 取出匹配g标签的id   
            g_id_trt = matching_g_a_node_tag_trt['id']
            g_id_opt =matching_g_a_node_tag_opt['id'] if matching_g_a_node_tag_opt else 'xxxxyyyyzzzz'
            # 将id添加到tr标签中
            tr_tag['onclick'] = f"parent.scrollToElementTRT('{g_id_trt}'); parent.scrollToElementOPT('{g_id_opt}')"

    tr_tags_opt = html_soup_opt.find_all('tr')
    for tr_tag in tr_tags_opt[1:]:
        # 找到tr下的第一个td内容，即node name
        # td_content = tr_tag.find('td').text
        node_name=str(tr_tag).split('\n')[2][4:-5]
        source_name=str(tr_tag).split('\n')[3][4:-5]
        # 在SVG中查找匹配的g标签
        matching_g_a_node_tag_trt = next((x for x in g_a_node_tags_trt if (x.next.attrs['xlink:title'].split('\n')[0][5:] if x.next.attrs['xlink:title'].split('\n')[0].startswith("Name:") else x.next.attrs['xlink:title'].split('\n')[0]) == source_name), None)
        matching_g_a_node_tag_opt = next((x for x in g_a_node_tags_opt if (x.next.attrs['xlink:title'].split('\n')[0][5:] if x.next.attrs['xlink:title'].split('\n')[0].startswith("Name:") else x.next.attrs['xlink:title'].split('\n')[0]) == node_name), None)

        if matching_g_a_node_tag_opt:
            # 取出匹配g标签的id   
            g_id_opt = matching_g_a_node_tag_opt['id']
            g_id_trt =matching_g_a_node_tag_trt['id'] if matching_g_a_node_tag_trt else 'xxxxyyyyzzzz'
            # 将id添加到tr标签中
            tr_tag['onclick'] = f"parent.scrollToElementTRT('{g_id_trt}'); parent.scrollToElementOPT('{g_id_opt}')"
    
    # 将修改后的HTML写回文件
    with open(html_file_trt, 'w') as f:
        f.write(html_soup_trt.prettify())
    with open(html_file_opt, 'w') as f:
        f.write(html_soup_opt.prettify())

def add_num_to_report(model_name_trt,model_name_opt,num_op_trt,num_op_opt,num_op_public,report_file):
    with open(report_file, 'r') as f:
        report_content = f.read()
    report_soup = BeautifulSoup(report_content, 'html.parser')
    tag = report_soup.find('strong', id='model_name_trt')
    tag.string = str(model_name_trt)
    tag = report_soup.find('strong', id='model_name_opt')
    tag.string = str(model_name_opt)
    tag = report_soup.find('strong', id='num_op_trt')
    tag.string = str(num_op_trt)
    tag = report_soup.find('strong', id='num_op_opt')
    tag.string = str(num_op_opt)
    tag = report_soup.find('strong', id='num_op_public')
    tag.string = str(num_op_public)
    with open(report_file, 'w') as f:
        f.write(report_soup.prettify())

def get_model_name(profile_path):
    if not os.path.exists(profile_path):
        print(f"Error: 文件目录{profile_path}不存在")
        print(">>> >>> profile文件检测失败")
        return False
    profile_files = glob.glob(os.path.join(profile_path, "*.profile.json"))
    if len(profile_files) > 1:
        print(f"Error: 多个*.profile.json文件: {profile_files}")
        print(">>> >>> profile文件检测失败")
        return False
    elif len(profile_files) == 0:
        print(f"Error: 缺少*.profile.json文件")
        print(">>> >>> profile文件检测失败")
        return False
    profile_file = profile_files[0]
    model_name = os.path.basename(profile_file)[:-13]
    return model_name

def compare_trt(trt_profile_path,opt_profile_path,save_path):
    trt_profile_path = os.path.normpath(trt_profile_path)
    opt_profile_path = os.path.normpath(opt_profile_path)
    save_path = os.path.normpath(save_path)
    model_name_opt=get_model_name(opt_profile_path)
    model_name_trt=get_model_name(trt_profile_path)
    opt_graph_json_file=os.path.join(opt_profile_path, f'{model_name_opt}.graph.json') 
    trt_graph_json_file=os.path.join(trt_profile_path, f'{model_name_trt}.graph.json')
    plan_opt=EnginePlan(opt_graph_json_file)
    plan_trt=EnginePlan(trt_graph_json_file)
    df_opt = plan_opt._df
    df_trt=plan_trt._df

    inputs_opt=[input.name for input in plan_opt.get_bindings()[0]]
    inputs_trt=[input.name for input in plan_trt.get_bindings()[0]]

    df_opt=simplify_dataframe(df_opt)
    df_trt=simplify_dataframe(df_trt)

    #找出df_opt和df_trt间Name相同的数据项组成新dataframe
    df_public_opt = pd.DataFrame(columns=df_opt.columns)
    df_public_trt = pd.DataFrame(columns=df_opt.columns)
    for index_opt, row_opt in df_opt.iterrows():
        for index_trt, row_trt in df_trt.iterrows():
            if name_equal(row_trt['Name'],row_opt['Name']):
                df_public_opt = df_public_opt.append(row_opt, ignore_index=True)
                df_public_trt = df_public_trt.append(row_trt, ignore_index=True)
                #df_opt、df_trt去除相同数据项
                df_opt = df_opt.drop(index_opt)
                df_trt = df_trt.drop(index_trt)
                break  

    if len(df_public_opt)<10:
        user_input = input("两个模型的共有算子个数小于10，模型差异过大，寻找共有来源的耗时可能非常长，是否继续运行？(y/n)")
        if user_input.upper()=="Y":
                #TODO 处理部分算子<多为PWD和reformat>在plan中的来源为自身，但实则在svg中有上游来源
                #根据节点输入输出tensor找到所有其公共来源节点,只保留第一个
                #首先处理opt
                for index_opt, row_opt in df_opt.iterrows():
                    top_source_nodes=find_top_source_nodes(index_opt,df_opt)#获取到所有非公有顶级节点，set:index
                    otherside_public_source_nodes=find_public_source_nodes(top_source_nodes,df_opt,df_public_opt,df_public_trt,inputs_trt)
                    df_opt.at[index_opt,'source']=list(otherside_public_source_nodes)[0] if len(otherside_public_source_nodes)>0 else None

                #再处理trt
                for index_trt, row_trt in df_trt.iterrows():
                    top_source_nodes=find_top_source_nodes(index_trt,df_trt)#获取到所有非公有顶级节点，set:index
                    otherside_public_source_nodes=find_public_source_nodes(top_source_nodes,df_trt,df_public_trt,df_public_opt,inputs_opt)
                    df_trt.at[index_trt,'source']=list(otherside_public_source_nodes)[0] if len(otherside_public_source_nodes)>0 else None

                # df_opt=df_opt.loc[:,['Name', 'subtype' ,'Input Format','Output Format','source']]
                # df_trt=df_trt.loc[:,['Name', 'subtype' ,'Input Format','Output Format','source']]
                try:
                    df_opt=df_opt.loc[:,['Name','source','subtype']]
                    df_trt=df_trt.loc[:,['Name','source','subtype']]
                except:
                    print("两个模型没有差异，请确认profile路径是否有误")
                    return
                df_opt = df_opt.rename(columns={'Name': '差异算子'})
                df_trt = df_trt.rename(columns={'Name': '差异算子'})

                #生成dataframe
                df_opt.to_html(save_path+"/opt.html")
                df_trt.to_html(save_path+"/trt.html")

                #复制svg、report
                shutil.copy2(opt_profile_path+"/report/structure/engine.svg",save_path+"/opt.svg")
                shutil.copy2(trt_profile_path+"/report/structure/engine.svg",save_path+"/trt.svg")
                sample_script_path = os.path.abspath(sys.argv[0])
                sample_script_directory = os.path.dirname(sample_script_path)
                shutil.copy2(sample_script_directory+"/html_template/reports/compare_trt.html",save_path+"/compare_trt.html")

                add_scroll_to_html_table(html_file_trt=save_path+"/trt.html",html_file_opt=save_path+"/opt.html",svg_file_trt=save_path+"/trt.svg",svg_file_opt=save_path+"/opt.svg")
                add_num_to_report(model_name_trt,
                                    model_name_opt,
                                    len(df_trt),
                                    len(df_opt),
                                    len(df_public_opt),
                                    save_path+"/compare_trt.html")
        else:
            sys.exit()

def main():
    parser = argparse.ArgumentParser(description='对比优化前后的trt差异')
    parser.add_argument('-b','--before',dest='before',type=str,required=True, help='优化前模型profile路径')
    parser.add_argument('-a','--after',dest='after',type=str,required=True, help='优化后模型profile路径')
    parser.add_argument('-s','--save',dest='save',type=str,required=True, help='报告保存路径')
    args = parser.parse_args()

    compare_trt(args.before,args.after,args.save)

if __name__=='__main__':
    main()

