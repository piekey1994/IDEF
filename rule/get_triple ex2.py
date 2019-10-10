from extractor import Extractor
import numpy as np
ex_class=Extractor()

sep1='*#*' 
sep2='|||' 

def create_triples(input_datas):
    res_list=[]
    for i,q in enumerate(input_datas):
        if i %(int(np.ceil(len(input_datas)/100))) == 0:
            print(i / len(input_datas))
        res_list.append(ex_class.chunk_str(''.join(q)))
    text_list=[]
    for i,line in enumerate(res_list):
        res=[sep2.join(triple) for triple in line]
        if res==[]:
            res=[sep2.join(input_datas[i])]
        res=sep1.join(res)
        text_list.append(''.join(input_datas[i])+'\t'+res+'\n')
    return text_list

def read_data(file_path):
    with open(file_path,'r',encoding='utf-8') as key_file:
        input_datas=[line.strip().split() for line in key_file.readlines()]
        return input_datas

pre='weibo.100w.test.key'
key_path='D:\\ieee\\code\\idef\\ex\\2\\'
input_datas=read_data(key_path+pre)
output_datas=create_triples(input_datas)
with open('output/'+pre+'.q.triple','w',encoding='utf-8') as output_file:
    output_file.writelines(output_datas)

pres=['weibo.100w.test.att','weibo.100w.test.beam1.num','weibo.100w.test.nmt']
for pre in pres:
    input_datas=read_data(key_path+pre+'.output')
    output_datas=create_triples(input_datas)
    with open('output/'+pre+'.a.triple','w',encoding='utf-8') as output_file:
        output_file.writelines(output_datas)