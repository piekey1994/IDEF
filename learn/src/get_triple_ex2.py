import sys
args=sys.argv

import os
os.environ["CUDA_VISIBLE_DEVICES"] = args[1]

from oie import OIE
model_path='/data/liupq/transformer-oie/models/23_09_2019_15_15/'
oie_model=OIE(model_path)

sep1='*#*' 
sep2='|||' 

from functools import wraps
import time
 
def func_timer(function):
    '''
    用装饰器实现函数计时
    :param function: 需要计时的函数
    :return: None
    '''
    @wraps(function)
    def function_timer(*args, **kwargs):
        print('[Function: {name} start...]'.format(name = function.__name__))
        t0 = time.time()
        result = function(*args, **kwargs)
        t1 = time.time()
        print('[Function: {name} finished, spent time: {time:.2f}s]'.format(name = function.__name__,time = t1 - t0)) 
        return result
    return function_timer


@func_timer
def create_triples(input_datas):
    res_list=oie_model.get_triples_v2(input_datas)
    text_list=[]
    for i,line in enumerate(res_list):
        res=[sep2.join(triple) for triple in line]
        if res==[]:
            res=[sep2.join(input_datas[i].split())]
        res=sep1.join(res)
        text_list.append(input_datas[i]+'\t'+res+'\n')
    return text_list

def read_data(file_path):
    with open(file_path,'r',encoding='utf-8') as key_file:
        input_datas=[line.strip() for line in key_file.readlines()]
        return input_datas

pre='Twitter.100w.test.key'
key_path='/data/liupq/transformer-oie/data/'
input_datas=read_data(key_path+pre)
output_datas=create_triples(input_datas)
with open('/data/liupq/transformer-oie/triple/'+pre+'.q.triple','w',encoding='utf-8') as output_file:
    output_file.writelines(output_datas)

pres=['Twitter.100w.test.att','Twitter.100w.test.beam1.num','Twitter.100w.test.nmt','Twitter.100w.test.attbeam1']
for pre in pres:
    input_datas=read_data(key_path+pre+'.output')
    output_datas=create_triples(input_datas)
    with open('/data/liupq/transformer-oie/triple/'+pre+'.a.triple','w',encoding='utf-8') as output_file:
        output_file.writelines(output_datas)