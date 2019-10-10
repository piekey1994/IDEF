import sys
args=sys.argv

begin=int(args[1])
end=int(args[2])


from extractor import Extractor
import jieba
import numpy as np
ex_class=Extractor()

sep1='*#*' 
sep2='|||' 
input_file_path='D:\\IOM\\dataset\\cn_conv\\weibo.txt'
output_q_path='weibo%s-%s.q.triple'%(begin,end)
output_a_path='weibo%s-%s.a.triple'%(begin,end)

# input_file_path='D:\\ieee\\code\\idef\\ex\\3\\cn.conv.txt'
# output_q_path='cn.conv%s-%s.q.triple'%(begin,end)
# output_a_path='cn.conv%s-%s.a.triple'%(begin,end)


with open(input_file_path,'r',encoding='utf-8') as input_file,open(output_q_path,'w',encoding='utf-8') as output_q_path,\
    open(output_a_path,'w',encoding='utf-8') as output_a_path:
    qlist=[]
    alist=[]
    for i,line in enumerate(input_file):
        if i>=begin and i<end:
            line=line.strip().split('\t')
            if len(line)!=2 or line[0].strip=='' or line[1].strip=='':
                continue
            qlist.append(line[0].strip())
            alist.append(line[1].strip())
        if i>=end:
            break
    res_list=[]
    for i,q in enumerate(qlist):
        if i %(int(np.ceil(len(qlist)/100))) == 0:
            print(i / len(qlist))
        res_list.append(ex_class.chunk_str(q))
    for i,line in enumerate(res_list):
        # print(line)
        res=[sep2.join(triple) for triple in line]
        if res==[]:
            res=[sep2.join(jieba.lcut(qlist[i]))]
        res=sep1.join(res)
        output_q_path.write(qlist[i]+'\t'+res+'\n')
    res_list=[]
    for i,a in enumerate(alist):
        if i %(int(np.ceil(len(alist)/100))) == 0:
            print(i / len(alist))
        res_list.append(ex_class.chunk_str(a))
    for i,line in enumerate(res_list):
        # print(line)
        res=[sep2.join(triple) for triple in line]
        if res==[]:
            res=[sep2.join(jieba.lcut(alist[i]))]
        res=sep1.join(res)
        output_a_path.write(alist[i]+'\t'+res+'\n')
            