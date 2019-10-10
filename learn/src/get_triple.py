import sys
args=sys.argv

import os
os.environ["CUDA_VISIBLE_DEVICES"] = args[1]
begin=int(args[2])
end=int(args[3])

from oie import OIE
model_path='/data/liupq/transformer-oie/models/23_09_2019_15_15/'
oie_model=OIE(model_path)



sep1='*#*' #组外
sep2='|||' #组内
input_file_path='/data/liupq/supervised-oie/ensrc/Twitter.big.txt'
output_q_path='/data/liupq/transformer-oie/triple/Twitter.big%s-%s.q.triple'%(begin,end)
output_a_path='/data/liupq/transformer-oie/triple/Twitter.big%s-%s.a.triple'%(begin,end)


with open(input_file_path,'r',encoding='utf-8') as input_file,open(output_q_path,'w',encoding='utf-8') as output_q_path,\
    open(output_a_path,'w',encoding='utf-8') as output_a_path:
    qlist=[]
    alist=[]
    for i,line in enumerate(input_file):
        if i>=begin and i<end:
            line=line.strip().split('\t')
            if len(line)<2 or line[0].strip=='' or line[1].strip=='':
                continue
            qlist.append(line[0].strip())
            alist.append(line[1].strip())
        if i>=end:
            break
    res_list=oie_model.get_triples_v2(qlist)
    for i,line in enumerate(res_list):
        # print(line)
        res=[sep2.join(triple) for triple in line]
        if res==[]:
            res=[sep2.join(qlist[i].split())]
        res=sep1.join(res)
        output_q_path.write(qlist[i]+'\t'+res+'\n')
    res_list=oie_model.get_triples_v2(alist)
    for i,line in enumerate(res_list):
        # print(line)
        res=[sep2.join(triple) for triple in line]
        if res==[]:
            res=[sep2.join(alist[i].split())]
        res=sep1.join(res)
        output_a_path.write(alist[i]+'\t'+res+'\n')
            