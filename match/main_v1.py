from eval import Evaluation
# import nltk
emb_path='D:\\IOM\\word2vec\\GoogleNews-vectors-negative300.bin'
# import jieba
# emb_path='D:\\IOM\\word2vec\\merge_sgns_bigram_char300.bin'
from gensim.models import KeyedVectors
wv_from_bin = KeyedVectors.load_word2vec_format(emb_path, binary=True)
eval_class=Evaluation('',wv_from_bin)

sep2='*#*' 
sep1='|||' 

def cut_triples(line):
    global notriple
    line=line.strip()
    triples=[]
    for triple_str in line.split(sep2):
        triple_es = [t for t in triple_str.split(sep1) if t!='']
        #没有三元组的修正
        # if len(triple_es)>3:
        #     return []
        triples.append(triple_es)
    return triples

args=[(0,250000),(250000,500000),(500000,750000),(750000,1000000),(1000000,1250000)]
# args=[(0,250000),(250000,500000),(500000,750000),(750000,1000000),(1000000,1250000),(1250000,1500000),(1500000,1750000),(1750000,2000000)]
for arg in args:
    print(arg)
    begin=arg[0]
    end=arg[1]
    q_file_path='data/weibo%s-%s.q.triple'%(begin,end)
    t_file_path='data/weibo%s-%s.a.triple'%(begin,end)
    res_file_path='output/weibo.v1%s-%s.score'%(begin,end)
    # q_file_path='data/Twitter.big%s-%s.q.triple'%(begin,end)
    # t_file_path='data/Twitter.big%s-%s.a.triple'%(begin,end)
    # res_file_path='output/Twitter.v1%s-%s.score'%(begin,end)

    with open(q_file_path,'r',encoding='utf-8') as q_file,\
        open(t_file_path,'r',encoding='utf-8') as t_file,\
        open(res_file_path,'w',encoding='utf-8') as res_file:
        
        for q_line,t_line in zip(q_file,t_file):
            q_line=q_line.strip().split('\t')
            t_line=t_line.strip().split('\t')
            q_triples=cut_triples(q_line[1])
            t_triples=cut_triples(t_line[1])
            if q_triples==[] or t_triples==[]:
                score=-1
            else:
                score=eval_class.eval(q_triples,t_triples)
            res_file.write(str(score)+'\n')
print('finish')