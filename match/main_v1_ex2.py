from eval import Evaluation
# import nltk
# emb_path='D:\\IOM\\word2vec\\GoogleNews-vectors-negative300.bin'
# import jieba
emb_path='D:\\IOM\\word2vec\\merge_sgns_bigram_char300.bin'
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
        triple_es = triple_str.split(sep1)
        # #没有三元组的修正
        # if len(triple_es)>3:
        #     return []
        triples.append(triple_es)
    return triples

# pres=['Twitter.100w.test.att','Twitter.100w.test.attbeam.num','Twitter.100w.test.nmt']
# key='Twitter.100w.test.key'
# key_path='D:\\ieee\\code\\idef\\ex\\2\\new\\'

pres=['weibo.100w.test.att','weibo.100w.test.beam1.num','weibo.100w.test.nmt']
key='weibo.100w.test.key'
key_path='D:\\ieee\\code\\idef\\rule\\output\\'

import time

for pre in pres:
    print(pre)
    q_file_path=key_path+key+'.q.triple'
    t_file_path=key_path+pre+'.a.triple'
    res_file_path='output/%s.score'%(pre)
    # q_file_path='data/Twitter.big%s-%s.q.triple'%(begin,end)
    # t_file_path='data/Twitter.big%s-%s.a.triple'%(begin,end)
    # res_file_path='output/Twitter.v1%s-%s.score'%(begin,end)

    with open(q_file_path,'r',encoding='utf-8') as q_file,\
        open(t_file_path,'r',encoding='utf-8') as t_file,\
        open(res_file_path,'w',encoding='utf-8') as res_file:
        scores=[]
        begin_time=time.time()
        for q_line,t_line in zip(q_file,t_file):
            q_line=q_line.strip().split('\t')
            t_line=t_line.strip().split('\t')
            q_triples=cut_triples(q_line[1])
            t_triples=cut_triples(t_line[1])
            if len(t_triples[0])>3:
                score=0
            else:
                score=eval_class.eval(q_triples,t_triples)
            scores.append(score)
            res_file.write(str(score)+'\n')
        end_time=time.time()
        # print(end_time-begin_time)
    highnum=0
    for score in scores:
        if score>=0.6:
            highnum+=1
    print(highnum,highnum/50000)
    print(sum(scores),sum(scores)/len(scores))