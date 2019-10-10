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


args=[(0,100)]
for arg in args:
    print(arg)
    begin=arg[0]
    end=arg[1]
    q_file_path='data/cn.conv%s-%s.q.triple'%(begin,end)
    t_file_path='data/cn.conv%s-%s.a.triple'%(begin,end)
    res_file_path='output/cn.conv.v1%s-%s.score'%(begin,end)

    with open(q_file_path,'r',encoding='utf-8') as q_file,\
        open(t_file_path,'r',encoding='utf-8') as t_file,\
        open(res_file_path,'w',encoding='utf-8') as res_file:
        num=0
        for q_line,t_line in zip(q_file,t_file):
            q_line=q_line.strip().split('\t')
            t_line=t_line.strip().split('\t')
            q_triples=cut_triples(q_line[1])
            t_triples=cut_triples(t_line[1])
            if len(t_triples[0])>3:
                score=0
                num+=1
            else:
                score=eval_class.eval(q_triples,t_triples)
            # score=eval_class.eval_v3(q_triples,t_triples,nltk.word_tokenize(q_line[0]),nltk.word_tokenize(t_line[0]))
            # score=eval_class.eval_v3(q_triples,t_triples,jieba.lcut(q_line[0]),jieba.lcut(t_line[0]))
            res_file.write(str(score)+'\t'+str(q_triples)+'\t'+str(t_triples)+'\n')
        print(num)
print('finish')