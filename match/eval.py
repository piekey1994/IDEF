import gensim
import numpy as np

class Evaluation(object):
    def __init__(self,word2vec_path,word2vec_model=None):
        if word2vec_model!=None:
            self.word2vec_model=word2vec_model
        else:
            self.word2vec_model=gensim.models.Word2Vec.load(word2vec_path)

    def levenshtein_distance(self,first, second):
        def sigmoid(x):
            return 1.0 / (1.0 + np.exp(-x))
        sentence1_len, sentence2_len = len(first), len(second)
        maxlen = max(sentence1_len, sentence2_len)
        if sentence1_len > sentence2_len:
            first, second = second, first

        distances = range(len(first) + 1)
        for index2, char2 in enumerate(second):
            new_distances = [index2 + 1]
            for index1, char1 in enumerate(first):
                if char1==char2:
                    new_distances.append(distances[index1])
                else:
                    new_distances.append(1 + min((distances[index1],
                                                distances[index1 + 1],
                                                new_distances[-1])))
            distances = new_distances
        levenshtein = distances[-1]
        d = float((maxlen - levenshtein)/maxlen)
        # smoothing
        s = (sigmoid(d * 6) - 0.5) * 2
        # print("smoothing[%s| %s]: %s -> %s" % (sentence1, sentence2, d, s))
        return s

    def get_gm_mat(self,word1,word2):
        if word1==None or word2==None or word1=='' or word2=='' or word1==[] or word2==[]:
            return []
        if isinstance(word1,str):
            word1=[word1]
        if isinstance(word2,str):
            word2=[word2]
        A = word1
        B = word2
        scores=[]
        for w1 in A:
            ss=[]
            for w2 in B:
                try:
                    ss.append(self.word2vec_model.similarity(w1,w2))
                except:
                    # print('bianji',w1,w2)
                    ss.append(self.levenshtein_distance(w1,w2))
            scores.append(ss)
        return scores

    def get_tm_score(self,triple1,triple2):
        words1=' '.join(triple1).lower().split()
        words2=' '.join(triple2).lower().split()
        min_len=min(len(words1),len(words2))
        scores=self.get_gm_mat(words1,words2)
        A=words1
        B=words2
        La = []
        Lb = []
        for i in range(len(A)):
            La.append(max(scores[i]))
        La.sort(reverse=True)
        La = La[:min_len]
        La = sum(La)/len(La)
        for i in range(len(B)):
            maxnum=0
            for j in range(len(A)):
                maxnum = scores[j][i] if scores[j][i]>maxnum else maxnum
            Lb.append(maxnum) 
        Lb.sort(reverse=True)
        Lb = Lb[:min_len]
        Lb = sum(Lb)/len(Lb)
        return (La+Lb)/2

    def punish(self,score,m=0.95,low=0.85):
        if score>m:
            return ((m-low)*score+(low-1)*m)/(m-1)
        else:
            return score

    def eval(self,triple_list1,triple_list2):
        scores=[]
        for t1 in triple_list1:
            scores.append(max([self.get_tm_score(t1,t2) for t2 in triple_list2]))
        # print(scores)
        res=sum(scores)/len(scores)
        return self.punish(res)
    
    def eval_v2(self,triple_list1,triple_list2):
        scores=[]
        e_list1=[]
        for t in triple_list1:
            t=[e.split() for e in t]
            e_list1+=t
        e_list2=[]
        for t in triple_list2:
            t=[e.split() for e in t]
            e_list2+=t
        for e1 in e_list1:
            scores.append(max([self.get_tm_score(e1,e2) for e2 in e_list2]))
        scores.sort(reverse=True)
        scores=scores[:min(len(e_list1),len(e_list2))]
        res=sum(scores)/len(scores)
        return self.punish(res)
    
    def eval_v3(self,triple_list1,triple_list2,words1,words2):
        if triple_list1==[] or triple_list2==[]:
            res=self.get_tm_score(words1,words2)
        else:
            scores=[]
            e_list1=[]
            for t in triple_list1:
                t=[e.split() for e in t]
                e_list1+=t
            e_list2=[]
            for t in triple_list2:
                t=[e.split() for e in t]
                e_list2+=t
            for e1 in e_list1:
                scores.append(max([self.get_tm_score(e1,e2) for e2 in e_list2]))
            scores.sort(reverse=True)
            scores=scores[:min(len(e_list1),len(e_list2))]
            res=sum(scores)/len(scores)
        return self.punish(res)