# -*- coding: utf-8 -*-
import configparser
conf = configparser.ConfigParser()
conf.read('conf.ini')

import itertools
from triple import Triple
from triple import Relation
from triple import Entity
from pyltp import Segmentor
from pyltp import Postagger
from pyltp import NamedEntityRecognizer
from pyltp import SentenceSplitter
from pyltp import Parser

class Extractor():

    def __init__(self):
        self.__triple_list = []
        self.__segmentor = Segmentor()
        self.__postagger = Postagger()
        self.__recognizer = NamedEntityRecognizer()
        self.__parser = Parser()
        self.__words_full_list = []
        self.__netags_full_list = []
        self.load()

    @property
    def triple_list(self):
        return self.__triple_list

    def load(self):
        ltp_dir=conf.get('config','ltp_dir')
        self.__segmentor.load(ltp_dir+'cws.model')
        self.__postagger.load(ltp_dir+'pos.model')
        self.__recognizer.load(ltp_dir+'ner.model')
        self.__parser.load(ltp_dir+'parser.model')

    def release(self):
        self.__segmentor.release()
        self.__postagger.release()
        self.__recognizer.release()
        self.__parser.release()

    def clear(self):
        self.__triple_list = []
        self.__words_full_list = []
        self.__netags_full_list = []

    def chunk_str(self, data):
        self.clear()
        sents = SentenceSplitter.split(data.strip())
        offset = 0
        for sent in sents:
            try:
                words = self.__segmentor.segment(sent)
                postags = self.__postagger.postag(words)
                netags = self.__recognizer.recognize(words, postags)
                arcs = self.__parser.parse(words, postags)
                self.__words_full_list.extend(list(words))
                self.__netags_full_list.extend(list(netags))
                self.chunk_sent(list(words), list(postags), list(arcs), offset)
                offset += len(list(words))
                
            except Exception as e:
                print(str(e))
                offset += len(list(words))
        return [t.to_list() for t in self.__triple_list]

    def chunk_sent(self, words, postags, arcs, offset):
        root = [i+1 for i,x in enumerate(arcs) if x.relation == 'HED']
        if len(root) > 1:
            raise Exception('More than 1 HEAD arc is detected!')
        root = root[0]
        relations = [i+1 for i, x in enumerate(arcs) if x.head == root and x.relation == 'COO']
        relations.insert(0,root)

        for rel in relations:
            e1=None
            left_arc = [i+1 for i, x in enumerate(arcs) if x.head == rel and x.relation == 'SBV']
            if len(left_arc) == 0:
                for i in range(rel-2,-1,-1):
                    x=arcs[i]
                    if x.head == rel:
                        left_arc=[i+1]
                        break
            if len(left_arc) > 0:
                left_arc = left_arc[-1]
                leftmost = find_farthest_att(arcs, left_arc)
                e1 = Entity(1, [words[i] for i in range(leftmost-1, left_arc)], offset + leftmost-1)

            right_arc = [i+1 for i, x in enumerate(arcs) if x.head == rel and x.relation == 'VOB']

            e2_list = []
            if not right_arc:
                e2 = None
                e2_list.append(e2)
            else:
                right_ext = find_farthest_vob(arcs, right_arc[0])

                items = [i+1 for i, x in enumerate(arcs) if x.head == right_ext and x.relation == 'COO']
                items = right_arc + items

                count = 0
                for item in items:
                    leftmost = find_farthest_att(arcs, item)


                    e2 = None

                    if count == 0:
                        e2 = Entity(2, [words[i] for i in range(leftmost-1, right_ext)], offset+leftmost-1)
                    else:
                        p1 = range(leftmost-1, right_arc[0]-1)
                        p2 = range(item-1, find_farthest_vob(arcs, item))
                        e2 = Entity(2, [words[i] for i in itertools.chain(p1, p2)])

                    e2_list.append(e2)
                    count += 1
            for e2 in e2_list:
                if e1==None:
                    e1=Entity(1,[])
                if e2==None:
                    e2=Entity(2,[])
                r=Relation(words[rel-1])
                t=Triple(e1,e2,r)
                self.__triple_list.append(t)


def find_farthest_att(arcs, loc):
    att = [i+1 for i, x in enumerate(arcs) if x.head == loc and (x.relation == 'ATT' or x.relation == 'SBV')]
    if not att:
        return loc
    else:
        return find_farthest_att(arcs, min(att))


def find_farthest_vob(arcs, loc):
    vob = [i+1 for i, x in enumerate(arcs) if x.head == loc and x.relation == 'VOB']
    if not vob:
        return loc
    else:
        return find_farthest_vob(arcs, max(vob))


if __name__ == "__main__":
    sent='你喜欢那个衣服吗'
    ex_class=Extractor()
    res=ex_class.chunk_str(sent)
    for t in res:
        print(t)