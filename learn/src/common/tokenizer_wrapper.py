import jieba.posseg as pseg
import spacy
from spacy.tokens import Doc
import re

class WhitespaceTokenizer(object):
    """
    White space tokenizer - assumes all text space-separated
    https://spacy.io/docs/usage/customizing-tokenizer
    """
    def __init__(self, vocab):
        """
        Get an initialized spacy object
        """
        self.vocab = vocab

    def __call__(self, text):
        """
        Call this tokenizer - just split based on space
        """
        words = re.split(r' +', text) # Allow arbitrary number of spaces
        # All tokens 'own' a subsequent space character in this tokenizer
        spaces = [True] * len(words)
        return Doc(self.vocab, words=words, spaces=spaces)

class tokenizer(object):
    def __init__(self,i,tag,word):
        self.i=i
        self.tag_=tag
        self.word=word

class TokenizerWrapper:
    def __init__(self,lang='en'):
        self.lang=lang
        if lang=='en':
            nlp = spacy.load("en_core_web_sm")
            nlp.tokenizer = WhitespaceTokenizer(nlp.vocab)
            self.enparser = nlp
            # self.enparser = spacy.load('en',
            #             create_make_doc = WhitespaceTokenizer)
    
    def parser(self,text):
        if self.lang=='en':
            return self.enparser(text)
        elif lang=='cn':
            tags = pseg.cut(''.join(text.split(' ')))
            result=[]
            i=0
            for word,flag in tags:
                res=tokenizer(i,flag,word)
                result.append(res)
                i+=1
            return result

    def word_tokenize(self,text):
        if self.lang=='en':
            words=self.enparser(text)
            res=[]
            for word in words:
                res.append(word.text)
            return res