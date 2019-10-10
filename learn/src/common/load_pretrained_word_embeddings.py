import numpy as np
import logging
logging.basicConfig(level = logging.DEBUG)
from symbols import UNK_INDEX, UNK_SYMBOL, UNK_VALUE
from tensorflow.keras.layers import Embedding
from gensim.test.utils import datapath, get_tmpfile
from gensim.models import KeyedVectors
from gensim.scripts.glove2word2vec import glove2word2vec
import os
import pickle

class EmbeddingConverter:
    """
    读取一个预训练的word2vec文件，返回一个keras的embedding层
    """
    def __init__(self, fn ,fntype='glove'):
        """
        创建一个预训练词向量文件到embedding层的转换器
        Parameters:
            fn - 文件名
            fntype - 预训练词向量文件类型，glove为glove官方文件类型，txt为gensim的文本格式，bin为gensim的二进制格式
        """
        self.fn = fn
        self.fntype = fntype
        logging.debug("读取预训练文件: {} ...".format(self.fn))
        self._load()
        logging.debug("Done!")

    def _load(self):
        """
        读取预训练文件，构建映射字典，比较费时
        """
        if os.path.exists(self.fn+'.pickle'):
            with open(self.fn+'.pickle','rb') as preFile:
                preDict=pickle.load(preFile)
                self.word_index=preDict['word_index']
                self.word2VecModel=preDict['word2VecModel']
                self.dim=preDict['dim']
                self.emb=preDict['emb']
                self.vocab_size=preDict['vocab_size']
        else:
            self.word_index = {UNK_SYMBOL : UNK_INDEX}
            emb = []
            if self.fntype=='glove':
                glove_file = datapath(self.fn)
                tmp_file = get_tmpfile(self.fn+".2gensim.txt")
                glove2word2vec(glove_file, tmp_file)
                self.word2VecModel = KeyedVectors.load_word2vec_format(tmp_file)
            elif self.fntype=='txt':
                self.word2VecModel = KeyedVectors.load_word2vec_format(self.fn)
            elif self.fntype=='bin':
                self.word2VecModel = KeyedVectors.load_word2vec_format(self.fn,binary=True)

            for key in self.word2VecModel.wv.vocab.keys():
                word=key
                coefs = np.asarray(self.word2VecModel[key], dtype='float32')
                self.dim = len(coefs)
                self.word_index[word] = len(emb) + 1
                emb.append(coefs)
            # Add UNK at the first index in the table
            self.emb = np.array([UNK_VALUE(self.dim)] + emb)
            # Set the vobabulary size
            self.vocab_size = len(self.emb)
            preDict={
                'word_index':self.word_index,
                'word2VecModel':self.word2VecModel,
                'dim':self.dim,
                'emb':self.emb,
                'vocab_size':self.vocab_size
            }
            with open(self.fn+'.pickle','wb') as preFile:
                pickle.dump(preDict,preFile)

    def get_word_index(self, word, lower = True):
        """
        Get the index of a given word (int).
        If word doesnt exists, returns UNK.
        lower - controls whether the word should be lowered before checking map
        """
        if lower:
            word = word.lower()
        return self.word_index[word] \
            if (word in self.word_index) else UNK_INDEX

    def get_embedding_matrix(self):
        """
        Return an embedding matrix for use in a Keras Embeddding layer
        https://blog.keras.io/using-pre-trained-word-embeddings-in-a-keras-model.html
        word_index - Maps words in the dictionary to their index (one-hot encoding)
        """
        return self.emb

    def get_keras_embedding(self, **args):
        """
        Get a Keras Embedding layer, loading this embedding as pretrained weights
        The additional arguments given to this function are passed to the Keras Embbeding constructor.
        """
        return Embedding(self.vocab_size,
                         self.dim,
                         weights = [self.get_embedding_matrix()],
                         **args)