from rnn_oie import RNNOIE_model
import logging
logging.basicConfig(level = logging.DEBUG)
import json
from tensorflow.keras.layers import Layer
from transformer import Attention,Position_Embedding
from tensorflow.keras.layers import Input, Dense, LSTM, Embedding, \
    TimeDistributed, concatenate, Bidirectional, Dropout, BatchNormalization
from tensorflow.keras.models import Model
import os
from common.tokenizer_wrapper import TokenizerWrapper
from common.symbols import get_tags_from_lang
import tensorflow as tf
from collections import defaultdict
from sample import Sample,Pad_sample
import numpy as np

class LayerReset(Layer):
    def call(self, x):
        return tf.map_fn(lambda y:tf.gather(y[0],y[1]) ,[x[0],x[1]],dtype=x[0].dtype)

class TransformerTree(RNNOIE_model):
    def pad_head_input_num(self,sequences,maxlen = None):
        ret = []

        # Determine the maxlen
        max_value = max(map(len, sequences))
        if maxlen is None:
            maxlen = max_value

        # Pad / truncate (done this way to deal with np.array)
        for sequence in sequences:
            cur_seq = list(sequence[:maxlen])
            cur_seq.extend([len(sequence)+i for i in range(maxlen - len(sequence))])
            assert(maxlen==len(cur_seq))
            ret.append(cur_seq)
        return ret

    def encode_inputs(self, sents):
        """
        将句子序列转换为神经网络的输入序列
        Parameters:
            sents - 已经通过get_sents_from_df分割好的句子序列
        Return:
            拥有三个key的字典"word_inputs", "predicate_inputs", "postags_inputs"
            其中每个value都是一个以句子为元素的list，以上三个句子的特征表示组合为一个训练数据
        """
        word_inputs = []
        pred_inputs = []
        pos_inputs = []
        head_inputs = []
        tkw = TokenizerWrapper('en')
        TAGS = get_tags_from_lang('en')

        # Preproc to get all preds per run_id
        # Sanity check - make sure that all sents agree on run_id
        assert(all([len(set(sent.run_id.values)) == 1
                    for sent in sents]))
        run_id_to_pred = dict([(int(sent.run_id.values[0]),
                                self.get_head_pred_word(sent))
                               for sent in sents])

        # Construct a mapping from running word index to pos
        word_id_to_pos = {}
        for sent in sents:
            indices = sent.index.values
            words = sent.word.values
            heads = []
            for index, word in zip(indices,
                                   tkw.parser(" ".join(words))):
                word_id_to_pos[index] = word.tag_
                heads.append(word.head.text)
            # print('%d %d'%(len(heads),len(words)))
            # assert(len(words)==len(heads))
            head_inputs.append(heads)
        fixed_size_sents = self.get_fixed_size(sents)
        fixed_size_heads = self.get_fixed_size(head_inputs)
        head_inputs=[]
        for sent,heads in zip(fixed_size_sents,fixed_size_heads):

            assert(len(set(sent.run_id.values)) == 1)

            assert(len(sent)==len(heads))

            word_indices = sent.index.values
            sent_words = sent.word.values

            sent_str = " ".join(sent_words)



            pos_tags_encodings = [(TAGS.index(word_id_to_pos[word_ind]) \
                                   if word_id_to_pos[word_ind] in TAGS \
                                   else 0)
                                  for word_ind
                                  in word_indices]

            word_encodings = [self.emb.get_word_index(w)
                              for w in sent_words]

            head_encodings = [self.emb.get_word_index(w)
                              for w in heads]

            # Same pred word encodings for all words in the sentence
            pred_word = run_id_to_pred[int(sent.run_id.values[0])]
            pred_word_encodings = [self.emb.get_word_index(pred_word)
                                    for _ in sent_words]

            word_inputs.append([Sample(w) for w in word_encodings])
            pred_inputs.append([Sample(w) for w in pred_word_encodings])
            pos_inputs.append([Sample(pos) for pos in pos_tags_encodings])
            head_inputs.append([Sample(pos) for pos in head_encodings])

        # Pad / truncate to desired maximum length
        ret = defaultdict(lambda: [])

        for name, sequence in zip(["word_inputs", "predicate_inputs", "postags_inputs",'head_inputs'],
                                  [word_inputs, pred_inputs, pos_inputs,head_inputs]):
            
            for samples in self.pad_sequences(sequence,
                                         pad_func = lambda : Pad_sample(),
                                         maxlen = self.sent_maxlen):
                ret[name].append([sample.encode() for sample in samples])
        # for head in self.pad_head_input_num(head_inputs,maxlen=self.sent_maxlen):
        #     ret['head_inputs'].append(head)


        return {k: np.array(v) for k, v in ret.items()}

    def stack_attention_layers(self,nb_head, size_per_head, n):
        """
        Stack n Attentions
        """
        return lambda x: self.attention_stack(x, [lambda : Attention(nb_head,size_per_head)] * n )

    def attention_stack(self, x, layers):
        """
        Stack layers (FIFO) by applying recursively on the output,
        until returing the input as the base case for the recursion
        """
        if not layers:
            return x # Base case of the recursion is the just returning the input
        else:
            output = self.attention_stack(x, layers[1:])
            return self.addNorm(output,layers[0]()([output,output,output]))

    def addNorm(self,x,y):
        ffn = Dense(self.hidden_units,activation='relu')
        return ffn(BatchNormalization()(concatenate([y,x])))

    def predict_classes(self):
        """
        Predict to the number of classes
        Named arguments are passed to the keras function
        """
        return lambda x: self.stack(x,
                                    [lambda : TimeDistributed(Dense(self.num_of_classes(),
                                                                    activation = "softmax"))] +                           
                                    [lambda : TimeDistributed(Dense(self.hidden_units,
                                                                    activation='relu'))] * 1)

    def set_training_model(self):
        """
        继承RNNOIE的网络构建函数，添加transformer层
        """
        logging.debug("Setting TransformerTree model")
        word_inputs = Input(shape = (self.sent_maxlen,),dtype="int32",name = "word_inputs")
        predicate_inputs = Input(shape = (self.sent_maxlen,),dtype="int32",name = "predicate_inputs")
        postags_inputs = Input(shape = (self.sent_maxlen,),dtype="int32",name = "postags_inputs")
        head_inputs = Input(shape = (self.sent_maxlen,),dtype="int32",name = "head_inputs")

        #dropout
        dropout = lambda: Dropout(self.pred_dropout)

        # 嵌入层
        word_embedding_layer = self.embed_word()
        pos_embedding_layer = self.embed_pos()

        # 时序特征转换层
        bilstm_layers = self.stack_latent_layers(1)
        position_embedding = Position_Embedding()

        # Transformer层
        mulit_head_layers= self.stack_attention_layers(8,16,self.num_of_latent_layers)

        # 顺序转换层
        # layer_resret=LayerReset()

        # 全连接层
        predict_layer = self.predict_classes()

        # 构建
        # emb_output = concatenate([dropout()(word_embedding_layer(word_inputs)),dropout()(word_embedding_layer(predicate_inputs)),pos_embedding_layer(postags_inputs)])
        # bilstm_output = dropout()(bilstm_layers(emb_output))
        # transformer_output = mulit_head_layers(bilstm_output)
        # link_res=concatenate([dropout()(transformer_output),emb_output])
        # reset=layer_resret([link_res,head_inputs])
        # output=predict_layer(BatchNormalization()(concatenate([link_res,reset])))

        emb_output = concatenate([dropout()(word_embedding_layer(word_inputs)),dropout()(word_embedding_layer(head_inputs)),pos_embedding_layer(postags_inputs)])
        bilstm_output = dropout()(bilstm_layers(emb_output))
        transformer_output = mulit_head_layers(bilstm_output)
        link_res=concatenate([dropout()(transformer_output),emb_output])
        # reset=layer_resret([link_res,head_inputs])
        output=predict_layer(BatchNormalization()(link_res))


        # Build model
        self.model = Model(inputs = [word_inputs,predicate_inputs,postags_inputs,head_inputs], outputs = [output])

        # Loss
        self.model.compile(optimizer='adam',
                            loss='categorical_crossentropy',
                            metrics=['categorical_accuracy'])
        self.model.summary()

        # Save model json to file
        self.save_model_to_file(os.path.join(self.model_dir, "model.json"))

