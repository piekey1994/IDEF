from rnn_oie import RNNOIE_model
import logging
logging.basicConfig(level = logging.DEBUG)
import json
from transformer import Attention,Position_Embedding,ICLayer
from tensorflow.keras.layers import Input, Dense, LSTM, Embedding, \
    TimeDistributed, concatenate, Bidirectional, Dropout, BatchNormalization
from tensorflow.keras.models import Model,model_from_json
import os
from elmo import createElmoLayer
from common.tokenizer_wrapper import TokenizerWrapper
from common.symbols import get_tags_from_lang
from sample import Sample,Pad_sample
from collections import defaultdict
import numpy as np
from tensorflow.keras import backend as K
import tensorflow as tf

class ElmoOIE(RNNOIE_model):

    def pad_sequences_elmo(self,sequences, maxlen = None):
        """
        Similar to keras.preprocessing.sequence.pad_sequence but using Sample as higher level
        abstraction.
        pad_func is a pad class generator.
        """
        ret = []

        # Determine the maxlen
        max_value = max(map(len, sequences))
        if maxlen is None:
            maxlen = max_value

        # Pad / truncate (done this way to deal with np.array)
        for sequence in sequences:
            cur_seq = list(sequence[:maxlen])
            cur_seq.extend([''] * (maxlen - len(sequence)))
            ret.append((cur_seq,[len(sequence)]))
        return ret

    def encode_inputs(self, sents):
        """
        将句子序列转换为神经网络的输入序列
        Parameters:
            sents - 已经通过get_sents_from_df分割好的句子序列
        Return:
            拥有5个key的字典"word_inputs", "predicate_inputs", "postags_inputs", "word_lens" , "pred_lens"
            其中每个value都是一个以句子为元素的list，以上三个句子的特征表示组合为一个训练数据
        """
        word_inputs = []
        word_lens=[]
        pred_inputs = []
        pred_lens=[]
        pos_inputs = []
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

            for index, word in zip(indices,
                                   tkw.parser(" ".join(words))):
                word_id_to_pos[index] = word.tag_

        fixed_size_sents = self.get_fixed_size(sents)


        for sent in fixed_size_sents:

            assert(len(set(sent.run_id.values)) == 1)

            word_indices = sent.index.values
            sent_words = sent.word.values

            sent_str = " ".join(sent_words)



            pos_tags_encodings = [(TAGS.index(word_id_to_pos[word_ind]) \
                                   if word_id_to_pos[word_ind] in TAGS \
                                   else 0)
                                  for word_ind
                                  in word_indices]

            word_encodings = [w for w in sent_words]

            # Same pred word encodings for all words in the sentence
            pred_word = run_id_to_pred[int(sent.run_id.values[0])]
            pred_word_encodings = [pred_word for _ in sent_words]

            word_inputs.append([w for w in word_encodings])
            pred_inputs.append([w for w in pred_word_encodings])
            pos_inputs.append([Sample(pos) for pos in pos_tags_encodings])

        # Pad / truncate to desired maximum length
        ret = defaultdict(lambda: [])

        for name, sequence in zip(["word_inputs", "predicate_inputs"],
                                  [word_inputs, pred_inputs]):
            
            for samples in self.pad_sequences_elmo(sequence,maxlen = self.sent_maxlen):
                ret[name].append([sample for sample in samples[0]])
                ret[name+'_len'].append(samples[1])

        for name, sequence in zip(["postags_inputs"],
                                  [pos_inputs]):
            
            for samples in self.pad_sequences(sequence,
                                         pad_func = lambda : Pad_sample(),
                                         maxlen = self.sent_maxlen):
                ret[name].append([sample.encode() for sample in samples])

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
        # return ffn(ICLayer(self.pred_dropout)(concatenate([y,x])))

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
    def model_from_json_with_custom(self):
        return model_from_json(open(os.path.join(self.model_dir,
                                                       "./model.json")).read(),
                                                       custom_objects={
                                                           'Attention': Attention,
                                                           'Position_Embedding':Position_Embedding,
                                                           'ICLayer':ICLayer
                                                       })

    def set_training_model(self):
        """
        继承RNNOIE的网络构建函数，添加transformer层
        """
        logging.debug("Setting TransformerELMO model")
        # 输入层三个
        word_inputs = Input(shape = (self.sent_maxlen,),dtype="string",name = "word_inputs")
        predicate_inputs = Input(shape = (self.sent_maxlen,),dtype="string",name = "predicate_inputs")
        word_inputs_len = Input(shape = (1,),dtype="int32",name = "word_inputs_len")
        predicate_inputs_len = Input(shape = (1,),dtype="int32",name = "predicate_inputs_len")
        postags_inputs = Input(shape = (self.sent_maxlen,),dtype="int32",name = "postags_inputs")

        #dropout
        dropout = lambda: Dropout(self.pred_dropout)
        bilstm_layers = self.stack_latent_layers(1)

        # 嵌入层
        word_embedding_layer = createElmoLayer('/data/liupq/transformer-oie/pretrained_word_embeddings/elmov2',self.trainable_emb,self.sent_maxlen)
        pos_embedding_layer = self.embed_pos()

        # Transformer层
        mulit_head_layers= self.stack_attention_layers(8,16,self.num_of_latent_layers)

        # 全连接层
        predict_layer = self.predict_classes()

        # 构建
        word_emb_input=word_embedding_layer([word_inputs,word_inputs_len])
        word_emb_input_dense=Dense(256,activation ='relu')(Dropout(0.1)(word_emb_input))
        pred_emb_input=word_embedding_layer([predicate_inputs,predicate_inputs_len])
        pred_emb_input_dense=Dense(256,activation ='relu')(Dropout(0.1)(pred_emb_input))
        pos_emb_input=pos_embedding_layer(postags_inputs)
        # emb_output = concatenate([Dense(self.hidden_units)(word_embedding_layer([word_inputs,word_inputs_len])),Dense(self.hidden_units)(word_embedding_layer([predicate_inputs,predicate_inputs_len])),pos_embedding_layer(postags_inputs)])
        emb_output = concatenate([dropout()(word_emb_input),dropout()(pred_emb_input),pos_emb_input])

        transformer_output = mulit_head_layers(dropout()(bilstm_layers(emb_output)))

        # emb_output = concatenate([dropout(word_embedding_layer(word_inputs)),dropout(word_embedding_layer(predicate_inputs)),pos_embedding_layer(postags_inputs)])
        # bilstm_output = dropout(bilstm_layers(dropout(word_embedding_layer(word_inputs))))
        # conect_output = concatenate([bilstm_output,dropout(word_embedding_layer(predicate_inputs)),pos_embedding_layer(postags_inputs)])
        # transformer_output = dropout(mulit_head_layers(conect_output))
        # transformer_output = dropout(self_attention([bilstm_output,bilstm_output,bilstm_output]))

        #pos_output = dropout(mulit_head_layers(position_embedding(emb_output)))
        # output=predict_layer(concatenate([bilstm_output,pos_output,emb_output]))
        link_output=concatenate([transformer_output,emb_output])
        output=predict_layer(dropout()(BatchNormalization()(link_output)))
        # output=predict_layer(bilstm_output)


        # Build model
        self.model = Model(inputs = [word_inputs,predicate_inputs,word_inputs_len,predicate_inputs_len,postags_inputs], outputs = [output])

        # Loss
        self.model.compile(optimizer='adam',
                            loss='categorical_crossentropy',
                            metrics=['categorical_accuracy'])
        self.model.summary()

        # Save model json to file
        self.save_model_to_file(os.path.join(self.model_dir, "model.json"))

        sess = K.get_session()
        init = tf.global_variables_initializer()
        sess.run(init)