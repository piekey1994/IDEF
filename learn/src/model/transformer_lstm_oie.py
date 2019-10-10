from rnn_oie import RNNOIE_model
import logging
logging.basicConfig(level = logging.DEBUG)
import json
from transformer import Attention,Position_Embedding,ICLayer
from tensorflow.keras.layers import Input, Dense, LSTM, Embedding, \
    TimeDistributed, concatenate, Bidirectional, Dropout, BatchNormalization
from tensorflow.keras.models import Model,model_from_json
import os


class TransformerLSTMOIE(RNNOIE_model):


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
        logging.debug("Setting TransformerLSTMOIE model")
        # 输入层三个
        word_inputs = Input(shape = (self.sent_maxlen,),dtype="int32",name = "word_inputs")
        predicate_inputs = Input(shape = (self.sent_maxlen,),dtype="int32",name = "predicate_inputs")
        postags_inputs = Input(shape = (self.sent_maxlen,),dtype="int32",name = "postags_inputs")

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
        self_attention = Attention(8,16)


        # 全连接层
        predict_layer = self.predict_classes()

        # 构建
        emb_output = concatenate([dropout()(word_embedding_layer(word_inputs)),dropout()(word_embedding_layer(predicate_inputs)),pos_embedding_layer(postags_inputs)])
        # emb_output = concatenate([dropout()(word_embedding_layer(word_inputs)),pos_embedding_layer(postags_inputs)])
        bilstm_output = dropout()(bilstm_layers(emb_output))
        # transformer_output = self_attention([bilstm_output,bilstm_output,bilstm_output])
        transformer_output = mulit_head_layers(bilstm_output)

        # emb_output = concatenate([dropout(word_embedding_layer(word_inputs)),dropout(word_embedding_layer(predicate_inputs)),pos_embedding_layer(postags_inputs)])
        # bilstm_output = dropout(bilstm_layers(dropout(word_embedding_layer(word_inputs))))
        # conect_output = concatenate([bilstm_output,dropout(word_embedding_layer(predicate_inputs)),pos_embedding_layer(postags_inputs)])
        # transformer_output = dropout(mulit_head_layers(conect_output))
        # transformer_output = dropout(self_attention([bilstm_output,bilstm_output,bilstm_output]))

        #pos_output = dropout(mulit_head_layers(position_embedding(emb_output)))
        # output=predict_layer(concatenate([bilstm_output,pos_output,emb_output]))
        output=predict_layer(BatchNormalization()(concatenate([dropout()(transformer_output),emb_output])))
        # output=predict_layer(bilstm_output)


        # Build model
        self.model = Model(inputs = [word_inputs,predicate_inputs,postags_inputs], outputs = [output])

        # Loss
        self.model.compile(optimizer='adam',
                            loss='categorical_crossentropy',
                            metrics=['categorical_accuracy'])
        self.model.summary()

        # Save model json to file
        self.save_model_to_file(os.path.join(self.model_dir, "model.json"))
