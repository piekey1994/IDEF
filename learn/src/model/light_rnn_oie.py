from rnn_oie import RNNOIE_model
import logging
logging.basicConfig(level = logging.DEBUG)
import json
from tensorflow.keras.layers import Input, Dense, LSTM, Embedding, \
    TimeDistributed, concatenate, Bidirectional, Dropout, BatchNormalization
from tensorflow.keras.models import Model
import os

class LightRNNOIE(RNNOIE_model):

    def addNorm(self,x,y):
        ffn = Dense(self.hidden_units,activation='relu')
        return ffn(BatchNormalization()(concatenate([y,x])))

    def stack_latent_layers(self, n):
        """
        Stack n bidi LSTMs
        """
        return lambda x: self.norm_stack(x, [lambda : Bidirectional(LSTM(self.hidden_units,dropout=0.1, recurrent_dropout=0.1,
                                                                    return_sequences = True))] * n )    

    def norm_stack(self, x, layers):

        if not layers:
            return x # Base case of the recursion is the just returning the input
        else:
            output = self.norm_stack(x, layers[1:])
            # return self.addNorm(output,layers[0]()(output)) 
            return layers[0]()(output)                                                         

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
        logging.debug("Setting LightRNNOIE model")
        # 输入层三个
        word_inputs = Input(shape = (self.sent_maxlen,),dtype="int32",name = "word_inputs")
        predicate_inputs = Input(shape = (self.sent_maxlen,),dtype="int32",name = "predicate_inputs")
        postags_inputs = Input(shape = (self.sent_maxlen,),dtype="int32",name = "postags_inputs")

        # Dropout
        dropout = lambda: Dropout(self.pred_dropout)

        # 嵌入层
        word_embedding_layer = self.embed_word()
        pos_embedding_layer = self.embed_pos()

        # 时序特征转换层
        bilstm_layers = self.stack_latent_layers(self.num_of_latent_layers)

        # 全连接层
        predict_layer = self.predict_classes()

        # 构建
        emb_output = concatenate([dropout()(word_embedding_layer(word_inputs)),dropout()(word_embedding_layer(predicate_inputs)),pos_embedding_layer(postags_inputs)])
        bilstm_output = dropout()(bilstm_layers(emb_output))

        output=predict_layer(BatchNormalization()(concatenate([bilstm_output,emb_output])))


        # Build model
        self.model = Model(inputs = [word_inputs,predicate_inputs,postags_inputs], outputs = [output])

        # Loss
        self.model.compile(optimizer='adam',
                            loss='categorical_crossentropy',
                            metrics=['categorical_accuracy'])
        self.model.summary()

        # Save model json to file
        self.save_model_to_file(os.path.join(self.model_dir, "model.json"))
