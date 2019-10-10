import numpy as np
import pandas
from collections import defaultdict
from operator import itemgetter
import json
import os
from pprint import pformat
from glob import glob
import uuid
import pickle
import logging
logging.basicConfig(level = logging.DEBUG)

from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import Input, Dense, LSTM, Embedding, \
    TimeDistributed, concatenate, Bidirectional, Dropout
from tensorflow.keras.callbacks import Callback, ModelCheckpoint,TensorBoard,EarlyStopping,ReduceLROnPlateau
from tensorflow.keras.models import Model
from tensorflow.keras.models import model_from_json
from tensorflow import set_random_seed 
import sklearn
from sklearn import metrics as sk_metrics
from sklearn.preprocessing import LabelEncoder

from sample import Sample,Pad_sample
import sys
sys.path.append("..")
from common.load_pretrained_word_embeddings import EmbeddingConverter
from common.tokenizer_wrapper import TokenizerWrapper

from common.symbols import get_tags_from_lang


class RNNOIE_model:
    '''
    RNN-OIE模型源码
    '''
    def __init__(self):
        self.encoder = LabelEncoder()
        self.tkw=TokenizerWrapper('en')

    def model_from_json_with_custom(self):
        return model_from_json(open(os.path.join(self.model_dir,
                                                       "./model.json")).read(),
                                                       custom_objects={})

    def get_head_pred_word(self, full_sent):
        """
        从一个句子的conll中获取核心动词.
        """
        assert(len(set(full_sent.head_pred_id.values)) == 1) # Sanity check
        pred_ind = full_sent.head_pred_id.values[0]

        return full_sent.word.values[pred_ind] \
            if pred_ind != -1 \
               else full_sent.pred.values[0].split(" ")[0]

    def get_fixed_size(self, sents):
        """
        将句子按照最大长度切割
        """
        return [sent[s_ind : s_ind + self.sent_maxlen]
                for sent in sents
                for s_ind in range(0, len(sent), self.sent_maxlen)]

    def pad_sequences(self,sequences, pad_func, maxlen = None):
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
            cur_seq.extend([pad_func()] * (maxlen - len(sequence)))
            ret.append(cur_seq)
        return ret

    def classes_(self):
        """
        Return the classes which are classified by this model
        """
        try:
            return self.encoder.classes_
        except:
            return self.classes

    def num_of_classes(self):
        """
        Return the number of ouput classes
        """
        return len(self.classes_())

    def transform_labels(self, labels):
        """
        Encode a list of textual labels
        """
        # Fallback:
        # return self.encoder.transform(labels)
        classes  = list(self.classes_())
        return [classes.index(label) for label in labels]

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
        tkw = self.tkw
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

            word_encodings = [self.emb.get_word_index(w)
                              for w in sent_words]

            # Same pred word encodings for all words in the sentence
            pred_word = run_id_to_pred[int(sent.run_id.values[0])]
            pred_word_encodings = [self.emb.get_word_index(pred_word)
                                    for _ in sent_words]

            word_inputs.append([Sample(w) for w in word_encodings])
            pred_inputs.append([Sample(w) for w in pred_word_encodings])
            pos_inputs.append([Sample(pos) for pos in pos_tags_encodings])

        # Pad / truncate to desired maximum length
        ret = defaultdict(lambda: [])

        for name, sequence in zip(["word_inputs", "predicate_inputs", "postags_inputs"],
                                  [word_inputs, pred_inputs, pos_inputs]):
            
            for samples in self.pad_sequences(sequence,
                                         pad_func = lambda : Pad_sample(),
                                         maxlen = self.sent_maxlen):
                ret[name].append([sample.encode() for sample in samples])

        return {k: np.array(v) for k, v in ret.items()}

    def encode_outputs(self, sents):
        """
        将句子序列转换为神经网络的输出标签序列
        Parameters:
            sents - 已经通过get_sents_from_df分割好的句子序列
        Return:
            三维的ndarray，句子数量，每个句子的最大长度，每个标签的类别总数
        """
        output_encodings = []
        sents = self.get_fixed_size(sents)
        # Encode outputs
        for sent in sents:
            output_encodings.append(list(to_categorical(list(self.transform_labels(sent.label.values)),
                                                                 num_classes = self.num_of_classes())))

        # Pad / truncate to maximum length
        return np.ndarray(shape = (len(sents),
                                  self.sent_maxlen,
                                  self.num_of_classes()),
                          buffer = np.array(self.pad_sequences(output_encodings,
                                                          lambda : \
                                                            np.zeros(self.num_of_classes()),
                                                          maxlen = self.sent_maxlen)))

    def get_sents_from_df(self, df):
        """
        将pd读取的数据按照run_id整理成句子序列
        """
        return [df[df.run_id == run_id]
                for run_id
                in sorted(set(df.run_id.values))]

    def load_dataset(self, fn):
        """
        读取数据集
        Parameters:
            fn - 数据集路径
        Return:
            模型的输入（词，动词，词性）和输出（标签序列）
        """

        df = pandas.read_csv(fn,
                             sep = '\t',
                             header = 0,
                             keep_default_na = False)

        # 将标签转换为数字
        if self.classes_() is None:
            self.encoder.fit(df.label.values)

        encodeFileName=fn+str(uuid.uuid3(uuid.NAMESPACE_URL,self.emb_filename+str(self.sent_maxlen)+self.__class__.__name__))
        if os.path.exists(encodeFileName):
            with open(encodeFileName,'rb') as encodeFile:
                encodeDict=pickle.load(encodeFile)
                x=encodeDict['x']
                y=encodeDict['y']
        else:
            # 将数据切分成句子序列[[w1,w2...],[..],...]
            sents = self.get_sents_from_df(df)
            x=self.encode_inputs(sents)
            y=self.encode_outputs(sents)
            with open(encodeFileName,'wb') as encodeFile:
                encodeDict={
                    'x':x,
                    'y':y
                }
                pickle.dump(encodeDict,encodeFile)
        return (x,y)

    def create_training_model(self,model_dir='./models/',sent_maxlen=20,emb_filename=None,
                            batch_size=5,seed=27,hidden_units = pow(2, 7),
                            emb_dropout = 0.1, num_of_latent_layers = 2,
                            epochs = 10, pred_dropout = 0.1, trainable_emb = True,
                            classes = None, pos_tag_embedding_size = 5,model_name=None):
        '''
        创建一个用于训练的RNNOIE模型
        Parameters:
            model_dir - 模型保存路径
            sent_maxlen - 每个输入序列的最大长度，超过该长度会被裁切，少于该长度会被pad
            emb_filename - 预训练词向量目录
            batch_size - batch大小
            seed - 初始化随机数种子
            sep - 分隔符
            hidden_units - lstm中隐层单元数量
            emb_dropout - 词嵌入层的丢弃值
            num_of_latent_layers - lstm层数
            epochs - 整体数据迭代轮数
            pred_dropout - pemb层的丢弃值
            trainable_emb - 是否启用预训练词向量
            classes - None表示不预先指定生成标签的类别总数，由训练集直接分析决定
            pos_tag_embedding_size - 词性嵌入长度
            model_name - 模型名称
        Return:
            RNNOIE_model类对象
        '''
        self.model_dir = model_dir
        self.sent_maxlen = sent_maxlen
        self.batch_size = batch_size
        self.seed = seed
        self.hidden_units = hidden_units
        self.emb_filename = emb_filename
        self.emb = EmbeddingConverter(emb_filename)
        self.embedding_size = self.emb.dim
        self.trainable_emb = trainable_emb
        self.emb_dropout = emb_dropout
        self.num_of_latent_layers = num_of_latent_layers
        self.epochs = epochs
        self.pred_dropout = pred_dropout
        self.classes = classes
        self.pos_tag_embedding_size = pos_tag_embedding_size
        self.model_name = model_name

        np.random.seed(self.seed)
        set_random_seed(self.seed)

    def get_callbacks(self, X):
        """
        Sets these callbacks as a class member.
        X is the encoded dataset used to print a sample of the output.
        Callbacks created:
        1. Sample output each epoch
        2. Save best performing model each epoch
        """

        class LossHistory(Callback):
            def __init__(self,save_dir):
                self.save_dir=save_dir

            def on_train_begin(self, logs={}):
                self.train_acc=[]
                self.val_acc=[]

            def on_epoch_end(self, epoch, logs={}):
                self.train_acc.append(float(logs.get('categorical_accuracy')))
                _val_acc=float(logs.get('val_categorical_accuracy'))
                self.val_acc.append(_val_acc)
                print('')
                print('max val F1:%f' % (max(self.val_acc)))

            def on_train_end(self,logs={}):
                with open(self.save_dir,'w',encoding='utf-8') as resultFile:
                    for i,(ta,va) in enumerate(zip(self.train_acc,self.val_acc)):
                        resultFile.write('epoch:%d train_acc:%f val_acc:%f\n' %(i+1,ta,va))

        class Metrics(Callback):
            def __init__(self, filepath,rnnmodel):
                self.file_path = filepath
                self.rnnmodel = rnnmodel

            def on_train_begin(self, logs={}):
                self.val_f1s = []
                self.best_val_f1 = 0

            def on_epoch_end(self, epoch, logs={}):
                y = self.model.predict(self.validation_data[:3])
                Y = self.validation_data[3]
                # Get most probable predictions and flatten
                val_targ = RNNOIE_model.consolidate_labels(self.rnnmodel.transform_output_probs(Y).flatten())
                val_predict = RNNOIE_model.consolidate_labels(self.rnnmodel.transform_output_probs(y).flatten())
                _val_f1 = sk_metrics.f1_score(val_targ, val_predict, average='micro')
                self.val_f1s.append(_val_f1)
                print('')
                print('F1:%f max F1:%f' % (_val_f1,max(self.val_f1s)))
                if _val_f1 > self.best_val_f1:
                    self.model.save_weights(self.file_path, overwrite=True)
                    self.best_val_f1 = _val_f1
                    print("best f1: {}".format(self.best_val_f1))
                else:
                    print("val f1: {}, but not the best f1".format(_val_f1))
                print('')
                return

        # metrics = Metrics(os.path.join(self.model_dir,"weights.hdf5"),self)

        checkpoint = ModelCheckpoint(os.path.join(self.model_dir,"weights.hdf5"),
                                     verbose = 1,
                                     monitor = 'val_categorical_accuracy',
                                     save_best_only = False)  
        tensorboard = TensorBoard(log_dir=self.model_dir, histogram_freq=0)
        metric_loss = LossHistory(os.path.join(self.model_dir,"acc.txt"))
        # stop = EarlyStopping()
        # learningrate = ReduceLROnPlateau()

        return [tensorboard,metric_loss,checkpoint]

    def create_sample(self, sent, head_pred_id):
        """
        Return a dataframe which could be given to encode_inputs
        """
        return pandas.DataFrame({"word": sent,
                                 "run_id": [-1] * len(sent), # Mock running id
                                 "head_pred_id": head_pred_id})

    def predict_sentences(self, sents):
        """
        Return a predicted label for each word in an arbitrary length sentence
        sent - a list of string tokens
        """
        tkw=self.tkw
        sents_attr=[]
        sent_samples={
            "word_inputs":[],
            "predicate_inputs":[],
            "postags_inputs":[]
        }
        print('prepare data')
        for sid,sent in enumerate(sents):
            if sid % (int(np.ceil(len(sents)/100))) == 0:
                print(sid / len(sents))
            sent_str = " ".join(sent)
            preds = [(word.i, str(word))
                    for word
                    in tkw.parser(sent_str)
                    if word.tag_.startswith("V")]
            num_of_samples = int(np.ceil(float(len(sent)) / self.sent_maxlen) * self.sent_maxlen)
            pred_list=[]
            for ind, pred in preds:
                cur_sample=self.encode_inputs([self.create_sample(sent, ind)])
                for name in ["word_inputs", "predicate_inputs", "postags_inputs"]:
                    sent_samples[name].append(cur_sample[name])
                pred_list.append((ind, pred))
            sents_attr.append((num_of_samples,pred_list,len(sent)))
        for key in sent_samples:
            sent_samples[key]=np.concatenate(sent_samples[key],axis=0)
        print('predict data')
        X = sent_samples
        Y=self.model.predict(X)
        # print(Y[0])
        # print(Y[2])
        res=[]
        p=0
        for attr in sents_attr:
            num_of_samples,pred_list,sent_len=attr
            sample_len=num_of_samples//self.sent_maxlen
            ret=[]
            for pid,(ind, pred) in enumerate(pred_list):
                ret.append(((ind, pred),
                            [(self.consolidate_label(label), float(prob))
                            for (label, prob) in
                            self.transform_output_probs(Y[p+pid*sample_len:p+(pid+1)*sample_len],     
                                                        get_prob = True).reshape(num_of_samples,
                                                                                2)[:sent_len]]))
            res.append(ret)
            p+=len(pred_list)*sample_len
        return res
    
    def predict_sentence(self, sent):
        """
        Return a predicted label for each word in an arbitrary length sentence
        sent - a list of string tokens
        """
        ret = []
        sent_str = " ".join(sent)
        tkw = self.tkw

        # Extract predicates by looking at verbal POS

        preds = [(word.i, str(word))
                 for word
                 in tkw.parser(sent_str)
                 if word.tag_.startswith("V")]

        # Calculate num of samples (round up to the nearst multiple of sent_maxlen)计算样本个数(取sent_maxlen的最接近整数倍)
        num_of_samples = int(np.ceil(float(len(sent)) / self.sent_maxlen) * self.sent_maxlen)

        # Run RNN for each predicate on this sentence
        for ind, pred in preds:
            cur_sample = self.create_sample(sent, ind)
            X = self.encode_inputs([cur_sample])
            ret.append(((ind, pred),
                        [(self.consolidate_label(label), float(prob))
                         for (label, prob) in
                         self.transform_output_probs(self.model.predict(X),           # "flatten" and truncate
                                                     get_prob = True).reshape(num_of_samples,
                                                                              2)[:len(sent)]]))
        return ret

    def train(self,train_fn,dev_fn,test_fn):
        """
        读取三个数据集，进行训练，并输出测试结果
        """

        X_train, Y_train = self.load_dataset(train_fn)
        X_dev, Y_dev = self.load_dataset(dev_fn)
        logging.debug("Classes: {}".format((self.num_of_classes(), self.classes_())))

        # 构建神经网络
        self.set_training_model()

        # 训练模型，并通过callback输出中间结果和保存中间模型
        logging.debug("Training model on {}".format(train_fn))
        self.model.fit(X_train, Y_train,
                       batch_size = self.batch_size,
                       epochs = self.epochs,
                       validation_data = (X_dev, Y_dev),
                       callbacks = self.get_callbacks(X_train))
        #测试
        print('last model test:')
        self.test(test_fn,eval_metrics = [("F1 (micro)",
                                            lambda Y, y: sk_metrics.f1_score(Y, y,
                                                                        average = 'micro'))
                                        ])
        print('best model test:')         
        self.model.load_weights(os.path.join(self.model_dir,"weights.hdf5"))
        _Y, _y, test_result = self.test(test_fn,
                            eval_metrics = [("F1 (micro)",
                                            lambda Y, y: sk_metrics.f1_score(Y, y,
                                                                        average = 'micro')),
                                            # ("F1 (macro)",
                                            # lambda Y, y: sk_metrics.f1_score(Y, y,
                                            #                             average = 'macro')),
                                        ])
        # with open(os.path.join(self.model_dir,"test.txt"),'w',encoding='utf-8') as textFile:
        #     for (metric_name, metric_val) in test_result:
        #         textFile.write("{}: {:.4f}\n".format(metric_name,
        #                                         metric_val))

    def embed_word(self):
        """
        Embed word sequences using self's embedding class
        """
        return self.emb.get_keras_embedding(trainable = self.trainable_emb,
                                            input_length = self.sent_maxlen)

    def embed_pos(self):
        """
        Embed Part of Speech using this instance params
        """
        return Embedding(output_dim = self.pos_tag_embedding_size,
                         input_dim = len(get_tags_from_lang('en')),
                         input_length = self.sent_maxlen)

    def stack_latent_layers(self, n):
        """
        Stack n bidi LSTMs
        """
        return lambda x: self.stack(x, [lambda : Bidirectional(LSTM(self.hidden_units,dropout=0.1, recurrent_dropout=0.1,
                                                                    return_sequences = True))] * n )

    def stack(self, x, layers):
        """
        Stack layers (FIFO) by applying recursively on the output,
        until returing the input as the base case for the recursion
        """
        if not layers:
            return x # Base case of the recursion is the just returning the input
        else:
            return layers[0]()(self.stack(x, layers[1:]))

    def predict_classes(self):
        """
        Predict to the number of classes
        Named arguments are passed to the keras function
        """
        return lambda x: self.stack(x,
                                    [lambda : TimeDistributed(Dense(self.num_of_classes(),
                                                                    activation = "softmax"))] +
                                    [lambda : TimeDistributed(Dense(self.hidden_units,
                                                                    activation='relu'))] * 3)
    
    def to_json(self):
        """
        Encode a json of the parameters needed to reload this model
        """
        return {
            "sent_maxlen": self.sent_maxlen,
            "batch_size": self.batch_size,
            "seed": self.seed,
            "classes": list(self.classes_()),
            "hidden_units": self.hidden_units,
            "trainable_emb": self.trainable_emb,
            "emb_dropout": self.emb_dropout,
            "num_of_latent_layers": self.num_of_latent_layers,
            "epochs": self.epochs,
            "pred_dropout": self.pred_dropout,
            "emb_filename": self.emb_filename,
            "pos_tag_embedding_size": self.pos_tag_embedding_size,
            "model_name":self.model_name
        }

    def save_model_to_file(self, fn):
        """
        Saves this model to file, also encodes class inits in the model's json
        """
        js = json.loads(self.model.to_json())

        # Add this model's params
        js["rnn"] = self.to_json()
        with open(fn, 'w') as fout:
            json.dump(js, fout)
        with open(os.path.join(self.model_dir, "model.summary"),'w') as summaryFile:
            self.model.summary(print_fn = lambda s: print(s, file=summaryFile))
                
        

    def set_training_model(self):
        """
        建立一个Keras模型来预测OIE是这个类的一员
        """
        logging.debug("Setting rnnoie model")
        # Build model

        ## Embedding Layer
        word_embedding_layer = self.embed_word()
        pos_embedding_layer = self.embed_pos()

        ## Deep layers
        latent_layers = self.stack_latent_layers(self.num_of_latent_layers)

        ## Dropout
        dropout = Dropout(self.pred_dropout)

        ## Prediction
        predict_layer = self.predict_classes()

        ## Prepare input features, and indicate how to embed them
        inputs_and_embeddings = [(Input(shape = (self.sent_maxlen,),
                                        dtype="int32",
                                        name = "word_inputs"),
                                  word_embedding_layer),
                                 (Input(shape = (self.sent_maxlen,),
                                        dtype="int32",
                                        name = "predicate_inputs"),
                                  word_embedding_layer),
                                 (Input(shape = (self.sent_maxlen,),
                                        dtype="int32",
                                        name = "postags_inputs"),
                                  pos_embedding_layer),
        ]

        ## Concat all inputs and run on deep network
        output = predict_layer(dropout(latent_layers(concatenate([dropout(embed(inp))
                                                            for inp, embed in inputs_and_embeddings]))))

        # Build model
        self.model = Model(inputs = list(map(itemgetter(0), inputs_and_embeddings)),
                           outputs = [output])

        # Loss
        self.model.compile(optimizer='adam',
                           loss='categorical_crossentropy',
                           metrics=['categorical_accuracy'])
        self.model.summary()

        # Save model json to file
        self.save_model_to_file(os.path.join(self.model_dir, "model.json"))

    @staticmethod
    def consolidate_labels(labels):
        """
        Return a consolidated list of labels, e.g., O-A1 -> O, A1-I -> A
        """
        return list(map(RNNOIE_model.consolidate_label , labels))

    @staticmethod
    def consolidate_label(label):
        """
        Return a consolidated label, e.g., O-A1 -> O, A1-I -> A
        """
        return label.split("-")[0] if label.startswith("O") else label

    def sample_labels(self, y, num_of_sents = 5, num_of_samples = 10,
                      num_of_classes = 3, start_index = 5, get_prob = True):
        """
        Get a sense of how labels in y look like
        """
        classes = self.classes_()
        ret = []
        am = lambda myList: [i[0] for i in sorted(enumerate(myList), key=lambda x:x[1], reverse= True)]

        for sent in y[:num_of_sents]:
            cur = []
            for word in sent[start_index: start_index + num_of_samples]:
                sorted_prob = am(word)
                cur.append([(classes[ind], word[ind]) if get_prob else classes[ind]
                            for ind in sorted_prob[:num_of_classes]])
            ret.append(cur)
        return ret

    def transform_output_probs(self, y, get_prob = False):
        """
        Given a list of probabilities over labels, get the textual representation of the
        most probable assignment
        """
        return np.array(self.sample_labels(y,
                                  num_of_sents = len(y), # all sentences
                                  num_of_samples = max(map(len, y)), # all words
                                  num_of_classes = 1, # Only top probability
                                  start_index = 0, # all sentences
                                  get_prob = get_prob, # Indicate whether to get only labels
        ))

    def test(self,test_fn, eval_metrics):
        """
        Evaluate this model on a test file
        eval metrics is a list composed of:
        (name, f: (y_true, y_pred) -> float (some performance metric))
        Prints and returns the metrics name and numbers
        """
        # Load gold and predict
        X, Y = self.load_dataset(test_fn)
        y = self.model.predict(X)

        # Get most probable predictions and flatten
        Y = RNNOIE_model.consolidate_labels(self.transform_output_probs(Y).flatten())
        y = RNNOIE_model.consolidate_labels(self.transform_output_probs(y).flatten())

        # Run evaluation metrics and report
        # TODO: is it possible to compare without the padding?
        ret = []
        for (metric_name, metric_func) in eval_metrics:
            ret.append((metric_name, metric_func(Y, y)))
            # logging.debug("calculating {}".format(ret[-1]))

        for (metric_name, metric_val) in ret:
            logging.info("{}: {:.4f}".format(metric_name,
                                             metric_val))
        return Y, y, ret

    def set_model_from_file(self):
        """
        Receives an instance of RNN and returns a model from the self.model_dir 接收RNN的一个实例，并从自身返回一个模型
        path which should contain a file named: model.json,应该包含名为model.json的文件的路径
        and a single file with the hdf5 extension.和一个扩展名为hdf5的文件
        Note: Use this function for a pretrained model, running model training
        on the loaded model will override the files in the model_dir
        """
        weights_fn = glob(os.path.join(self.model_dir, "*.hdf5"))
        assert len(weights_fn) == 1, "More/Less than one weights file in {}: {}".format(self.model_dir,
                                                                                        weights_fn)
        weights_fn = weights_fn[0]
        logging.debug("Weights file: {}".format(weights_fn))
        self.model = self.model_from_json_with_custom()
        self.model.load_weights(weights_fn)
        self.model.compile(optimizer="adam",
                           loss='categorical_crossentropy',
                           metrics = ["accuracy"])

    #用于断点续练或单独测试
    def load_pretrained_model(self,model_dir):
        """ Static trained model loader function """
        rnn_params = json.load(open(os.path.join(model_dir,
                                                "./model.json")))["rnn"]

        logging.info("Loading model from: {}".format(model_dir))
        self.create_training_model(model_dir = model_dir,
                        **rnn_params)
        #从目录中读取神经网络参数
        self.set_model_from_file()

    def train_again(self,train_fn,dev_fn,test_fn,missing_epochs):
        X_train, Y_train = self.load_dataset(train_fn)
        X_dev, Y_dev = self.load_dataset(dev_fn)
        logging.debug("Classes: {}".format((self.num_of_classes(), self.classes_())))

        # 训练模型，并通过callback输出中间结果和保存中间模型
        logging.debug("Continue training model on {}".format(train_fn))
        self.model.fit(X_train, Y_train,
                       batch_size = self.batch_size,
                       epochs = self.missing_epochs,
                       validation_data = (X_dev, Y_dev),
                       callbacks = self.get_callbacks(X_train))

        #测试
        _Y, _y, test_result = self.test(test_fn,
                            eval_metrics = [("F1 (micro)",
                                            lambda Y, y: sk_metrics.f1_score(Y, y,
                                                                        average = 'micro')),
                                            ("F1 (macro)",
                                            lambda Y, y: sk_metrics.f1_score(Y, y,
                                                                        average = 'macro')),
                                                                        
                                            # ("AUC (micro)",
                                            # lambda Y, y: sk_metrics.roc_auc_score(Y, y))
                                        ])
                                        
        with open(os.path.join(self.model_dir,"test.txt"),'w',encoding='utf-8') as textFile:
            for (metric_name, metric_val) in test_result:
                textFile.write("{}: {:.4f}\n".format(metric_name,
                                                metric_val))
