import tensorflow as tf
import tensorflow_hub as hub
from tensorflow.keras import backend as K
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Layer,Input,Lambda,Reshape,Dense,TimeDistributed
import numpy as np
from tensorflow.keras.utils import to_categorical

def createElmoLayer(emb_path,trainable,max_length):
    def ElmoLayer(x):
        elmo = hub.Module(emb_path, trainable=trainable)
        result =  elmo(
                    inputs={
                        "tokens": x[0],
                        "sequence_len": K.flatten(x[1])
                    },
                    signature="tokens",
                    as_dict=True)["elmo"]
        return Reshape((max_length,1024))(result)
    
    return Lambda(ElmoLayer,output_shape=(max_length, 1024))
    

if __name__=='__main__':
    mask_str=''
    max_length=6
    class_num=5

    elmoEmb = createElmoLayer('../../pretrained_word_embeddings/elmov2',True,max_length)
    input_text = Input(shape=(max_length,), dtype="string",name = "input_text")
    input_len = Input(shape=(1,), dtype="int32",name = "input_len")
    output = elmoEmb([input_text,input_len])
    output = TimeDistributed(Dense(class_num, activation='sigmoid'))(output)
    model = Model(inputs=[input_text,input_len], outputs=output)
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['categorical_accuracy'])
    model.summary()

    texts=['the cat is on the big mat','dogs are in the fog']
    y=to_categorical(np.random.randint(class_num, size=(2, max_length)),class_num)
    test_x=[]
    test_len=[]
    for text in texts:
        words=text.split(' ')
        word_nums=len(words)
        if word_nums>=max_length:
            words=words[:max_length]
            word_nums=max_length
        else:
            for i in range(max_length-word_nums):
                words.append(mask_str)
        test_x.append(words)
        test_len.append([word_nums])

    print(test_x)
    print(test_len)

    x={
        'input_text':np.array(test_x),
        'input_len':np.array(test_len)
    }

    sess = K.get_session()
    init = tf.global_variables_initializer()
    sess.run(init)

    model.fit(x, y)


    