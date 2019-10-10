from tensorflow.keras import backend as K
from tensorflow.keras.layers import Layer
from tensorflow.keras.layers import Dense, Activation, Multiply, Add, Lambda,LSTMCell,RNN,Input,Embedding
from tensorflow.keras.initializers import Constant
from tensorflow.keras.models import Model

class HighwayDense(Layer):

    activation = None
    transform_gate_bias = None

    def __init__(self, activation='relu', transform_gate_bias=-1, **kwargs):
        self.activation = activation
        self.transform_gate_bias = transform_gate_bias
        super(HighwayDense, self).__init__(**kwargs)

    def build(self, input_shape):
        # Create a trainable weight variable for this layer.
        self.dim=input_shape[-1]
        transform_gate_bias_initializer = Constant(self.transform_gate_bias)
        self.dense_1 = Dense(units=self.dim, bias_initializer=transform_gate_bias_initializer)
        self.dense_1.build(input_shape)
        self.dense_2 = Dense(units=self.dim)
        self.dense_2.build(input_shape)
        self._trainable_weights = self.dense_1.trainable_weights + self.dense_2.trainable_weights

        super(HighwayDense, self).build(input_shape)  # Be sure to call this at the end

    def call(self, x):
        transform_gate = self.dense_1(x)
        transform_gate = Activation("sigmoid")(transform_gate)
        carry_gate = Lambda(lambda x: 1.0 - x, output_shape=(self.dim,))(transform_gate)
        transformed_data = self.dense_2(x)
        transformed_data = Activation(self.activation)(transformed_data)
        transformed_gated = Multiply()([transform_gate, transformed_data])
        identity_gated = Multiply()([carry_gate, x])
        value = Add()([transformed_gated, identity_gated])
        return value

    def get_config(self):
        config = super().get_config()
        config['activation'] = self.activation
        config['transform_gate_bias'] = self.transform_gate_bias
        return config

class HighwayLSTMCell(LSTMCell):
    def __init__(self, couple_carry_transform_gates=True, carry_bias_init=1.0, **kwargs):
        self.couple_carry_transform_gates = couple_carry_transform_gates
        self.carry_bias_init = carry_bias_init
        super(HighwayLSTMCell, self).__init__(**kwargs)

    def build(self, input_shape):
        self.kernel = self.add_weight(shape=(input_shape[-1], self.units),
                                      initializer='uniform',
                                      name='kernel')
        self.recurrent_kernel = self.add_weight(
            shape=(self.units, self.units),
            initializer='uniform',
            name='recurrent_kernel')
        super(HighwayLSTMCell, self).build(input_shape)

    def call(self, inputs, states, training=None):
        h, hc = super(HighwayLSTMCell, self).call(inputs, states, training)
        return h, [h, hc[1]]
        
    

def HighwayLSTM(units,
    activation='tanh',
    recurrent_activation='hard_sigmoid',
    use_bias=True,
    kernel_initializer='glorot_uniform',
    recurrent_initializer='orthogonal',
    bias_initializer='zeros',
    unit_forget_bias=True,
    kernel_regularizer=None,
    recurrent_regularizer=None,
    bias_regularizer=None,
    activity_regularizer=None,
    kernel_constraint=None,
    recurrent_constraint=None,
    bias_constraint=None,
    dropout=0.0,
    recurrent_dropout=0.0,
    implementation=1,
    return_sequences=False,
    return_state=False,
    go_backwards=False,
    stateful=False,
    unroll=False):

    highwayLSTMCell=HighwayLSTMCell(
                units=units,
                activation=activation,
                recurrent_activation=recurrent_activation,
                use_bias=use_bias,
                kernel_initializer=kernel_initializer,
                recurrent_initializer=recurrent_initializer,
                bias_initializer=bias_initializer,
                unit_forget_bias=unit_forget_bias,
                kernel_regularizer=kernel_regularizer,
                recurrent_regularizer=recurrent_regularizer,
                bias_regularizer=bias_regularizer,
                kernel_constraint=kernel_constraint,
                recurrent_constraint=recurrent_constraint,
                bias_constraint=bias_constraint,
                dropout=dropout,
                recurrent_dropout=recurrent_dropout,
                implementation=implementation)

    return RNN(highwayLSTMCell,
        return_sequences=return_sequences,
        return_state=return_state,
        go_backwards=go_backwards,
        stateful=stateful,
        unroll=unroll)

if __name__=='__main__':
    highwayLSTM = HighwayLSTM(128,dropout=0.1, recurrent_dropout=0.1,return_sequences = False)
    X_seqs = Input(shape=(20,), dtype='int32')
    embeddings = Embedding(30, 128)
    highwayDense = HighwayDense()
    output=highwayDense(highwayLSTM(embeddings(X_seqs)))
    model = Model(inputs = [X_seqs], outputs = [output])
    model.compile(optimizer='adam',
                              loss='categorical_crossentropy',
                              metrics=['categorical_accuracy'])
    model.summary()