from rnn_oie import RNNOIE_model
from transformer_lstm_oie import TransformerLSTMOIE
from light_rnn_oie import LightRNNOIE
from transformer_tree import TransformerTree
from elmo_oie import ElmoOIE

def get_model(model_name):
    if model_name is None or model_name == 'transformer':
        rnn=TransformerLSTMOIE()
    elif model_name == 'rnnoie':
        rnn=RNNOIE_model()
    elif model_name == 'light':
        rnn=LightRNNOIE()
    elif model_name == 'tree':
        rnn=TransformerTree()
    elif model_name == 'elmo':
        rnn=ElmoOIE()
    return rnn