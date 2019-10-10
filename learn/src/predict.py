""" Usage:
predict [--model=MODEL_DIR] --in=INPUT_FILE --out=OUTPUT_FILE [--tokenize] [--conll] [--model_name=MODEL_NAME]

Run a trined OIE model on raw sentences.

MODEL_DIR - Pretrained RNN model folder (containing model.json and pretrained weights).
INPUT FILE - File where each row is a tokenized sentence to be parsed with OIE.
OUTPUT_FILE - File where the OIE tuples will be output.
tokenize - indicates that the input sentences are NOT tokenized.
conll - Print a CoNLL represenation with probabilities

Format of OUTPUT_FILE:
    Sent, prob, pred, arg1, arg2, ...
"""

from docopt import docopt
import logging
import re
import numpy as np
import os
import time
from collections import defaultdict
import sys
import json
from functools import reduce
sys.path.append("model")
sys.path.append("common")
from model.model_list import get_model 
from common.tokenizer_wrapper import TokenizerWrapper

logging.basicConfig(level = logging.DEBUG)


def load_pretrained_model(model_dir):
    """ Static trained model loader function """
    rnn_params = json.load(open(os.path.join(model_dir,
                                            "./model.json")))["rnn"]
    model_name=rnn_params['model_name']
    logging.info("Loading model from: {}".format(model_dir))
    logging.info("model name: {}".format(model_name))
    
    rnn=get_model(model_name)
    rnn.create_training_model(model_dir = model_dir, **rnn_params)
    #从目录中读取神经网络参数
    rnn.set_model_from_file()
    return rnn

class Trained_oie:
    """
    Compose OIE extractions given a pretrained RNN OIE model predicting classes per word
    """
    def __init__(self, model, tokenize):
        """
        model - pretrained supervised model
        tokenize - instance-wide indication whether all of the functions should
                   tokenize their input
        """
        self.model = model
        self.tokenize = tokenize
        if tokenize:
            tkw = TokenizerWrapper('en')

    def split_words(self, sent):
        """
        Apply tokenization if needed, else just split by space
        sent - string
        """
        return tkw.word_tokenize(sent) if self.tokenize\
            else re.split(r' +', sent) # Allow arbitrary number of spaces


    def get_extractions(self, sent):
        """
        Returns a list of OIE extractions for a given sentence
        sent - a list of tokens
        """
        ret = []

        for ((pred_ind, pred_word), labels) in self.model.predict_sentence(sent):
            cur_args = []
            cur_arg = []
            probs = []

            # collect args
            for (label, prob), word in zip(labels, sent):
                if label.startswith("A"):
                    cur_arg.append(word)
                    probs.append(prob)

                elif cur_arg:
                    cur_args.append(cur_arg)
                    cur_arg = []

            # Create extraction
            if cur_args:
                ret.append(Extraction(sent,
                                      pred_word,
                                      cur_args,
                                      probs
                                  ))
        return ret

    def conll_with_prob(self, sent):
        """
        Returns a conll representation of sentence
        Format:
        word index, word, pred_index, label, probability
        """
        logging.debug("Parsing: {}".format(sent))
        sent = self.split_words(sent)
        ret = ""
        for ((pred_ind, pred_word), labels) in self.model.predict_sentence(sent):
            for (word_ind, ((label, prob), word)) in enumerate(zip(labels, sent)):
                ret+= "\t".join(map(str,
                                         [word_ind, word, pred_ind, label, prob]
                                     )) + '\n'
            ret += '\n'
        return ret

    def parse_sent(self, sent):
        """
        Returns a list of extractions for the given sentence
        sent - a tokenized sentence
        tokenize - boolean indicating whether the sentences should be tokenized first
        """
        logging.debug("Parsing: {}".format(sent))
        return self.get_extractions(self.split_words(sent))

    def parse_sents(self, sents):
        """
        Returns a list of extractions per sent in sents.
        sents - list of tokenized sentences
        tokenize - boolean indicating whether the sentences should be tokenized first
        """
        return [self.parse_sent(sent)
                for sent in sents]

class Extraction:
    """
    Store and print an OIE extraction
    """
    def __init__(self, sent, pred, args, probs,calc_prob = lambda probs:np.mean(probs)):
                #  calc_prob = lambda probs: 1.0 / (reduce(lambda x, y: x * y, probs) + 0.001)):
        """
        sent - Tokenized sentence - list of strings
        pred - Predicate word
        args - List of arguments (each a string)
        probs - list of float in [0,1] indicating the probability
               of each of the items in the extraction
        calc_prob - function which takes a list of probabilities for each of the
                    items and computes a single probability for the joint occurence of this extraction.
        """
        self.sent = sent
        self.calc_prob = calc_prob
        self.probs = probs
        self.prob = self.calc_prob(self.probs)
        self.pred = pred
        self.args = args
        logging.debug(self)

    def __str__(self):
        """
        Format (tab separated):
        Sent, prob, pred, arg1, arg2, ...
        """
        return '\t'.join(map(str,
                             [' '.join(self.sent),
                              self.prob,
                              self.pred,
                              '\t'.join([' '.join(arg)
                                         for arg in self.args])]))
class Mock_model:
    """
    Load a conll file annotated with labels And probabilities
    and present an external interface of a trained rnn model (through predict_sentence).
    This can be used to alliveate running the trained model.
    """
    def __init__(self, conll_file):
        """
        conll_file - file from which to load the annotations
        """
        self.conll_file = conll_file
        logging.debug("loading file {}".format(self.conll_file))
        self.dic, self.sents = self.load_annots(self.conll_file)
        logging.debug("Done loading file")

    def load_annots(self, conll_file):
        """
        Updates internal state according to file
        for ((pred_ind, pred_word), labels) in self.model.predict_sentence(sent):
                    for (label, prob), word in zip(labels, sent):
        """
        cur_ex = []
        cur_sent = []
        pred_word = ''
        ret = defaultdict(lambda: {})
        sents = []

        # Iterate over lines and populate return dictionary
        for line_ind, line in enumerate(open(conll_file)):
            if not (line_ind % pow(10,5)):
                logging.debug(line_ind)
            line = line.strip()
            if not line:
                if cur_ex:
                    assert(pred_word != '') # sanity check
                    cur_sent = " ".join(cur_sent)

                    # This is because of the dups bug --
                    # doesn't suppose to happen any more
                    ret[cur_sent][pred_word] = (((pred_ind, pred_word), cur_ex),)

                    sents.append(cur_sent)
                    cur_ex = []
                    pred_ind = -1
                    cur_sent = []
            else:
                word_ind, word, pred_ind, label, prob = line.split('\t')
                prob = float(prob)
                word_ind = int(word_ind)
                pred_ind = int(pred_ind)
                cur_sent.append(word)
                if word_ind == pred_ind:
                    pred_word = word
                cur_ex.append((label, prob))
        return (self.flatten_ret_dic(ret, 1),
                list(set(sents)))

    def flatten_ret_dic(self, dic, num_of_dups):
        """
        Given a dictionary of dictionaries, flatten it
        to a dictionary of lists
        """
        ret = defaultdict(lambda: [])
        for sent, preds_dic in dic.items():
            for pred, exs in preds_dic.items():
                ret[sent].extend(exs * num_of_dups)
        return ret

    def predict_sentence(self, sent):
        """
        Return a pre-predicted answer
        """
        return self.dic[" ".join(sent)]

if __name__ == "__main__":
    args = docopt(__doc__)
    logging.debug(args)
    model_dir = args["--model"]
    input_fn = args["--in"]
    output_fn = args["--out"]
    tokenize = args["--tokenize"]
    
    

    if model_dir:
        # If model dir is given, use it to load the model
        model = load_pretrained_model(model_dir)
        sents = [line.strip() for line in open(input_fn) if line.strip()]

    else:
        # If no model_dir is given, assume input file already contains annotations and probs
        model = Mock_model(input_fn)
        sents = model.sents

    oie = Trained_oie(model,
                      tokenize = tokenize)

    logging.debug("generating output for {} sentences".format(len(sents)))


    # Iterate over all raw sentences
    if args["--conll"]:
        save_dir = os.path.join(output_fn,model_dir.split('/')[-1]+'_'+input_fn.split('/')[-1]+".conll")
        with open(save_dir, 'w') as fout:
            fout.write('\n\n'.join([oie.conll_with_prob(sent.strip())
                                    for sent in sents]))

    else:
        save_dir = os.path.join(output_fn,input_fn.split('/')[-1]+".tri")
        with open(save_dir, 'w') as fout:
            fout.write('\n'.join([str(ex)
                                  for sent in sents
                                  for ex in oie.parse_sent(sent.strip())
            ]))
