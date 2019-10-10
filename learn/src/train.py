""" Usage:
    model --train=TRAIN_FN --dev=DEV_FN --test=TEST_FN [--load_hyperparams=MODEL_JSON] [--saveto=MODEL_DIR] [--model_name=MODEL_NAME] [--pretrained=MODEL_DIR] [--missing_epochs=NUM]
"""

from docopt import docopt
import logging
logging.basicConfig(level = logging.DEBUG)
import json
from pprint import pformat
import time
import os
import sys
sys.path.append("model")
sys.path.append("common")
# from model.rnn_oie import RNNOIE_model
from model.model_list import get_model

if __name__ == "__main__":
    args = docopt(__doc__) #高级传参模块
    logging.debug(args)
    train_fn = args["--train"]
    dev_fn = args["--dev"]
    test_fn = args["--test"]

    model_name = args["--model_name"]
    rnn=get_model(model_name)

    #完整训练
    if args["--pretrained"] is None:

        if args["--load_hyperparams"] is not None:
            # load hyperparams from json file
            json_fn = args["--load_hyperparams"]
            logging.info("Loading model from: {}".format(json_fn))
            rnn_params = json.load(open(json_fn))["rnn"]

        else:
            # Use some default params
            rnn_params = {"sent_maxlen":  20,
                          "hidden_units": pow(2, 10),#每个门结构中神经元的数量
                          "num_of_latent_layers": 2,
                          "epochs": 10,
                          "trainable_emb": True,
                          "batch_size": 50,
                          "emb_filename": "../../pretrained_word_embeddings/glove.6B.50d.txt",
            }
        logging.debug("hyperparams:\n{}".format(pformat(rnn_params)))

        if args["--saveto"] is not None:
            model_dir = os.path.join(args["--saveto"], "{}/".format(time.strftime("%d_%m_%Y_%H_%M")))
        else:
            model_dir = "../../models/{}/".format(time.strftime("%d_%m_%Y_%H_%M"))
        logging.debug("Saving models to: {}".format(model_dir))
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
        
        
        rnn.create_training_model(model_dir = model_dir,model_name=model_name,**rnn_params)
        rnn.train(train_fn, dev_fn, test_fn)

    #断点续练
    else:
        assert(args["--missing_epochs"] != None)
        missing_epochs=int(args["--missing_epochs"])
        rnn.load_pretrained_model(args["--pretrained"])
        rnn.train_again(train_fn, dev_fn, test_fn,missing_epochs)
        