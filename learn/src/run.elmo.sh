export BASE_DIR=/data/liupq/transformer-oie
# export TF_CPP_MIN_LOG_LEVEL=3
#train.noisy.oie
CUDA_VISIBLE_DEVICES=6 python train.py  \
    --train=$BASE_DIR/data/train.noisy.oie.conll \
    --dev=$BASE_DIR/data/dev.oie.conll   \
    --test=$BASE_DIR/data/test.oie.conll   \
    --load_hyperparams=$BASE_DIR/hyperparams/confidence_elmo.json   \
    --saveto=$BASE_DIR/models \
    --model_name=elmo
    #[--pretrained=MODEL_DIR]   \
    #[--missing_epochs=NUM]  \