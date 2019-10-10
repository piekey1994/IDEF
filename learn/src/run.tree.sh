export BASE_DIR=/data/liupq/transformer-oie
#train.noisy.oie
CUDA_VISIBLE_DEVICES=1 python train.py  \
    --train=$BASE_DIR/data/train.noisy.oie.conll \
    --dev=$BASE_DIR/data/dev.oie.conll   \
    --test=$BASE_DIR/data/test.oie.conll   \
    --load_hyperparams=$BASE_DIR/hyperparams/confidence.json   \
    --saveto=$BASE_DIR/models \
    --model_name=tree
    #[--pretrained=MODEL_DIR]   \
    #[--missing_epochs=NUM]  \