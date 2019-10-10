CUDA_VISIBLE_DEVICES=7 python ./predict.py \
    --model=/data/liupq/transformer-oie/models/23_09_2019_15_38 \
    --in=/data/liupq/transformer-oie/data/wiki1.test.txt \
    --out=/data/liupq/transformer-oie/output \
    --conll