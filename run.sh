DATASET=$1
gpu_n=$2
multigpu=0
EPOCH=100
LR=0.01
num_layers=2
seed=5
BATCH_SIZE=32
SLIDE_WIN=5
dim=64
out_layer_num=1
SLIDE_STRIDE=1
topk=5
out_layer_inter_dim=128
val_ratio=0.2
decay=0


path_pattern="${DATASET}"
COMMENT="${DATASET}"


report='best'
if [[ "$DATASET" == "uci" ]]; then
    CUDA_VISIBLE_DEVICES=$gpu_n python main.py \
        --dataset $DATASET \
        --multi_gpu $multigpu \
        --epoch $EPOCH \
        --lr $LR \
        --num_layers $num_layers \
        --in_channels 1809 \
        --hidden_channels 256 \
        --out_channels 256 \
        --n_samples 8 \
        --timestamp 12 \
        --num_classes 2
    
fi

# if [[ "$gpu_n" == "cpu" ]]; then
#     python main.py \
#         -dataset $DATASET \
#         -save_path_pattern $path_pattern \
#         -slide_stride $SLIDE_STRIDE \
#         -slide_win $SLIDE_WIN \
#         -batch $BATCH_SIZE \
#         -epoch $EPOCH \
#         -comment $COMMENT \
#         -random_seed $seed \
#         -decay $decay \
#         -dim $dim \
#         -out_layer_num $out_layer_num \
#         -out_layer_inter_dim $out_layer_inter_dim \
#         -decay $decay \
#         -val_ratio $val_ratio \
#         -report $report \
#         -topk $topk \
#         -device 'cpu'
# else
#     CUDA_VISIBLE_DEVICES=$gpu_n  python main.py \
#         -dataset $DATASET \
#         -save_path_pattern $path_pattern \
#         -slide_stride $SLIDE_STRIDE \
#         -slide_win $SLIDE_WIN \
#         -batch $BATCH_SIZE \
#         -epoch $EPOCH \
#         -comment $COMMENT \
#         -random_seed $seed \
#         -decay $decay \
#         -dim $dim \
#         -out_layer_num $out_layer_num \
#         -out_layer_inter_dim $out_layer_inter_dim \
#         -decay $decay \
#         -val_ratio $val_ratio \
#         -report $report \
#         -topk $topk
# fi