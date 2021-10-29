gpu_n=$1
dataset=$2
multi_gpu=0
epoch=50
lr=0.01
weight_decay=0.01
num_layers=2
add_ev=$3

# seed=5
# BATCH_SIZE=32
# SLIDE_WIN=5
# dim=64
# out_layer_num=1
# SLIDE_STRIDE=1
# topk=5
# out_layer_inter_dim=128
# val_ratio=0.2

# path_pattern="${DATASET}"
# COMMENT="${DATASET}"
# report='best'


if [[ "$dataset" == "uci" ]]; then
    CUDA_VISIBLE_DEVICES=$gpu_n python main.py \
        --dataset $dataset \
        --multi_gpu $multi_gpu \
        --epoch $epoch \
        --lr $lr \
        --weight_decay  $weight_decay \
        --num_layers $num_layers \
        --num_nodes 1809 \
        --in_channels_gnn 1809 \
        --hidden_channels_gnn 512 \
        --out_channels_gnn 256 \
        --add_ev $add_ev \
        --n_samples 8 \
        --timestamp 6 \
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