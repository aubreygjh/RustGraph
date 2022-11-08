#!bin/bash
dataset=$1
anomaly_ratio=$2
a_lst=(0.2 0.4 0.6 0.8 1.0)
b_lst=(0.2 0.4 0.6 0.8 1.0)
c_lst=(0.2 0.4 0.6 0.8 1.0)


if [[ "$dataset" == "uci" ]]; 
then
    for a in "${a_lst[@]}"; do
        for b in "${b_lst[@]}"; do
            for c in "${c_lst[@]}"; do
                CUDA_VISIBLE_DEVICES=0 python main.py --dataset $dataset --snap_size 1000 --train_ratio 0.5 --anomaly_ratio $anomaly_ratio  \
                --epoch 200 --lr 0.001 --x_dim 256 --h_dim 256 --z_dim 256 \
                --gen_loss_weight $a --con_loss_weight $b --at_alpha $c
            done
        done
    done
elif [[ "$dataset" == "btc_otc" ]]; 
then
    for a in "${a_lst[@]}"; do
        for b in "${b_lst[@]}"; do
            for c in "${c_lst[@]}"; do
                CUDA_VISIBLE_DEVICES=1 python main.py --dataset $dataset --snap_size 1000 --train_ratio 0.5 --anomaly_ratio $anomaly_ratio  \
                --epoch 400 --lr 0.0001 --x_dim 64 --h_dim 64 --z_dim 64  \
                --gen_loss_weight $a --con_loss_weight $b --at_alpha $c
            done
        done
    done
elif [[ "$dataset" == "btc_alpha" ]]; 
then
    for a in "${a_lst[@]}"; do
        for b in "${b_lst[@]}"; do
            CUDA_VISIBLE_DEVICES=2 python main.py --dataset $dataset --snap_size 2000 --train_ratio 0.5 --anomaly_ratio $anomaly_ratio  \
            --epoch 400 --lr 0.0005 --initial_epochs 60 --iter_num 5 --iter_epochs 60 \
            --x_dim 256 --h_dim 256 --z_dim 256   \
            --gen_loss_weight $a --con_loss_weight $b --at_alpha 1
        done
    done
    for a in "${a_lst[@]}"; do
        for c in "${c_lst[@]}"; do
            CUDA_VISIBLE_DEVICES=2 python main.py --dataset $dataset --snap_size 2000 --train_ratio 0.5 --anomaly_ratio $anomaly_ratio  \
            --epoch 400 --lr 0.0005 --initial_epochs 60 --iter_num 5 --iter_epochs 60 \
            --x_dim 256 --h_dim 256 --z_dim 256   \
            --gen_loss_weight $a --con_loss_weight 1 --at_alpha $c
        done
    done
    for b in "${b_lst[@]}"; do
        for c in "${c_lst[@]}"; do
            CUDA_VISIBLE_DEVICES=2 python main.py --dataset $dataset --snap_size 2000 --train_ratio 0.5 --anomaly_ratio $anomaly_ratio  \
            --epoch 400 --lr 0.0005 --initial_epochs 60 --iter_num 5 --iter_epochs 60 \
            --x_dim 256 --h_dim 256 --z_dim 256   \
            --gen_loss_weight 1 --con_loss_weight $b --at_alpha $c
        done
    done
    
fi