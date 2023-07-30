#!bin/bash
dataset=$1
a_lst=(0.2 0.4 0.6 0.8 1.0)
b_lst=(0.2 0.4 0.6 0.8 1.0)
c_lst=(0.2 0.4 0.6 0.8 1.0)


if [[ "$dataset" == "uci" ]]; 
then
    for a in "${a_lst[@]}"; do
        for b in "${b_lst[@]}"; do
            CUDA_VISIBLE_DEVICES=0 python main.py --dataset "uci" --snap_size 1000 --train_ratio 0.5 --anomaly_ratio 0.1  \
            --epoch 200 --lr 0.001 --x_dim 128 --h_dim 128 --z_dim 128   \
            --bce_weight 1 --reg_weight 1 --gen_weight $a --con_weight $b
        done
    done
    for a in "${a_lst[@]}"; do
        for b in "${b_lst[@]}"; do
            CUDA_VISIBLE_DEVICES=0 python main.py --dataset "uci" --snap_size 1000 --train_ratio 0.5 --anomaly_ratio 0.1  \
            --epoch 200 --lr 0.001 --x_dim 128 --h_dim 128 --z_dim 128   \
            --bce_weight $a --reg_weight $b --gen_weight 1
        done
    done
    for a in "${a_lst[@]}"; do
        for c in "${c_lst[@]}"; do
            CUDA_VISIBLE_DEVICES=0 python main.py --dataset "uci" --snap_size 1000 --train_ratio 0.5 --anomaly_ratio 0.1  \
            --epoch 200 --lr 0.001 --x_dim 128 --h_dim 128 --z_dim 128   \
            --bce_weight $a --reg_weight 1 --gen_weight $c
        done
    done
    for b in "${b_lst[@]}"; do
        for c in "${c_lst[@]}"; do
            CUDA_VISIBLE_DEVICES=0 python main.py --dataset "uci" --snap_size 1000 --train_ratio 0.5 --anomaly_ratio 0.1  \
            --epoch 200 --lr 0.001 --x_dim 128 --h_dim 128 --z_dim 128   \
            --bce_weight 1 --reg_weight $b --gen_weight $c
        done
    done
elif [[ "$dataset" == "btc_otc" ]]; 
then
    for a in "${a_lst[@]}"; do
        for b in "${b_lst[@]}"; do
            CUDA_VISIBLE_DEVICES=2 python main.py --dataset "btc_otc" --snap_size 1000 --train_ratio 0.5 --anomaly_ratio 0.1  \
            --epoch 200 --lr 0.0005 --x_dim 128 --h_dim 128 --z_dim 128   \
            --bce_weight $a --reg_weight $b --gen_weight 1
        done
    done
    for a in "${a_lst[@]}"; do
        for c in "${c_lst[@]}"; do
            CUDA_VISIBLE_DEVICES=2 python main.py --dataset "btc_otc" --snap_size 1000 --train_ratio 0.5 --anomaly_ratio 0.1  \
            --epoch 200 --lr 0.0005 --x_dim 128 --h_dim 128 --z_dim 128   \
            --bce_weight $a --reg_weight 1 --gen_weight $c
        done
    done
    for b in "${b_lst[@]}"; do
        for c in "${c_lst[@]}"; do
            CUDA_VISIBLE_DEVICES=2 python main.py --dataset "btc_otc" --snap_size 1000 --train_ratio 0.5 --anomaly_ratio 0.1  \
            --epoch 200 --lr 0.0005 --x_dim 128 --h_dim 128 --z_dim 128   \
            --bce_weight 1 --reg_weight $b --gen_weight $c
        done
    done
elif [[ "$dataset" == "email" ]]; 
then
    for a in "${a_lst[@]}"; do
        for b in "${b_lst[@]}"; do
            CUDA_VISIBLE_DEVICES=4 python main.py --dataset "email" --snap_size 2000 --train_ratio 0.5 --anomaly_ratio 0.1  \
            --epoch 200 --lr 0.001 --x_dim 512 --h_dim 512 --z_dim 512   \
            --bce_weight $a --reg_weight $b --gen_weight 1
        done
    done
    for a in "${a_lst[@]}"; do
        for c in "${c_lst[@]}"; do
            CUDA_VISIBLE_DEVICES=4 python main.py --dataset "email" --snap_size 2000 --train_ratio 0.5 --anomaly_ratio 0.1  \
            --epoch 200 --lr 0.001 --x_dim 512 --h_dim 512 --z_dim 512   \
            --bce_weight $a --reg_weight 1 --gen_weight $c
        done
    done
    for b in "${b_lst[@]}"; do
        for c in "${c_lst[@]}"; do
            CUDA_VISIBLE_DEVICES=4 python main.py --dataset "email" --snap_size 2000 --train_ratio 0.5 --anomaly_ratio 0.1  \
            --epoch 200 --lr 0.001 --x_dim 512 --h_dim 512 --z_dim 512   \
            --bce_weight 1 --reg_weight $b --gen_weight $c
        done
    done  
fi