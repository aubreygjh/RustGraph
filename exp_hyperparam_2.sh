#!bin/bash
dataset=$1
a_lst=(0.2 0.4 0.6 0.8 1.0)
b_lst=(0.2 0.4 0.6 0.8 1.0)
c_lst=(0.2 0.4 0.6 0.8 1.0)


if [[ "$dataset" == "uci_bce" ]]; 
then
    for a in "${a_lst[@]}"; do
            CUDA_VISIBLE_DEVICES=1 python main.py --dataset "uci" --snap_size 1000 --train_ratio 0.5 --anomaly_ratio 0.1  \
            --epoch 200 --lr 0.001 --x_dim 128 --h_dim 128 --z_dim 128   \
            --bce_weight 0 --reg_weight $a --gen_weight 1 --con_weight 1
    done
    for a in "${a_lst[@]}"; do
            CUDA_VISIBLE_DEVICES=1 python main.py --dataset "uci" --snap_size 1000 --train_ratio 0.5 --anomaly_ratio 0.1  \
            --epoch 200 --lr 0.001 --x_dim 128 --h_dim 128 --z_dim 128   \
            --bce_weight 0 --reg_weight 1 --gen_weight $a --con_weight 1
    done
    for a in "${a_lst[@]}"; do
            CUDA_VISIBLE_DEVICES=1 python main.py --dataset "uci" --snap_size 1000 --train_ratio 0.5 --anomaly_ratio 0.1  \
            --epoch 200 --lr 0.001 --x_dim 128 --h_dim 128 --z_dim 128   \
            --bce_weight 0 --reg_weight 1 --gen_weight 1 --con_weight $a
    done
elif [[ "$dataset" == "uci_reg" ]]; 
then
    for a in "${a_lst[@]}"; do
            CUDA_VISIBLE_DEVICES=2 python main.py --dataset "uci" --snap_size 1000 --train_ratio 0.5 --anomaly_ratio 0.1  \
            --epoch 200 --lr 0.001 --x_dim 128 --h_dim 128 --z_dim 128   \
            --bce_weight $a --reg_weight 0 --gen_weight 1 --con_weight 1
    done
    for a in "${a_lst[@]}"; do
            CUDA_VISIBLE_DEVICES=2 python main.py --dataset "uci" --snap_size 1000 --train_ratio 0.5 --anomaly_ratio 0.1  \
            --epoch 200 --lr 0.001 --x_dim 128 --h_dim 128 --z_dim 128   \
            --bce_weight 1 --reg_weight 0 --gen_weight $a --con_weight 1
    done
    for a in "${a_lst[@]}"; do
            CUDA_VISIBLE_DEVICES=2 python main.py --dataset "uci" --snap_size 1000 --train_ratio 0.5 --anomaly_ratio 0.1  \
            --epoch 200 --lr 0.001 --x_dim 128 --h_dim 128 --z_dim 128   \
            --bce_weight 1 --reg_weight 0 --gen_weight 1 --con_weight $a
    done
elif [[ "$dataset" == "uci_gen" ]]; 
then
    for a in "${a_lst[@]}"; do
            CUDA_VISIBLE_DEVICES=1 python main.py --dataset "uci" --snap_size 1000 --train_ratio 0.5 --anomaly_ratio 0.1  \
            --epoch 200 --lr 0.001 --x_dim 128 --h_dim 128 --z_dim 128   \
            --bce_weight $a --reg_weight 1 --gen_weight 0 --con_weight 1
    done
    for a in "${a_lst[@]}"; do
            CUDA_VISIBLE_DEVICES=1 python main.py --dataset "uci" --snap_size 1000 --train_ratio 0.5 --anomaly_ratio 0.1  \
            --epoch 200 --lr 0.001 --x_dim 128 --h_dim 128 --z_dim 128   \
            --bce_weight 1 --reg_weight $a --gen_weight 0 --con_weight 1
    done
    for a in "${a_lst[@]}"; do
            CUDA_VISIBLE_DEVICES=1 python main.py --dataset "uci" --snap_size 1000 --train_ratio 0.5 --anomaly_ratio 0.1  \
            --epoch 200 --lr 0.001 --x_dim 128 --h_dim 128 --z_dim 128   \
            --bce_weight 1 --reg_weight 1 --gen_weight 0 --con_weight $a
    done 
elif [[ "$dataset" == "uci_con" ]]; 
then
    for a in "${a_lst[@]}"; do
            CUDA_VISIBLE_DEVICES=2 python main.py --dataset "uci" --snap_size 1000 --train_ratio 0.5 --anomaly_ratio 0.1  \
            --epoch 200 --lr 0.001 --x_dim 128 --h_dim 128 --z_dim 128   \
            --bce_weight $a --reg_weight 1 --gen_weight 1 --con_weight 0
    done
    for a in "${a_lst[@]}"; do
            CUDA_VISIBLE_DEVICES=2 python main.py --dataset "uci" --snap_size 1000 --train_ratio 0.5 --anomaly_ratio 0.1  \
            --epoch 200 --lr 0.001 --x_dim 128 --h_dim 128 --z_dim 128   \
            --bce_weight 1 --reg_weight $a --gen_weight 1 --con_weight 0
    done
    for a in "${a_lst[@]}"; do
            CUDA_VISIBLE_DEVICES=2 python main.py --dataset "uci" --snap_size 1000 --train_ratio 0.5 --anomaly_ratio 0.1  \
            --epoch 200 --lr 0.001 --x_dim 128 --h_dim 128 --z_dim 128   \
            --bce_weight 1 --reg_weight 1 --gen_weight $a --con_weight 0
    done 
fi