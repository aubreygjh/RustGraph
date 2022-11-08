#!bin/bash
dataset=$1
anomaly_ratio_lst=(0.1 0.05 0.01)
emb_dim_lst=(8 16 32 64 128 256 512)


if [[ "$dataset" == "uci" ]]; 
then
for anomaly_ratio in "${anomaly_ratio_lst[@]}"; do
for emb_dim in "${emb_dim_lst[@]}"; do
    CUDA_VISIBLE_DEVICES=2 python main.py --dataset "uci" --snap_size 1000 --train_ratio 0.5 --anomaly_ratio $anomaly_ratio --noise_ratio 0 \
    --epoch 400 --lr 0.001 --x_dim $emb_dim --h_dim $emb_dim --z_dim $emb_dim
done
done
elif [[ "$dataset" == "digg" ]];
then
for anomaly_ratio in "${anomaly_ratio_lst[@]}"; do
for emb_dim in "${emb_dim_lst[@]}"; do
    CUDA_VISIBLE_DEVICES=0 python main.py --dataset "digg" --snap_size 6000 --train_ratio 0.5 --anomaly_ratio $anomaly_ratio --noise_ratio 0 \
    --epoch 600 --lr 0.0005 --x_dim $emb_dim --h_dim $emb_dim --z_dim $emb_dim
done
done
elif [[ "$dataset" == "btc_otc" ]];
then
for anomaly_ratio in "${anomaly_ratio_lst[@]}"; do
for emb_dim in "${emb_dim_lst[@]}"; do
    CUDA_VISIBLE_DEVICES=2 python main.py --dataset "btc_otc" --snap_size 1000 --train_ratio 0.5 --anomaly_ratio $anomaly_ratio --noise_ratio 0 \
    --epoch 600 --lr 0.0005 --x_dim $emb_dim --h_dim $emb_dim --z_dim $emb_dim
done
done
elif [[ "$dataset" == "btc_alpha" ]]; 
then
for anomaly_ratio in "${anomaly_ratio_lst[@]}"; do
for emb_dim in "${emb_dim_lst[@]}"; do
    CUDA_VISIBLE_DEVICES=0 python main.py --dataset "btc_alpha" --snap_size 2000 --train_ratio 0.5 --anomaly_ratio $anomaly_ratio --noise_ratio 0 \
    --epoch 600 --lr 0.0005  --x_dim $emb_dim --h_dim $emb_dim --z_dim $emb_dim
done
done
elif [[ "$dataset" == "email" ]]; 
then
for anomaly_ratio in "${anomaly_ratio_lst[@]}"; do
for emb_dim in "${emb_dim_lst[@]}"; do
    CUDA_VISIBLE_DEVICES=1 python main.py --dataset "email" --snap_size 2000 --train_ratio 0.5 --anomaly_ratio $anomaly_ratio --noise_ratio 0 \
    --epoch 400 --lr 0.001 --x_dim $emb_dim --h_dim $emb_dim --z_dim $emb_dim 
done
done
elif [[ "$dataset" == "as_topology" ]]; 
then
for anomaly_ratio in "${anomaly_ratio_lst[@]}"; do
for emb_dim in "${emb_dim_lst[@]}"; do
    CUDA_VISIBLE_DEVICES=2 python main.py --dataset "as_topology" --snap_size 6000 --train_ratio 0.5 --anomaly_ratio $anomaly_ratio --noise_ratio 0 \
    --epoch 400 --lr 0.001 --x_dim $emb_dim --h_dim $emb_dim --z_dim $emb_dim
done
done
fi