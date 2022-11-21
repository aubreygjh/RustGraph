#!bin/bash
dataset=$1
train_ratio_lst=(0.2 0.3 0.4 0.5 0.6)
# before executing, modify the model.py and main.py to print each timestamp's auc value!
if [[ "$dataset" == "uci" ]]; 
then
for train_ratio in "${train_ratio_lst[@]}"; do
    CUDA_VISIBLE_DEVICES=4 python main.py --dataset "uci" --snap_size 1000 --train_ratio $train_ratio --anomaly_ratio 0.1 --noise_ratio 0 \
    --epoch 200 --lr 0.001 --x_dim 128 --h_dim 128 --z_dim 128
done
elif [[ "$dataset" == "btc_otc" ]];
then
for train_ratio in "${train_ratio_lst[@]}"; do
    CUDA_VISIBLE_DEVICES=4 python main.py --dataset "btc_otc" --snap_size 1000 --train_ratio $train_ratio --anomaly_ratio 0.1 --noise_ratio 0 \
    --epoch 200 --lr 0.0005 --x_dim 128 --h_dim 128 --z_dim 128 
done
elif [[ "$dataset" == "email" ]]; 
then
for train_ratio in "${train_ratio_lst[@]}"; do
    CUDA_VISIBLE_DEVICES=4 python main.py --dataset "email" --snap_size 2000 --train_ratio $train_ratio --anomaly_ratio 0.1 --noise_ratio 0 \
    --epoch 200 --lr 0.001 --x_dim 512 --h_dim 512 --z_dim 512 
done
elif [[ "$dataset" == "btc_alpha" ]]; 
then
for train_ratio in "${train_ratio_lst[@]}"; do
    CUDA_VISIBLE_DEVICES=3 python main.py --dataset "btc_alpha" --snap_size 2000 --train_ratio $train_ratio --anomaly_ratio 0.1 --noise_ratio 0 \
    --epoch 400 --lr 0.0005  --x_dim 128 --h_dim 128 --z_dim 128 
done
fi