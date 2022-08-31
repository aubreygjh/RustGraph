dataset=$1
anomaly_ratio_lst=(0.01 0.05 0.1)
if [[ "$dataset" == "uci" ]]; 
then
for anomaly_ratio in "${anomaly_ratio_lst[@]}"; do
    CUDA_VISIBLE_DEVICES=0 python main.py --dataset $dataset --snap_size 1000 --train_ratio 0.5 --anomaly_ratio $anomaly_ratio  \
    --epoch 200 --lr 0.001 --initial_epochs 60 --iter_num 5 --iter_epochs 60 \
    --x_dim 256 --h_dim 256 --z_dim 256 
done
elif [[ "$dataset" == "digg" ]]; 
then
for anomaly_ratio in "${anomaly_ratio_lst[@]}"; do
    CUDA_VISIBLE_DEVICES=1 python main.py --dataset $dataset --snap_size 6000 --train_ratio 0.5 --anomaly_ratio $anomaly_ratio  \
    --epoch 400 --lr 0.0001 --initial_epochs 60 --iter_num 5 --iter_epochs 60 \
    --x_dim 256 --h_dim 100 --z_dim 100 
done
elif [[ "$dataset" == "btc_otc" ]]; 
then
for anomaly_ratio in "${anomaly_ratio_lst[@]}"; do
    CUDA_VISIBLE_DEVICES=1 python main.py --dataset $dataset --snap_size 1000 --train_ratio 0.5 --anomaly_ratio $anomaly_ratio  \
    --epoch 400 --lr 0.0005 --initial_epochs 60 --iter_num 5 --iter_epochs 60 \
    --x_dim 64 --h_dim 64 --z_dim 64 
done
elif [[ "$dataset" == "btc_alpha" ]]; 
then
for anomaly_ratio in "${anomaly_ratio_lst[@]}"; do
    CUDA_VISIBLE_DEVICES=1 python main.py --dataset $dataset --snap_size 2000 --train_ratio 0.5 --anomaly_ratio $anomaly_ratio  \
    --epoch 400 --lr 0.0005 --initial_epochs 60 --iter_num 5 --iter_epochs 60 \
    --x_dim 256 --h_dim 256 --z_dim 256 
done
elif [[ "$dataset" == "email" ]]; 
then
for anomaly_ratio in "${anomaly_ratio_lst[@]}"; do
    CUDA_VISIBLE_DEVICES=0 python main.py --dataset $dataset --snap_size 2000 --train_ratio 0.5 --anomaly_ratio $anomaly_ratio  \
    --epoch 200 --lr 0.001 --initial_epochs 60 --iter_num 5 --iter_epochs 60 \
    --x_dim 256 --h_dim 256 --z_dim 256 
done
elif [[ "$dataset" == "as_topology" ]]; 
then
for anomaly_ratio in "${anomaly_ratio_lst[@]}"; do
    CUDA_VISIBLE_DEVICES=0 python main.py --dataset $dataset --snap_size 6000 --train_ratio 0.5 --anomaly_ratio $anomaly_ratio  \
    --epoch 200 --lr 0.001 --initial_epochs 60 --iter_num 5 --iter_epochs 60 \
    --x_dim 48 --h_dim 48 --z_dim 48 
done
fi
