dataset=$1
anomaly_ratio_lst=(0.1 0.05 0.01)


if [[ "$dataset" == "uci" ]]; 
then
for anomaly_ratio in "${anomaly_ratio_lst[@]}"; do
    CUDA_VISIBLE_DEVICES=0 python main.py --dataset $dataset --snap_size 1000 --train_ratio 0.5 --anomaly_ratio $anomaly_ratio --noise_ratio 0 \
    --epoch 200 --lr 0.001 --x_dim 128 --h_dim 128 --z_dim 128 
done

elif [[ "$dataset" == "digg" ]]; 
then
for anomaly_ratio in "${anomaly_ratio_lst[@]}"; do
    CUDA_VISIBLE_DEVICES=0 python main.py --dataset $dataset --snap_size 6000 --train_ratio 0.5 --anomaly_ratio $anomaly_ratio --noise_ratio 0 \
    --epoch 400 --lr 0.0005 --x_dim 128 --h_dim 128 --z_dim 128 
done

elif [[ "$dataset" == "btc_otc" ]]; 
then
for anomaly_ratio in "${anomaly_ratio_lst[@]}"; do
    CUDA_VISIBLE_DEVICES=0 python main.py --dataset $dataset --snap_size 1000 --train_ratio 0.5 --anomaly_ratio $anomaly_ratio --noise_ratio 0 \
    --epoch 200 --lr 0.0005 --x_dim 128 --h_dim 128 --z_dim 128 
done

elif [[ "$dataset" == "btc_alpha" ]]; 
then
for anomaly_ratio in "${anomaly_ratio_lst[@]}"; do
    CUDA_VISIBLE_DEVICES=1 python main.py --dataset $dataset --snap_size 2000 --train_ratio 0.5 --anomaly_ratio $anomaly_ratio --noise_ratio 0 \
    --epoch 400 --lr 0.0005  --x_dim 128 --h_dim 128 --z_dim 128 
done

elif [[ "$dataset" == "email" ]]; 
then
for anomaly_ratio in "${anomaly_ratio_lst[@]}"; do
    CUDA_VISIBLE_DEVICES=0 python main.py --dataset $dataset --snap_size 2000 --train_ratio 0.5 --anomaly_ratio $anomaly_ratio --noise_ratio 0 \
    --epoch 200 --lr 0.001 --x_dim 512 --h_dim 512 --z_dim 512 
done

elif [[ "$dataset" == "as_topology" ]]; 
then
for anomaly_ratio in "${anomaly_ratio_lst[@]}"; do
    CUDA_VISIBLE_DEVICES=5 python main.py --dataset $dataset --snap_size 6000 --train_ratio 0.5 --anomaly_ratio $anomaly_ratio --noise_ratio 0 \
    --epoch 400 --lr 0.001 --x_dim 64 --h_dim 64 --z_dim 64 
done
fi
