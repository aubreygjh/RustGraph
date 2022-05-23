dataset=$1
anomaly_ratio=$2
if [[ "$dataset" == "uci" ]]; 
then
    CUDA_VISIBLE_DEVICES=0 python main.py --dataset $dataset --snap_size 1000 --train_ratio 0.5 --anomaly_ratio $anomaly_ratio  \
    --epoch 200 --lr 0.001 --initial_epochs 60 --iter_num 5 --iter_epochs 60 \
    --x_dim 256 --h_dim 256 --z_dim 256 

elif [[ "$dataset" == "digg" ]]; 
then
    CUDA_VISIBLE_DEVICES=2 python main.py --dataset $dataset --snap_size 6000 --train_ratio 0.5 --anomaly_ratio $anomaly_ratio  \
    --epoch 400 --lr 0.0001 --initial_epochs 60 --iter_num 5 --iter_epochs 60 \
    --x_dim 256 --h_dim 100 --z_dim 100 

elif [[ "$dataset" == "btc_otc" ]]; 
then
    CUDA_VISIBLE_DEVICES=0 python main.py --dataset $dataset --snap_size 1000 --train_ratio 0.5 --anomaly_ratio $anomaly_ratio  \
    --epoch 400 --lr 0.0001 --initial_epochs 60 --iter_num 5 --iter_epochs 60 \
    --x_dim 64 --h_dim 64 --z_dim 64 

elif [[ "$dataset" == "btc_alpha" ]]; 
then
    CUDA_VISIBLE_DEVICES=3 python main.py --dataset $dataset --snap_size 2000 --train_ratio 0.5 --anomaly_ratio $anomaly_ratio  \
    --epoch 400 --lr 0.0005 --initial_epochs 60 --iter_num 5 --iter_epochs 60 \
    --x_dim 256 --h_dim 256 --z_dim 256 

elif [[ "$dataset" == "email" ]]; 
then
    CUDA_VISIBLE_DEVICES=0 python main.py --dataset $dataset --snap_size 2000 --train_ratio 0.5 --anomaly_ratio $anomaly_ratio  \
    --epoch 200 --lr 0.001 --initial_epochs 60 --iter_num 5 --iter_epochs 60 \
    --x_dim 256 --h_dim 256 --z_dim 256 

elif [[ "$dataset" == "as_topology" ]]; 
then
    CUDA_VISIBLE_DEVICES=0 python main.py --dataset $dataset --snap_size 6000 --train_ratio 0.5 --anomaly_ratio $anomaly_ratio  \
    --epoch 200 --lr 0.001 --initial_epochs 60 --iter_num 5 --iter_epochs 60 \
    --x_dim 48 --h_dim 48 --z_dim 48 
fi
