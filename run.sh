dataset=$1
anomaly_ratio=$2
if [[ "$dataset" == "uci" ]]; 
then
    python main.py --device cuda:3 --dataset $dataset --snap_size 1000 --train_ratio 0.5 --anomaly_ratio $anomaly_ratio  \
    --epoch 100 --lr 0.001  \
    --x_dim 256 --h_dim 256 --z_dim 256 

elif [[ "$dataset" == "digg" ]]; 
then
    python main.py --device cuda:3 --dataset $dataset --snap_size 6000 --train_ratio 0.5 --anomaly_ratio $anomaly_ratio  \
    --epoch 200 --lr 0.001 \
    --x_dim 256 --h_dim 100 --z_dim 100 

elif [[ "$dataset" == "btc_otc" ]]; 
then
    python main.py --device cuda:0 --dataset $dataset --snap_size 2000 --train_ratio 0.5 --anomaly_ratio $anomaly_ratio  \
    --epoch 200 --lr 0.001  \
    --x_dim 64 --h_dim 64 --z_dim 64 

elif [[ "$dataset" == "btc_alpha" ]]; 
then
    python main.py --device cuda:1 --dataset $dataset --snap_size 1000 --train_ratio 0.5 --anomaly_ratio $anomaly_ratio  \
    --epoch 200 --lr 0.001  \
    --x_dim 256 --h_dim 512 --z_dim 512 

elif [[ "$dataset" == "email" ]]; 
then
    python main.py --device cuda:0 --dataset $dataset --snap_size 500 --train_ratio 0.5 --anomaly_ratio $anomaly_ratio  \
    --epoch 100 --lr 0.001  \
    --x_dim 256 --h_dim 256 --z_dim 256 

elif [[ "$dataset" == "as_topology" ]]; 
then
    python main.py --device cuda:0 --dataset $dataset --snap_size 4000 --train_ratio 0.5 --anomaly_ratio $anomaly_ratio  \
    --epoch 100 --lr 0.001  \
    --x_dim 48 --h_dim 32 --z_dim 32 

elif [[ "$dataset" == "hepth" ]]; 
then
    python main.py --device cuda:2 --dataset $dataset --snap_size 4000 --train_ratio 0.5 --anomaly_ratio $anomaly_ratio  \
    --epoch 100 --lr 0.001  \
    --x_dim 256 --h_dim 256 --z_dim 256 
fi
