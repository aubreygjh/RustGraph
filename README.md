# RustGraph
To reproduce the results in the paper, execute `bash run.sh`
## Requirements
python==3.9
pytorch
pytorch-geometric
tensorboard
networkx
matplotlib
## 0. Python environment setup with Conda
```
conda create --name TAPE python=3.8
conda activate TAPE

conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=11.3 -c pytorch
conda install -c pyg pytorch-sparse
conda install -c pyg pytorch-scatter
conda install -c pyg pytorch-cluster
conda install -c pyg pyg
pip install ogb
conda install -c dglteam/label/cu113 dgl
pip install yacs
pip install transformers
pip install --upgrade accelerate
```