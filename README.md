# RustGraph: Robust Anomaly Detection in Dynamic Graphs by Jointly Learning Structural-Temporal Dependency

The PyTorch implementation of the IEEE TKDE paper "RustGraph: Robust Anomaly Detection in Dynamic Graphs by Jointly Learning Structural-Temporal Dependency".

Link to this paper: https://doi.ieeecomputersociety.org/10.1109/TKDE.2023.3328645



## Requirements
```
python==3.9
pytorch==1.12.1
pytorch-geometric==2.3.0
tensorboard==2.6.0
networkx==3.1
matplotlib==3.7.1
```



## Reproducibility

To reproduce the main results in the paper (Section 5.2), execute `bash run.sh $DATASET`

To reproduce the results of noisy labels (Section 5.4), execute `bash exp_noise_ratio.sh $DATASET`

To reproduce the results of sensitivity analysis (Section 5.5), execute `bash exp_emb_dim.sh $DATASET`, `bash exp_train_ratio.sh $DATASET`, `bash exp_hyperparam.sh $DATASET`



## Citation

If you find this work interesting, please cite RustGraph using the following Bibtext:

```
@ARTICLE {rustgraph,
author = {J. Guo and S. Tang and J. Li and K. Pan and L. Wu},
journal = {IEEE Transactions on Knowledge &amp; Data Engineering},
title = {RustGraph: Robust Anomaly Detection in Dynamic Graphs by Jointly Learning Structural-Temporal Dependency},
year = {5555},
volume = {},
number = {01},
issn = {1558-2191},
pages = {1-14},
keywords = {noise measurement;anomaly detection;image edge detection;task analysis;training;representation learning;data models},
doi = {10.1109/TKDE.2023.3328645},
publisher = {IEEE Computer Society},
address = {Los Alamitos, CA, USA},
month = {oct}
}

```



## Contact

Please feel free to contact me through guojianhao@zju.edu.cn if you have any problems:)
