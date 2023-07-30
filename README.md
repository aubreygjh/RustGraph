# RustGraph: Robust Anomaly Detection in Dynamic Graphs by Jointly Learning Structural-Temporal Dependency

<img src="framework.pdf">


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


