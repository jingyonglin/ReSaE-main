# ReSaE
hyper-relational knowledge graph encoder 



## Requirements
* Python>=3.8
* PyTorch 1.8.0
* torch-geometric 2.2.0
* torch-scatter 2.0.6
* torch-sparse 0.6.10
* torch-spline 1.2.1
* torch-cluster 1.5.9
* tqdm


## Experiments
cuda 11.0
### Datasets
Specified as `DATASET` in the running script
* `jf17k`
* `wikipeople`
* `wd50k` [default]
* `wd50k_33` 
* `wd50k_66`
* `wd50k_100`

### training

* `train on wd50k`
> python run.py DATASET wd50k
* `train on wd50k 66% ratio data with qualifier on device cuda:0`
> python run.py DEVICE cuda:0 DATASET wd50k_66

* `train on jf17k data with hyper-relational setting, on deive cuda:0`
> python run.py DEVICE cuda:0 DATASET jf17k CLEANED_DATASET False
* `train on wikipeople with hyper-relational setting, on device cuda:0`
> python run.py DEVICE cuda:0 DATASET wikipeople CLEANED_DATASET False


### acknowledge

The code framework is derived from https://github.com/migalkin/StarE
