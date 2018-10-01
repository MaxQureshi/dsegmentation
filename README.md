# Shape Aware Weight Maps for Instances Segmentation
This repos contains all implementations used in the paper "MULTICLASS WEIGHTED LOSS FOR INSTANCE SEGMENTATION OF CLUTTERED CELLS" accepted at [ICIP 2018](https://ieeexplore.ieee.org/document/8451187).

## Clonning the repository
This repository depends on **netframework** submodule. To correctly clone **dsegmentation** do:
>git clone --recursive https://github.com/fagp/dsegmentation.git

## Dependencies
The code was tested on Ubuntu 16.04 and Python 3.6 version of Anaconda 5.2. 
The additional packages used are:
- pytorch v0.4.1
- visdom v0.1.8.5
- torchvision v0.2.1

## Computing weight maps
 To precompute shape aware weight maps use `createwm/test.py` script. Change `path_images` variable to indicate the ground truth path. Images are assumed to have png extensions (line 26).

## Data preparation
Your own dataset can be used for network training. Training datasets must be defined in `src/defaults/dataconfig_train.json`. For every entry in `dataconfig_train.json` must be an entry with the same key in `src/defaults/dataconfig_test.json` for validation dataset specifications. More information about defaults definitions can be see in [netframework](https://github.com/fagp/netframework). 

## Network training 
For training you can simply do
> bash workload.sh

A sequence of experiments defined in `experiments/experiments.param` will be executed. Every line define a different experiment. The complete list of arguments are defined in `src/netframework/netutil/NetFramework.py`. You can create your experiment by using this parameters:

- `--experiment`: Name used in the output folder and visdom visualization
- `--model`: Architecture name e.g. unet_3. Defined in `src/defaults/modelconfig.json`
- `--modelparam`: Dictionary with model parameters to be used
- `--dataset`: Dataset to be used. Corresponds to datasets key specified in `src/defaults/dataconfig_*.json`
- `--datasetparam`: Dictionary with dataset parameters to be used
- `--visdom`: Use it only if visdom visualization is required
- `--show_rate`: Visdom plots after num of iterations (used with --visdom)
- `--print_rate`: Print after num of iterations
- `--save_rate`: Save models after num of epochs (if --save_rate=0 then no save is done during training)
- `--use_cuda`: GPU device (if --use_cuda=-1 then CPU used)
- `--parallel`: Use multiples GPU (used only if --use_cuda>-1)
- `--epochs`: Training epochs
- `--batch_size`: Minibatch size
- `--train_worker`: Number of training workers
- `--test_worker`: Number of validation workers
- `--optimizer`: Optimizer to be used `['Adam','SGD','RMSprop']`
- `--optimizerparam`: Dictionary with optimizer parameters e.g. {'lr':'0.0001'}
- `--lrschedule`: Learning rate schedule to be used  `['rop','step','exp','none']`
- `--loss`: Loss function to be used e.g. `wce` (Weighted Cross Entropy). Defined in `src/defaults/loss_definition.json`
- `--lossparam`: Loss function parameters
- `--resume`: Use it to resume training



