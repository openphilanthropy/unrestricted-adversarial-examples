## Installation

```bash
cd unrestricted-adversarial-examples/baselines
export PYTHONPATH=$PYTHONPATH:`pwd`
```

Fetch Imagenet
```bash
mkdir -p ~/datasets/cloudtpu-imagenet-data && gsutil -m rsync -r gs://cloudtpu-imagenet-data ~/datasets/cloudtpu-imagenet-data
```

Update tensorflow
```bash
pip uninstall tensorboard-tensorflow tensorflow tensorflow-gpu
pip install tensorflow-gpu==1.10.0rc1
```

[Optional] Fetch checkpoints
```bash
mkdir -p /root/tb/tcu-imagenet/
gsutil -m rsync -r gs://tomfeelslucky-experiments/tb/tcu-imagenet/ /root/tb/tcu-imagenet/
```

## Train resnet-50 on imagenet

This should take about 11 hours on 8 V100 GPUs and should result in an accuracy of 75.56%

(Note that this uses fp16 weights)
```bash
export model_dir="/root/tb/resnet_official/bs=2048-`date +%s`" &&\
python python unrestricted_advex/tf_official_resnet_baseline/imagenet_main.py --model_dir "$model_dir" --batch_size 2048
```

## Overfit a single batch

```bash
CUDA_VISIBLE_DEVICES=0 python imagenet_main.py --model_dir /root/tb/test_overfit_clean/overfit-`timestamp` --batch_size 32 --train_epochs 10000 --epochs_between_evals=10000 --repeat_single_batch
```

Evaluate on a single batch
```bash
CUDA_VISIBLE_DEVICES=1 python eval_attacks.py --batch_size 32 --repeat_single_batch --model_dir= '/root/tb/test_overfit_clean/overfit-2018-08-04_16-16-15/model.ckpt-3000'
```
