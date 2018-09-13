# Baseline codes

## Undefended pytorch model

To train the network (and evaluate every epoch):

```
cd examples/undefended_pytorch_resnet
python main.py --evaluate
```

To run prediction and make coverage-error plot:

```
gsutil cp gs://unrestricted-advex/example_defenses/undefended_pytorch_resnet.pth.tar /tmp/undefended_pytorch_resnet.pth.tar
CUDA_VISIBLE_DEVICES=1 python main.py --resume=/tmp/undefended_pytorch_resnet.pth.tar --evaluate
```
