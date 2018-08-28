# Baseline codes

## Undefended pytorch model

To train the network (and evaluate every epoch):

```
python pytorch_undefended.py --evaluate
```

To run prediction and make coverage-error plot:

```
python pytorch_undefended.py --resume=checkpoint.pth.tar --predict
```
