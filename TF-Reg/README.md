# Teacher-free-Knowledge-Distillation

Build a new environment and install:
```
pip install -r requirements.txt
```

Vgg16 for CIFAR10 dataset
```
python main.py --model_dir experiments/base_experiments/base_vgg16_CIFAR10/ --regularization --num_class 10 
```

Vgg16 for CIFAR100 dataset
```
python main.py --model_dir experiments/base_experiments/base_vgg16_CIFAR100/ --regularization --num_class 100
```

Vgg16 for CIFAR10 dataset
```
python main.py --model_dir experiments/base_experiments/base_resnet56_CIFAR10/ --regularization --num_class 10 
```

Vgg16 for CIFAR10 dataset
```
python main.py --model_dir experiments/base_experiments/base_resnet56_CIFAR100/ --regularization --num_class 100 
```
