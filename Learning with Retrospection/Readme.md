Learning with Retrospection 

Change the main.py file for different dataset / models.

For CIFAR10 with Resnet56
```
options = {
      "batch_size": 128,
      "num_workers": 8,
      "epochs": 20,
      "learning_rate": 0.1,
      "weight_decay": 5e-4,
      "momentum": 0.9,
      "model": "resnet56",
      "arch": "resnet",
      "dataset": "cifar10",
      "k": 5,
      "t": 10,
      "gamma":0.1
  }
```

For CIFAR100 with Resnet56
```
options = {
      "batch_size": 128,
      "num_workers": 8,
      "epochs": 20,
      "learning_rate": 0.1,
      "weight_decay": 5e-4,
      "momentum": 0.9,
      "model": "resnet56",
      "arch": "resnet",
      "dataset": "cifar100",
      "k": 5,
      "t": 10,
      "gamma":0.1
  }
```

For CIFAR10 with VGG16
```
options = {
      "batch_size": 128,
      "num_workers": 8,
      "epochs": 20,
      "learning_rate": 0.1,
      "weight_decay": 5e-4,
      "momentum": 0.9,
      "model": "vgg16",
      "arch": "vgg",
      "dataset": "cifar10",
      "k": 5,
      "t": 10,
      "gamma":0.1
  }
```

For CIFAR100 with VGG16
```
options = {
      "batch_size": 128,
      "num_workers": 8,
      "epochs": 20,
      "learning_rate": 0.1,
      "weight_decay": 5e-4,
      "momentum": 0.9,
      "model": "vgg16",
      "arch": "vgg",
      "dataset": "cifar100",
      "k": 5,
      "t": 10,
      "gamma":0.1
  }
```
