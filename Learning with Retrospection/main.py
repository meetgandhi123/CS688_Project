import os
import argparse
import time
import sys
import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
import torch.backends.cudnn as cudnn
from models.resnet56 import get_resnet_model
#from models.vgg16 import vgg16
from models.vgg import vgg16
from dataset.cifar100 import get_cifar100_dataloaders
from dataset.cifar10 import get_cifar10_dataloaders

class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def accuracy(output, target):
    with torch.no_grad():
        batch_size = target.size(0)
        _, pred = output.topk(1, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        correct_k = correct[:1].reshape(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
        return res

def get_traintarget():
    targets=torch.zeros(50000, dtype=torch.long)
    batch_size=200
    train_loader, val_loader, n = get_cifar100_dataloaders(batch_size=batch_size, num_workers=8, is_instance=True, is_soft=False, is_shuffle=False)
    for idx, (input, target, index) in enumerate(train_loader):
        targets[index] = target
    return targets

class Temperture_Softmax(nn.Module):
    def __init__(self, T):
        super(Temperture_Softmax, self).__init__()
        self.T = T

    def forward(self, y):    
        p = F.softmax(y/self.T, dim=1)
        return p

class KL(nn.Module):
    def __init__(self, T):
        super(KL, self).__init__()
        self.T = T

    def forward(self, y_s, p_t):
        p_s = F.log_softmax(y_s/self.T, dim=1)
        loss = F.kl_div(p_s, p_t, size_average=False) * (self.T**2) / p_s.shape[0]
        return loss   

def train_normal(epoch, train_loader, model, criterion, optimizer, options, train_logits):
    model.train()
    losses = AverageMeter()
    top1 = AverageMeter()
    
    if torch.cuda.is_available():
        train_logits=train_logits.cuda()
    
    for idx, (image, labels, softlabel, index) in enumerate(train_loader):
        image = image.float()
        if torch.cuda.is_available():
            image = image.cuda()
            labels = labels.cuda()
        output = model(image)
        train_logits[index] = output
        loss = criterion(output, labels)
        acc1 = accuracy(output, labels)
        losses.update(loss.item(), image.size(0))
        top1.update(acc1[0], image.size(0))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
    return top1.avg, losses.avg, train_logits




def train_lwr(epoch, train_loader, model, criterion_list, optimizer, options, train_logits):
    model.train()
    if torch.cuda.is_available():
        train_logits=train_logits.cuda()

    criterion_cls = criterion_list[0]
    critetion_soft = criterion_list[1]
    criterion_kl = criterion_list[2]

    losses = AverageMeter()
    top1 = AverageMeter()

    
    for idx, data in enumerate(train_loader):
        image, labels, logits, index = data
        image = image.float()
        if torch.cuda.is_available():
            image = image.cuda()
            labels = labels.cuda()
            logits = logits.cuda()
        
        soft_label = critetion_soft(logits)        
        preact = False
        logit_s = model(image, is_feat=False, preact=preact)
        train_logits[index] = logit_s
        loss_cls = criterion_cls(logit_s, labels)        
        loss_kl = criterion_kl(logit_s, soft_label)

        if epoch<=options["k"]:
            loss = loss_cls 
        else:
            num_5 = int(epoch/options["k"])
            cure = num_5*options["k"]
            
            loss = (options["gamma"]+(1-cure/240)*(1-options["gamma"])) * loss_cls + cure/240*(1-options["gamma"])* loss_kl

        acc1 = accuracy(logit_s, labels)
        losses.update(loss.item(), image.size(0))
        top1.update(acc1[0], image.size(0))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return top1.avg, losses.avg, train_logits



def validate(val_loader, model, criterion, options):
    
    losses = AverageMeter()
    top1 = AverageMeter()

    model.eval()

    with torch.no_grad():
        
        for idx, (image, label) in enumerate(val_loader):

            image = image.float()
            if torch.cuda.is_available():
                image = image.cuda()
                label = label.cuda()

            output = model(image)
            loss = criterion(output, label)
            acc1 = accuracy(output, label)
            losses.update(loss.item(), image.size(0))
            top1.update(acc1[0], image.size(0))
            
    return top1.avg,losses.avg

def getOptions():

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
  return options

options = getOptions()
if options["dataset"] == 'cifar100':                
    n_cls = 100
else:
    n_cls=10

if(options["model"]=="vgg16"):
    model = vgg16(n_cls)
else:
    model = get_resnet_model(n_cls)

print(model)
optimizer = optim.SGD(model.parameters(),
                          lr=options["learning_rate"],
                          momentum=options["momentum"],
                          weight_decay=options["weight_decay"])

criterion = nn.CrossEntropyLoss()  
criterion_cls = nn.CrossEntropyLoss()    
    
criterion_soft = Temperture_Softmax(options["t"])

criterion_kl = KL(options["t"])
criterion_list = nn.ModuleList([])
criterion_list.append(criterion_cls)    # classification loss
criterion_list.append(criterion_soft)    # KL divergence loss, original knowledge distillation
criterion_list.append(criterion_kl)

if torch.cuda.is_available():
    model = model.cuda()
    criterion = criterion.cuda()
    criterion_list.cuda()
    cudnn.benchmark = True

train_logits = torch.zeros((50000,n_cls))

if(n_cls==100):
    train_loader, val_loader = get_cifar100_dataloaders(batch_size=options["batch_size"], num_workers=options["num_workers"], is_instance=False, 
                                                          is_shuffle=True, is_soft=True, train_softlabels=train_logits)
else:
    train_loader, val_loader = get_cifar10_dataloaders(batch_size=options["batch_size"], num_workers=options["num_workers"], is_instance=False, 
                                                          is_shuffle=True, is_soft=True, train_softlabels=train_logits)

best_acc = 0
for epoch in range(1, int(options["epochs"])+ 1):

        if options["arch"]=='vgg':
          steps = int(epoch/20)
          if steps > 0:
              new_lr = options["learning_rate"] * (0.5 ** steps)
              for param_group in optimizer.param_groups:
                  param_group['lr'] = new_lr
        elif options["arch"]=='resnet':
            if epoch > 100 and epoch <=150:
                new_lr = options["learning_rate"] * 0.1
                for param_group in optimizer.param_groups:
                    param_group['lr'] = new_lr
            elif epoch > 150:
                new_lr = options["learning_rate"] * 0.1 * 0.1
                for param_group in optimizer.param_groups:
                    param_group['lr'] = new_lr
        else:
          print("wrong model..")


        if epoch<=options["k"]:
            train_acc, train_loss, train_logits= train_normal(epoch, train_loader, model, criterion, optimizer, options, train_logits)
        else:
            train_acc, train_loss, train_logits= train_lwr(epoch, train_loader, model, criterion_list, optimizer, options, train_logits)
         
        
        train_logits = train_logits.detach().cpu()
            
        if epoch>=options["k"] and epoch%options["k"]==0:
            print('label update')
            train_loader, val_loader = get_cifar100_dataloaders(batch_size=options["batch_size"], num_workers=options["num_workers"], is_instance=False, is_shuffle=True, is_soft=True, train_softlabels=train_logits)
        
        print('epoch {}'.format(epoch))


        test_acc, test_loss = validate(val_loader, model, criterion, options)

        print("Current Test Accuracy:",test_acc)
        # calculate best accuracy.
        if test_acc > best_acc:
            best_acc = test_acc
        
        print('best accuracy:', best_acc)

