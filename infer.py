import sys
sys.dont_write_bytecode = True
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
import random
import os
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import roc_auc_score
import argparse
from train_model import train_model
from train_attacker import train_attacker
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from utils import load_and_split_dataset, wrap_collect_outputs_and_labels, CNNModel
from torchvision.models import vgg16, resnet18, vgg19, alexnet

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='cifar10', help='Which dataset to use (cifar10 or mnist)')
parser.add_argument('--target_model', default='cnn', help='Which target model')
parser.add_argument('--shadow_model', default='cnn', help='Which target model')
parser.add_argument('--data_path', default='/home/pc/zhujie/data/cifar10', help='Path to store data')
parser.add_argument('--num_epoch', type=int, default=50, help='Number of epochs to train shadow/target models')

opt = parser.parse_args()


def setup_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     random.seed(seed)
     torch.backends.cudnn.deterministic = True


setup_seed(42)

class Attacker(nn.Module):
    def __init__(self, n_in, n_out=2, hidden = 64):
        super(Attacker, self).__init__()
        self.fc1 = nn.Linear(n_in, hidden)
        self.fc2 = nn.Linear(hidden, n_out)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.softmax(x)
        return x
	


device = 'cuda' if torch.cuda.is_available() else 'cpu'

### divide and load dataset for shadow/target model training
"""
we follow the setting in the given code but only obtain around 70% precision for CIFAR10. But it is lower than the reported result in the paper.
We produce 62% precision for MNIST, similar to the reported result in the paper.
"""
shadow_train_loader, shadow_out_loader, train_loader, test_loader, num_classes = load_and_split_dataset(dataset_type=opt.dataset, batch_size=100, root=opt.data_path)


## initial shadow/target model
model_dict = {'cnn':CNNModel, 'alexnet':alexnet, 'vgg16':vgg16, 'vgg19':vgg19, 'resnet18':resnet18}

if opt.dataset=='mnist':
    shape = [1,28,28]
elif opt.dataset == 'cifar10':
	shape = [3,32,32]


"""
Note that we do not support alexnet, vgg, and resnet18 on mnist currently. 
"""
if opt.shadow_model!='cnn':
    shadow_model = model_dict[opt.shadow_model](num_classes=num_classes)
else:
	shadow_model = model_dict[opt.shadow_model](n_in=shape, num_classes=num_classes)

if opt.target_model!='cnn':
    target_model = model_dict[opt.target_model](num_classes=num_classes)
else:
	target_model = model_dict[opt.target_model](n_in=shape, num_classes=num_classes)


target_model.to(device)
shadow_model.to(device)

## train shadow/target model
print('train shadow/target model')
shadow_model = train_model(device, shadow_train_loader, shadow_out_loader, shadow_model, epochs=50, learning_rate=0.001, l2_ratio=1e-7)
target_model = train_model(device, train_loader, test_loader, target_model, epochs=50, learning_rate=0.001, l2_ratio=1e-7)


## generate dataset to train attacker
attacktrainset, attacktestset = wrap_collect_outputs_and_labels(shadow_train_loader, shadow_out_loader, train_loader, test_loader, shadow_model, target_model, num_classes)

attacktrain_loader = DataLoader(attacktrainset, batch_size=10, shuffle=True)
attacktest_loader = DataLoader(attacktestset, batch_size=10, shuffle=True)

## initial attacker
attacker = Attacker(n_in=3 if num_classes>=3 else 2).cuda()

print('train attacker')
attacker = train_attacker(device, attacktrain_loader, attacktest_loader, attacker, epochs=50, learning_rate=0.01, l2_ratio=1e-6)

print('finish')

