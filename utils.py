import torch
from torchvision import datasets, transforms
from torch.utils.data import Subset, DataLoader
import numpy as np
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset, DataLoader


def load_and_split_dataset(dataset_type='cifar10', batch_size=64, root='/home/pc/zhujie/data/cifar10'):


    # load dataset
    if dataset_type == 'mnist':
        # define data pipeline
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
        train_dataset_total = datasets.MNIST(root=root, train=True, download=True, transform=transform)
        num_classes = 10
    elif dataset_type == 'cifar10':
        cifar10_mean = (0.4914, 0.4822, 0.4465)
        cifar10_std = (0.2023, 0.1994, 0.2010)
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(cifar10_mean, cifar10_std)])
        train_dataset_total = datasets.CIFAR10(root=root, train=True, download=True, transform=transform)
        num_classes = 10
    else:
        raise ValueError("Unsupported dataset type. Supported types are 'mnist' and 'cifar10'.")

    # divide the dataset
    total_size = len(train_dataset_total)
    subset_size = total_size // 4
    # 
    # generate the index
    indices = np.random.permutation(total_size)
    print(indices[:10]) 
    
    """
    [20061 33675 33022  7405 25549 22179  1210 40685 23029  3124] the order generated in my machine for cifar10
    [12628 37730 39991  8525  8279 51012 14871 15127  9366 33322] the order generated in my machine for mnist
    """
    # create subset
    shadow_train_dataset = Subset(train_dataset_total, indices[:subset_size])
    shadow_out_dataset = Subset(train_dataset_total, indices[subset_size:2 * subset_size])
    train_dataset = Subset(train_dataset_total, indices[2 * subset_size:3 * subset_size])
    test_dataset = Subset(train_dataset_total, indices[3 * subset_size:])

    # create dataloader
    shadow_train_loader = DataLoader(shadow_train_dataset, batch_size=batch_size, shuffle=True, num_workers=8)
    shadow_out_loader = DataLoader(shadow_out_dataset, batch_size=batch_size, shuffle=True, num_workers=8)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=8)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, num_workers=8)

    # print data size
    print(f"Shadow Train Size: {len(shadow_train_loader.dataset)}")
    print(f"Shadow Out Size: {len(shadow_out_loader.dataset)}")
    print(f"Train Size: {len(train_loader.dataset)}")
    print(f"Test Size: {len(test_loader.dataset)}")

    return shadow_train_loader, shadow_out_loader, train_loader, test_loader, num_classes


class AttackDataset(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = {'input': self.data[idx], 'label': self.labels[idx]}
        return sample


## collect data to train the attacker

def collect_outputs_and_labels(train_loader, test_loader, model):
    model.eval()  # Set the model to evaluation mode

    all_outputs = []
    all_labels = []

    # Collect outputs and labels for train_loader
    with torch.no_grad():
        for inputs, _ in train_loader:
            inputs = inputs.cuda()
            outputs = model(inputs)
            all_outputs.append(outputs)
            all_labels.append(torch.ones(outputs.shape[0]).cuda().long())  # Label 1 for train_loader

    # Collect outputs and labels for test_loader
    with torch.no_grad():
        for inputs, _ in test_loader:
            inputs = inputs.cuda()
            outputs = model(inputs)
            all_outputs.append(outputs)
            all_labels.append(torch.zeros(outputs.shape[0]).cuda().long())  # Label 0 for test_loader

    # Stack the outputs and labels
    stacked_outputs = torch.cat(all_outputs, dim=0)
    stacked_labels = torch.cat(all_labels, dim=0)

    return stacked_outputs, stacked_labels

def wrap_collect_outputs_and_labels(shadow_train_loader, shadow_out_loader, train_loader, test_loader, shadow_model, target_model, num_class):
    outputs, labels = collect_outputs_and_labels(train_loader, test_loader, target_model)
    shadow_outputs, shadow_labels = collect_outputs_and_labels(shadow_train_loader, shadow_out_loader, shadow_model)

    if num_class>=3:
        shadow_outputs = torch.sort(shadow_outputs, dim=1, descending=True)[0][:, :3]
        print(torch.sort(outputs, dim=1, descending=True)[0][:3,:])
        outputs = torch.sort(outputs, dim=1, descending=True)[0][:, :3]
    else:
        shadow_outputs = torch.sort(shadow_outputs, dim=1, descending=True)[0][:2]
        outputs = torch.sort(outputs, dim=1, descending=True)[0][:2]

    print(shadow_outputs.size())
    print(shadow_labels.size())
    trainset = AttackDataset(shadow_outputs, shadow_labels)
    testset = AttackDataset(outputs,labels)

    return trainset, testset


class CNNModel(nn.Module):
    def __init__(self, n_in, num_classes, n_hidden=128):
        super(CNNModel, self).__init__()

        self.conv1 = nn.Conv2d(n_in[0], 32, kernel_size=5, padding=2)
        self.maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(32, 32, kernel_size=5)
        self.maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.relu2 = nn.ReLU()
        self.fc = nn.Linear(32 * (n_in[1]//4-2)*(n_in[2]//4-2), n_hidden)
        self.tanh = nn.Tanh()

        self.output = nn.Linear(n_hidden, num_classes)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.maxpool1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.maxpool2(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        x = self.tanh(x)
        x = self.output(x)
        x = self.softmax(x)
        return x



