
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score, classification_report


def train_attacker(device, train_loader, test_loader, attacker, epochs=100, learning_rate=0.01, l2_ratio=1e-7):
    # Assuming that DataLoader returns a tuple of (inputs, targets)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(attacker.parameters(), lr=learning_rate, weight_decay=l2_ratio)

    print('Training...')
    for epoch in range(epochs):
        running_loss = 0.0
        for samples in train_loader:
            optimizer.zero_grad()
            inputs = samples['input'].to(device)
            targets = samples['label'].to(device)
            outputs = attacker(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        if epoch % 10 == 0:
            print('Epoch {}, train loss: {:.3f}'.format(epoch, running_loss))

    print('Testing...')
    with torch.no_grad():
        test_all_targets = []
        test_all_predicted = []
        for samples in test_loader:
            inputs = samples['input'].to(device)
            targets = samples['label'].to(device)
            outputs = attacker(inputs)
            _, predicted = torch.max(outputs, 1)
            test_all_targets.extend(targets.cpu().numpy())
            test_all_predicted.extend(predicted.cpu().numpy())

        print('Testing Accuracy: {:.4f}'.format(accuracy_score(test_all_targets, test_all_predicted)))

    print('More detailed results:')
    print(classification_report(test_all_targets, test_all_predicted))

    return attacker # finished trained.


















