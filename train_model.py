import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score, classification_report

# 
# shadow_train_loader, shadow_out_loader, train_loader, test_loader, num_classes = load_and_split_dataset()

def train_model(device, train_loader, test_loader, model, epochs=100, learning_rate=0.01, l2_ratio=1e-7):
    # Assuming that DataLoader returns a tuple of (inputs, targets)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=l2_ratio)

    print('Training...')
    for epoch in range(epochs):
        running_loss = 0.0
        for inputs, targets in train_loader:
            optimizer.zero_grad()
            inputs = inputs.to(device)
            targets = targets.to(device)
            outputs = model(inputs)
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
        for inputs, targets in test_loader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            test_all_targets.extend(targets.cpu().numpy())
            test_all_predicted.extend(predicted.cpu().numpy())

        print('Testing Accuracy: {:.4f}'.format(accuracy_score(test_all_targets, test_all_predicted)))

    print('More detailed results:')
    print(classification_report(test_all_targets, test_all_predicted))

    return model # finished trained.

