import json 
from nltk_utils import tokenize, stem, bag_of_words
from model import NeuralNet

import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

import matplotlib.pyplot as plt

def fetch_from_json(json_path):
    with open(json_path, 'r') as f:
        intents = json.load(f)

    all_words = []
    tags = []
    xy = []

    for intent in intents['intents']:
        tag = intent['tag']
        tags.append(tag)
    
        for pattern in intent['patterns']:
            # tokenize training pattern
            tokens = tokenize(pattern)

            # make a set of all tokends 
            all_words.extend(tokens)
            xy.append((tokens, tag))

    ignored_words = ['?', '!', '.', ',']
    all_words = [stem(w) for w in all_words if w not in ignored_words]
    all_words = sorted(set(all_words))
    tags = sorted(set(tags))

    return tags, all_words, xy


def prepare_training_data(tags, all_words, xy):
    X_train = []
    y_train = []

    # for each tokenized pattern
    for (pattern_sentence, tag) in xy:

        # bag has len(all_words) for each tokenized pattern
        bag = bag_of_words(pattern_sentence, all_words)
        X_train.append(bag)

        label = tags.index(tag)
        y_train.append(label)

    X_train = np.array(X_train)
    y_train = np.array(y_train)

    return X_train, y_train


class ChatDataset(Dataset):
    def __init__(self):
        self.n_samples = len(X_train)
        self.x_data = X_train
        self.y_data = y_train

    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        return self.n_samples


def count_pos(num_pos, predictions, labels):

    pred_labels = torch.argmax(predictions, dim=1)
    pred_labels = list(pred_labels)
    labels = list(labels)

    for (pred_label, label) in zip(pred_labels, labels):
        num_pos += (pred_label == label)

    return num_pos

def train_model(input_size, 
                hidden_size, 
                output_size, 
                lr_rate, 
                num_epochs, 
                train_loader):
    # loss and optimizer 
    model = NeuralNet(input_size, hidden_size, output_size)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr_rate)

    # save loss and accuracy 
    losses = []
    results = []

    for epoch in range(num_epochs):
        num_pos = 0
        for (words, labels) in train_loader:
            words = words.to(device)
            labels = labels.to(device)

            # forward
            outputs = model(words)
            loss = criterion(outputs, labels)

            # backward and optimizer step 
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # num_pos = count_pos(num_pos, outputs, labels)

        if (epoch + 1) % 100 == 0:
            # losses.append(loss.item())
            # accuracy = num_pos / X_train.shape[0]
            # results.append(accuracy)

            print(f'epoch {epoch + 1}/{num_epochs}, loss={loss.item():.4f}')
            # print(f'accuracy: {accuracy}')

    print(f'final loss, loss={loss.item():.4f}')

    data = {
        "model_state": model.state_dict(),
        "input_size": input_size,
        "output_size": output_size,
        "hidden_size": hidden_size,
        "all_words": all_words,
        "tags": tags
    }

    FILE = "data.pth"
    torch.save(data, FILE)

    print(f'training complete. File save to {FILE}')

# epoches = np.arange(0, num_epochs, 5)
# plt.plot(epoches, results)
# plt.xlabel('epoch')
# plt.ylabel('accuracy')
# plt.show()
    
if __name__ == '__main__':

    json_path = "./intents.json"
    tags, all_words, xy = fetch_from_json(json_path)
    X_train, y_train = prepare_training_data(tags, all_words, xy)

    # Hyperparameters 
    batch_size = 8
    hidden_size = 8
    output_size = len(tags)
    input_size = len(X_train[0])
    lr_rate = 0.001
    num_epochs = 1000

    dataset = ChatDataset()
    train_loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    train_model(input_size, 
                hidden_size, output_size, 
                lr_rate, num_epochs, 
                train_loader)


    





