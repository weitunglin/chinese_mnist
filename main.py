from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data.dataset import random_split
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchsummary import summary
import pandas as pd
from dataset import ChineseMNISTDataset

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.pool = nn.MaxPool2d(2, 2)
        self.conv1 = nn.Conv2d(1, 512, 5)
        self.conv2 = nn.Conv2d(512, 256, 5)
        self.conv3 = nn.Conv2d(256, 128, 5)

        self.fc1 = nn.Linear(2048, 512)
        self.fc2 = nn.Linear(512, 128)
        self.fc3 = nn.Linear(128, 15)
        self.softmax = nn.LogSoftmax(dim=0)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        x = self.softmax(x)
        # print(x.shape)
        return x

def train(model, criterion, optimizer, train_loader):
    labels = torch.tensor([]).detach()
    preds  = torch.tensor([]).detach()

    running_loss = 0.0

    model.train()
    for i, (images, codes) in enumerate(train_loader):
        # print(i)
        # print(len(images), images.shape, len(codes), codes.shape)
        # print(codes)

        optimizer.zero_grad()

        outputs = model(images)
        # print(outputs, codes)
        loss = criterion(outputs, codes)
        # print(loss)
        loss.backward()
        optimizer.step()

        labels = torch.cat((labels, codes))
        for item in outputs:
            preds = torch.cat((preds, torch.argmax(item).unsqueeze(-1)))

        running_loss += loss.item()
        if i % 20 == 19:
            print(f"loss: {running_loss/20:>.3f} [{i * len(images)}/{len(train_loader)*len(images)}]")
            running_loss = 0
    
    total_count = labels.size(0)
    correct_count = (labels == preds).sum().item()
    print(f'Training Accuracy: {correct_count/total_count}\n')

def test(model, test_loader):
    labels = torch.tensor([]).detach()
    preds  = torch.tensor([]).detach()

    model.eval()
    for i, (images, codes) in enumerate(test_loader):
        outputs = model(images)

        labels = torch.cat((labels, codes))
        for item in outputs:
            preds = torch.cat((preds, torch.argmax(item).unsqueeze(-1)))
    
    total_count = labels.size(0)
    correct_count = (labels == preds).sum().item()
    print(f'Testing Accuracy: {correct_count/total_count}\n')

def main():
    parser = argparse.ArgumentParser(description="Chinese MNIST simple CNN")
    parser.add_argument("--batch-size", type=int, default=64, metavar="N", help="input batch size for training (default: 64)")
    parser.add_argument("--test-batch-size", type=int, default=64, metavar="N", help="input batch size for testing (default: 64)")
    parser.add_argument("--epochs", type=int, default=3, metavar="N", help="number of epochs to train (default: 3)")
    parser.add_argument("--data-file-path", type=str, default="/Users/allen/ml/chinese_mnist/chinese_mnist.csv", help="path to the csv data file")
    parser.add_argument("--img-folder-path", type=str, default="/Users/allen/ml/chinese_mnist/img", help="path to the csv data file")

    args = parser.parse_args()
    print(f"args: {args}")

    data = pd.read_csv(args.data_file_path, low_memory=False)
    data["img_path"] = data.apply(lambda row: f"{args.img_folder_path}/input_{row.suite_id}_{row.sample_id}_{row.code}.jpg", axis=1)
    print(data.head())

    train_dataset = data.sample(frac=0.8, random_state=42)
    test_dataset = data.drop(train_dataset.index)

    transform = transforms.Compose([transforms.ToTensor()])

    train_loader = DataLoader(ChineseMNISTDataset(train_dataset, args.img_folder_path, transform), batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(ChineseMNISTDataset(test_dataset, args.img_folder_path, transform), batch_size=args.test_batch_size, shuffle=True)

    model = Net()
    print(summary(model, (1, 64, 64)))

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    for epoch in range(args.epochs):
        print(f"Epoch {epoch + 1}\n-----------------------")
        train(model, criterion, optimizer, train_loader)
        test(model, test_loader)

    MODEL_PATH = './chinese_mnist.pth'
    torch.save(model.state_dict(), MODEL_PATH)

if __name__ == "__main__":
    main()