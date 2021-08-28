import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data.dataset import random_split
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchsummary import summary
from torchvision.utils import make_grid
from torch.utils.tensorboard import SummaryWriter
from dataset import ChineseMNISTDataset
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

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
        # self.softmax = nn.LogSoftmax(dim=0)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        # x = self.softmax(x)
        # print(x.shape)
        return x

def train(model, criterion, optimizer, train_loader, epoch, tensor_board):
    labels = torch.tensor([]).detach()
    preds  = torch.tensor([]).detach()

    running_loss = 0.0

    model.train()
    for i, (images, codes) in enumerate(train_loader):
        optimizer.zero_grad()
        outputs = model(images)

        loss = criterion(outputs, codes)
        loss.backward()
        optimizer.step()

        labels = torch.cat((labels, codes))
        for item in outputs:
            preds = torch.cat((preds, torch.argmax(item).unsqueeze(-1)))

        running_loss += loss.item()
        if i % 20 == 19:
            print(f"Loss: {running_loss / 20:>.3f} [{(i + 1) * len(images)}/{len(train_loader) * len(images)}]")
            tensor_board.add_scalar(f"Epoch {epoch} Training Loss (by batch)", running_loss / 20, i)
            running_loss = 0
 
    total_count = labels.size(0)
    correct_count = (labels == preds).sum().item()
    tensor_board.add_scalar(f"Training Accuracy (by epoch)", correct_count / total_count, epoch)
    print(f"Training Accuracy: {correct_count / total_count}")

def test(model, test_loader, epoch, tensor_board):
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
    tensor_board.add_scalar(f"Validation Accuracy (by epoch)", correct_count / total_count, epoch)
    print(f"Validation Accuracy: {correct_count / total_count}")

def main():
    parser = argparse.ArgumentParser(description="Chinese MNIST simple CNN")
    parser.add_argument("--batch-size", type=int, default=64, metavar="N", help="input batch size for training (default: 64)")
    parser.add_argument("--test-batch-size", type=int, default=64, metavar="N", help="input batch size for testing (default: 64)")
    parser.add_argument("--epochs", type=int, default=3, metavar="N", help="number of epochs to train (default: 3)")
    parser.add_argument("--data-file-path", type=str, default="/Users/allen/ml/chinese_mnist/chinese_mnist.csv", help="path to the csv data file")
    parser.add_argument("--img-folder-path", type=str, default="/Users/allen/ml/chinese_mnist/img", help="path to the csv data file")
    parser.add_argument("--run-name", type=str, default=datetime.now().strftime("%Y/%m/%d %H:%M:%S"), help="run id for tensor board")
    parser.add_argument("--random-seed", type=int, default=42, help="random seed")

    args = parser.parse_args()
    print("----- args -----")
    print(args)

    data = pd.read_csv(args.data_file_path, low_memory=False)
    data["img_path"] = data.apply(lambda row: f"{args.img_folder_path}/input_{row.suite_id}_{row.sample_id}_{row.code}.jpg", axis=1)
    print("----- dataset (head & describe) -----")
    print(data.head())
    print(data.describe())

    train_dataset = data.sample(frac=0.8, random_state=args.random_seed)
    test_dataset = data.drop(train_dataset.index)

    transform = transforms.Compose([
        transforms.RandomRotation(10),
        transforms.ToTensor()])

    train_loader = DataLoader(ChineseMNISTDataset(train_dataset, args.img_folder_path, transform), batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(ChineseMNISTDataset(test_dataset, args.img_folder_path, transform), batch_size=args.test_batch_size, shuffle=True)

    model = Net()
    print(summary(model, (1, 64, 64)))

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    tensor_board = SummaryWriter(f"runs/{args.run_name}")
    sample_images = iter(test_loader).next()[0]
    tensor_board.add_image("chinese mnist", make_grid(sample_images))
    tensor_board.add_graph(model, sample_images)

    for epoch in range(1, args.epochs + 1):
        print(f"----- Epoch {epoch} -----")
        train(model, criterion, optimizer, train_loader, epoch, tensor_board)
        test(model, test_loader, epoch, tensor_board)
        print(f"----- End of Epoch {epoch} -----")

    model.eval()
    dataset = DataLoader(ChineseMNISTDataset(data, args.img_folder_path, transform), batch_size=args.batch_size) 

    classes = ("零", "一", "二", "三", "四", "五", "六", "七", "八", "九", "百", "千", "萬", "億")
    class_probs = []
    class_label = []
    with torch.no_grad():
        for data in dataset:
            images, labels = data
            output = model(images)
            class_probs_batch = [F.softmax(el, dim=0) for el in output]

            class_probs.append(class_probs_batch)
            class_label.append(labels)

    test_probs = torch.cat([torch.stack(batch) for batch in class_probs])
    test_label = torch.cat(class_label)

    def add_pr_curve_tensorboard(class_index, test_probs, test_label, global_step=0):
        tensorboard_truth = test_label == class_index
        tensorboard_probs = test_probs[:, class_index]

        tensor_board.add_pr_curve(classes[class_index],
                            tensorboard_truth,
                            tensorboard_probs,
                            global_step=global_step)

    for i in range(len(classes)):
        add_pr_curve_tensorboard(i, test_probs, test_label)

    test_preds = np.array([], dtype=int)
    for batch in class_probs:
        for i in batch: 
            test_preds = np.append(test_preds, torch.argmax(i).unsqueeze(-1))
    df_confusion = pd.crosstab(test_label, test_preds, rownames=["Actual"], colnames=["Predicted"], margins=True)
    print("----- Confusion Matrix -----")
    print(df_confusion)

    torch.save(model.state_dict(), f"result/{args.run_name}.pth")
    tensor_board.close()

if __name__ == "__main__":
    main()
