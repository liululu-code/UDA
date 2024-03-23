import os
import sys

import torch
from torch import nn
from tqdm import tqdm
from wilds import get_dataset
from wilds.common.data_loaders import get_train_loader
import torchvision.transforms as transforms
from torchvision import models
import matplotlib.pyplot as plt

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'utils'))
from model_name import get_num, save_num


def train(data_path):
    # -------------超参数设置---------------
    lamd = 0.1
    epoch = 10
    batch_size = 128
    experiment_num = 4
    # -------------------------------------

    dataset = get_dataset(dataset="camelyon17", root_dir=data_path, download=True)
    # print(type(dataset))

    total_experiment_accuracy = 0
    avg_experiment_accuracy = 0
    validation_accuracy = 0
    for n in range(experiment_num):
        train_data_a = dataset.get_subset(
            "train",
            transform=transforms.Compose(
                [
                    # transforms.Resize((448, 448)),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=(.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
                ]
            ),
        )
        train_loader_a = get_train_loader("standard", train_data_a, batch_size=batch_size)

        val_data_a = dataset.get_subset(
            "val",
            transform=transforms.Compose(
                [
                    # transforms.Resize((448, 448)),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=(.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
                ]
            ),
        )
        val_loader_a = get_train_loader("standard", val_data_a, batch_size=batch_size)

        test_data_a = dataset.get_subset(
            "test",
            transform=transforms.Compose(
                [
                    # transforms.Resize((448, 448)),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=(.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
                ]
            ),
        )
        test_loader_a = get_train_loader("standard", test_data_a, batch_size=batch_size)

        net = models.resnet50(num_classes=2, norm_layer=nn.InstanceNorm2d)
        optimizer = torch.optim.SGD(net.parameters(), lr=3e-3, momentum=0.9, weight_decay=1e-3)
        torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.99)
        loss_fn = nn.CrossEntropyLoss()



        device = ('cuda' if torch.cuda.is_available() else 'cpu')
        print(f'Using device: {device}')
        net.to(device)

        accuracys = []
        max_accuracy = 0
        model_num = get_num()

        for epoch in range(epoch):
            # train
            net.train()
            for x, y, metadata in tqdm(train_loader_a, desc=f'epoch {epoch} train:'):
                x, y = x.to(device), y.to(device)
                optimizer.zero_grad()

                outputs = net(x)
                loss = loss_fn(outputs, y)

                loss.backward()
                optimizer.step()

            # validation
            net.eval()
            total_exact_num = 0
            total_num = 0

            with torch.no_grad():
                pbar = tqdm(val_loader_a, desc=f'epoch {epoch} validation:')
                for x, y, metadata in pbar:
                    x, y = x.to(device), y.to(device)
                    outputs = net(x)
                    pred_y = torch.argmax(outputs, dim=1, keepdim=True).squeeze(1)
                    exact_num = (y == pred_y).sum().item()
                    acc_tmp = exact_num / len(y)
                    pbar.set_postfix(acc=acc_tmp, refresh=False)
                    total_exact_num += exact_num
                    total_num += len(y)
            accuracy = total_exact_num / total_num
            print(f'epoch {epoch}, validation accuracy: {accuracy}')
            accuracys.append(accuracy)

            if accuracy > max_accuracy:
                max_accuracy = accuracy
                torch.save(net.state_dict(), f'../models/model{model_num}_acc{max_accuracy}.pt')
                print(f'save model num: {model_num}')
            print('---------------------------------------------------------------------------')

        print(f'epoch {epoch} : validation_accuracy={max_accuracy}')
        fig, ax = plt.subplots(figsize=(10, 8))
        ax.plot(range(len(accuracys)), accuracys, label='validation')
        ax.set_xlabel('epoch')
        ax.set_ylabel('accuracy')
        ax.set_title(f'validation accuracy epoch')
        plt.legend()
        fig.show()

        # test集上进行测试
        net.load_state_dict(torch.load(f'../models/model{model_num}_acc{max_accuracy}.pt'))
        net.eval()
        total_exact_num = 0
        total_num = 0

        with torch.no_grad():
            pbar = tqdm(test_loader_a, desc='test:')
            for x, y, metadata in pbar:
                x, y = x.to(device), y.to(device)
                outputs = net(x)
                pred_y = torch.argmax(outputs, dim=1, keepdim=True).squeeze(1)
                exact_num = (y == pred_y).sum().item()
                acc_tmp = exact_num / len(y)
                pbar.set_postfix(acc=acc_tmp, refresh=False)
                total_exact_num += exact_num
                total_num += len(y)
        accuracy = total_exact_num / total_num
        print(f'epoch {epoch} : test accuracy: {accuracy}')
        print('---------------------------------------------------------------------------')
        total_experiment_accuracy += accuracy
    avg_accuracy = total_experiment_accuracy / experiment_num
    print('average test accuracy: ', avg_accuracy)


if __name__ == '__main__':
    data_path = os.path.normpath(r"F:\Datasets\WILDS")

    train(data_path)
