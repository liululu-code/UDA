import torch
from tqdm import tqdm
from wilds import get_dataset
from wilds.common.data_loaders import get_train_loader
import torchvision.transforms as transforms
import os
from torchvision import models


def test_dataset(data_path):
    dataset = get_dataset(dataset="camelyon17",root_dir=data_path, download=False)
    # print(type(dataset))

    train_data = dataset.get_subset(
        "train",
        transform = transforms.Compose(
            [transforms.Resize((448, 448)), transforms.ToTensor()]
        ),
    )
    train_loader = get_train_loader("standard", train_data, batch_size=256, num_workers=4)

    net = models.resnet18()

    tmp = []
    for x, y, metadata in tqdm(train_loader):
        max_values, max_value_indexes = torch.max(metadata, dim=0)
        tmp.append(max_values)
    tmp = torch.cat(tmp, dim=0)
    max_values, _ = torch.max(tmp, dim=0)
    print(max_values)


        # print('hello')



def download_dataset(download_path):
    dataset = get_dataset(dataset="rxrx1", root_dir=download_path, download=True)




if __name__ == '__main__':
    data_path = './../data'
    download_path = './../'
    data_path = os.path.normpath(data_path)
    print(data_path)
    test_dataset(data_path)
    # download_dataset(download_path)
