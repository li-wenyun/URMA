import h5py
import numpy as np
import torch
from torchvision import datasets, models, transforms
from functools import partial


transforms_train = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.RandomHorizontalFlip(), # data augmentation
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]) # normalization
])

transforms_test = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

def load_dataset(dataset,path,num_label):
    # if datasets == 'AID':
    #     num_label=30
    # elif datasets == 'NWPU':
    #     num_label=45
    def target_to_oh(target,num_label):
        one_hot = torch.eye(num_label)[target]
        return one_hot
    target_to_oh=partial(target_to_oh,num_label=num_label)
    dataset=datasets.ImageFolder(path, transform=transforms_test,target_transform=target_to_oh)
    return dataset


def split_dataset(dataset):
    train_size = int(0.65 * len(dataset))
    val_size = int(0.15 * len(dataset))
    test_size = len(dataset) - train_size - val_size
    train_set, val_set, database_set = torch.utils.data.random_split(dataset, [train_size, val_size, test_size])
    return train_set, val_set, database_set


def allocate_dataset(X, Y, L):
    train_images = torch.from_numpy(X['train'])
    train_texts = torch.from_numpy(Y['train'])
    train_labels = torch.from_numpy(L['train'])
    database_images = torch.from_numpy(X['retrieval'])
    database_texts = torch.from_numpy(Y['retrieval'])
    database_labels = torch.from_numpy(L['retrieval'])
    test_images = torch.from_numpy(X['query'])
    test_texts = torch.from_numpy(Y['query'])
    test_labels = torch.from_numpy(L['query'])
    return train_images, train_texts, train_labels, database_images, database_texts, database_labels, test_images, test_texts, test_labels



