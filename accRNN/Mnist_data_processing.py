import math
import torch
from torchvision import datasets, transforms
from torch.utils.data.sampler import SubsetRandomSampler
from fastNLP import DataSet
from fastNLP import Instance


def get_fastnlp_dataset():
    # Hyper parameters
    output_dim = 10
    SEQUENCE_LENGTH = 28
    mnist_train_length = 60000
    validation_samples = 5000
    BATCH_SIZE = 1
    transform_train_test = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    indices = torch.randperm(mnist_train_length)
    train_indices = indices[:len(indices) - validation_samples]
    val_indices = indices[len(indices) - validation_samples:]

    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST('./data', train=True, download=True,
                       transform=transform_train_test),
        sampler=SubsetRandomSampler(train_indices),
        batch_size=BATCH_SIZE, shuffle=False)
    val_loader = torch.utils.data.DataLoader(
        datasets.MNIST('./data', train=True, download=False,
                       transform=transform_train_test),
        sampler=SubsetRandomSampler(val_indices),
        batch_size=BATCH_SIZE, shuffle=False)

    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST('./data', train=False, transform=transform_train_test),
        batch_size=BATCH_SIZE, shuffle=True)

    TRAIN_ITERS = int(math.ceil((mnist_train_length - validation_samples) / BATCH_SIZE))
    VAL_ITERS = int(math.ceil(validation_samples / BATCH_SIZE))
    TEST_ITERS = int(math.ceil(len(test_loader.dataset) / BATCH_SIZE))

    train_data = DataSet()
    val_data = DataSet()
    test_data = DataSet()
    for batch_idx, (x, y) in enumerate(train_loader):
        x = x[0][0].numpy()
        train_data.append(Instance(word_seq=x, target=int(y)))
    for batch_idx, (x, y) in enumerate(val_loader):
        x = x[0][0].numpy()
        val_data.append(Instance(word_seq=x, target=int(y)))
    for batch_idx, (x, y) in enumerate(test_loader):
        x = x[0][0].numpy()
        test_data.append(Instance(word_seq=x, target=int(y)))

    # 设定特征域和标签域
    train_data.set_input("word_seq")
    test_data.set_input("word_seq")
    val_data.set_input("word_seq")
    train_data.set_target("target")
    test_data.set_target("target")
    val_data.set_target("target")

    return train_data, val_data, test_data


def load_model(model, model_path):
    states = torch.load(model_path).state_dict()
    model.load_state_dict(states)
    return model
