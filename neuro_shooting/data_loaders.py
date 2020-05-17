import torchvision.datasets as datasets
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

def get_data_loaders(dataset, data_aug=False, batch_size=128, test_batch_size=1000, num_workers=0):

    #supported_datasets = datasets.__all__ # e.g., ['MNIST', 'FashionMNIST', 'CIFAR10', 'CIFAR100']
    supported_datasets = ['MNIST', 'CIFAR10', 'CIFAR100']

    nr_of_labels_per_dataset = {'MNIST': 10,
                                'CIFAR10': 10,
                                'CIFAR100': 100}

    if dataset not in supported_datasets:
        raise ValueError('Currently supported datasets {}'.format(supported_datasets))

    if data_aug:
        transform_train = transforms.Compose([
            transforms.RandomCrop(28, padding=4),
            transforms.ToTensor(),
        ])
    else:
        transform_train = transforms.Compose([
            transforms.ToTensor(),
        ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
    ])

    data_root = '.data/{}'.format(dataset)
    current_dataset = getattr(datasets,dataset)

    nr_of_classes = nr_of_labels_per_dataset[dataset]
    #class_labels = current_dataset.classes

    train_loader = DataLoader(
        current_dataset(root=data_root, train=True, download=True, transform=transform_train), batch_size=batch_size,
        shuffle=True, num_workers=num_workers, drop_last=True
    )

    train_eval_loader = DataLoader(
        current_dataset(root=data_root, train=True, download=True, transform=transform_test),
        batch_size=test_batch_size, shuffle=False, num_workers=num_workers, drop_last=True
    )

    test_loader = DataLoader(
        current_dataset(root=data_root, train=False, download=True, transform=transform_test),
        batch_size=test_batch_size, shuffle=False, num_workers=num_workers, drop_last=True
    )

    return train_loader, test_loader, train_eval_loader, nr_of_classes

