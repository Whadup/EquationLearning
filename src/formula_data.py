from PIL import Image
from tqdm import tqdm
import argparse
import img_trans
import numpy as np
import random
import torch
import torch.nn
import torchvision
import torchvision.transforms as transforms


class TripleData(torch.utils.data.Dataset):
    """ This class is a torch dataset, which gives a pair of real arxiv formulas.
        If the label is 1 the two formulas occur in the same paper.
    """

    # isProcessed determines if the Dataset was already built. Then it will be loaded from a file.
    def __init__(self, name, root="", is_processed=True, pairs_per_sample=1, one_per_class=False):
        self.SIM_LABEL = 1
        self.DISSIM_LABEL = -1
        self.labels = [self.DISSIM_LABEL, self.SIM_LABEL]
        self.pairs_per_sample = pairs_per_sample
        self.one_per_class = one_per_class

        self.store_file = "index_label_tensor_{}.pt".format(name)

        # Get data from directory structure
        self.dataset = SingleData(name, root, is_processed=is_processed)

        # if the pairs were already created and saved they only need to be loaded
        if False and is_processed:
            self.index_label_tensor = torch.load(self.store_file)
        # otherwise we have to create a new dataset with pairs
        else:
            # Extract labels and data from the trainset
            labels = [self.dataset[i][1] for i in range(len(self.dataset))]
            labels = np.array(labels)
            self.labels = labels

            # Gets a list with arrays.
            # Each array stores the indices of all places, where you can find a specific class in the train data
            digit_indices = \
                [np.where(labels == i)[0]
                 for i in range(int(self.dataset.min_class().item()), 1 + int(self.dataset.max_class().item()))]
            self.digit_indices = digit_indices
            self.num_classes = len(digit_indices)
            # self.index_label_tensor = self.create_pairs(digit_indices)
            # torch.save(self.index_label_tensor, self.store_file)

    def __len__(self):
        return int(len(self.dataset))

    def __getitem__(self, index):
        x = index
        # print(index,x,sim)
        d = int(self.labels[x])
        # print(x,d,len(self.digit_indices))
        n = len(self.digit_indices[d])
        if n < 2:
            x2 = x
        else:
            inc = random.randrange(0, n)
            while self.digit_indices[d][inc] == x:
                inc = random.randrange(0, n)
            x2 = self.digit_indices[d][inc]
        # rejection sampling to sample from all other example classes uniformly
        x3 = random.randrange(0, len(self))
        d3 = int(self.labels[x3])
        while d3 == d:
            x3 = random.randrange(0, len(self))
            d3 = int(self.labels[x3])
        return self.dataset[x][0], self.dataset[x2][0], self.dataset[x3][0]


class PairData(torch.utils.data.Dataset):
    """ This class is a torch dataset, which gives a pair of real arxiv formulas.
        If the label is 1 the two formulas occur in the same paper.
    """

    # isProcessed determines if the Dataset was already built. Then it will be loaded from a file.
    def __init__(self, name, root="", is_processed=True, pairs_per_sample=1, one_per_class=False, with_dot_product=True):
        self.SIM_LABEL = 1
        if with_dot_product:
            self.DISSIM_LABEL = -1
        else:
            self.DISSIM_LABEL = 0

        # Get data from directory structure
        self.dataset = SingleData(name, root, is_processed=is_processed)

        # otherwise we have to create a new dataset with pairs
        # Extract labels and data from the trainset
        labels = [self.dataset[i][1] for i in range(len(self.dataset))]
        labels = np.array(labels)
        self.labels = labels

        # Gets a list with arrays.
        # Each array stores the indices of all places, where you can find a specific class in the train data
        digit_indices = \
            [np.where(labels == i)[0]
             for i in range(int(self.dataset.min_class().item()), 1 + int(self.dataset.max_class().item()))]
        print("min class", self.dataset.min_class())
        self.digit_indices = digit_indices
        self.num_classes = len(digit_indices)

    def __len__(self):
        return int(len(self.dataset))

    def __getitem__(self, index):

        sim = np.random.choice(2, 1, p=(0.5, 0.5))
        x = index
        # print(index,x,sim)
        d = int(self.labels[x])
        # print(x,d,len(self.digit_indices))
        n = len(self.digit_indices[d])
        if sim == 0:
            if n < 2:
                # print("shit",x)
                return self.dataset[x][0], self.dataset[x][0], self.SIM_LABEL
            inc = random.randrange(0, n)
            while self.digit_indices[d][inc] == x:
                inc = random.randrange(0, n)
            x2 = self.digit_indices[d][inc]
            # print(x,x2,d,n,self.SIM_LABEL)
            assert self.dataset[x][1] == self.dataset[x2][1]
            return self.dataset[x][0], self.dataset[x2][0], self.SIM_LABEL
        else:
            x2 = random.randrange(0, len(self))
            d2 = int(self.labels[x2])
            while d2 == d:
                x2 = random.randrange(0, len(self))
                d2 = int(self.labels[x2])
            # print(x,x2,d,dn,n,len(self.digit_indices[dn]), self.DISSIM_LABEL)
            # print(x,x2,self.DISSIM_LABEL)
            return self.dataset[x][0], self.dataset[x2][0], self.DISSIM_LABEL


class SingleData(torch.utils.data.Dataset):
    """  This class is a torch dataset, which gives single real arxiv formulas.
    """

    def __init__(self, name, root="", is_processed=True):
        self.img_file = "arxiv_img_tensor_{}_grey.pt".format(name)
        self.label_file = "arxiv_label_tensor_{}_grey.pt".format(name)

        if is_processed:
            self.imgs = torch.load(self.img_file)
            self.labels = torch.load(self.label_file)
        else:
            transform = transforms.Compose(
                [img_trans.convert_to_la,
                 transforms.CenterCrop((32, 333)),
                 transforms.ToTensor(),
                 img_trans.take_alpha_channel
                 ])
            # Get data from directory structure
            self.dataset = torchvision.datasets.ImageFolder(
                root=root, transform=transform, loader=img_loader)
            self.imgs = torch.ones((len(self.dataset), 1, 32, 333))
            self.labels = torch.ones(len(self.dataset))
            # print(enumerate(self.dataset))
            for i in tqdm(range(len(self.dataset))):
                img, label = self.dataset[i]
                self.imgs[i] = img
                self.labels[i] = label

            torch.save(self.imgs, self.img_file)
            torch.save(self.labels, self.label_file)

    def __len__(self):
        return self.labels.size()[0]

    def __getitem__(self, item):
        return self.imgs[item], self.labels[item]

    def num_classes(self):
        return torch.max(self.labels)

    def min_class(self):
        return self.labels[0]

    def max_class(self):
        return self.labels[len(self.labels) - 1]


def img_loader(path):
    return Image.open(path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Crawl the pngs from arxivs.')
    parser.add_argument('name', metavar='N', type=str,
                        help='containing folder of the arxiv folder')
    parser.add_argument('root', metavar='N', type=str,
                        help='containing folder of the arxiv folder')
    args = parser.parse_args()
    d = SingleData(name=args.name, root=args.root, is_processed=False)
