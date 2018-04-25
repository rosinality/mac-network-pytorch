import sys
import pickle

from tqdm import tqdm
from torch.utils.data import DataLoader
from dataset import CLEVR, collate_data, transform
from model import RelationNetworks

batch_size = 64
n_epoch = 180

train_set = DataLoader(CLEVR(sys.argv[1], transform=transform),
                    batch_size=batch_size, num_workers=4)

for epoch in range(n_epoch):
    dataset = iter(train_set)
    pbar = tqdm(dataset)

    for image, question, q_len, answer in pbar:
        pass
