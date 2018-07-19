import sys
import pickle
from collections import Counter

import torch
from tqdm import tqdm
from torch.utils.data import DataLoader
from dataset import CLEVR, collate_data, transform

batch_size = 64
n_epoch = 180

train_set = DataLoader(
    CLEVR(sys.argv[1], 'val', transform=None),
    batch_size=batch_size,
    num_workers=4,
    collate_fn=collate_data,
)
net = torch.load(sys.argv[2])
net.eval()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

for epoch in range(n_epoch):
    dataset = iter(train_set)
    pbar = tqdm(dataset)
    family_correct = Counter()
    family_total = Counter()

    for image, question, q_len, answer, family in pbar:
        image, question = image.to(device), question.to(device)
        output = net(image, question, q_len)
        correct = output.detach().argmax(1) == answer.to(device)
        for c, fam in zip(correct, family):
            if c:
                family_correct[fam] += 1
            family_total[fam] += 1

    print(
        'Avg Acc: {:.5f}'.format(
            sum(family_correct.values()) / sum(family_total.values())
        )
    )

