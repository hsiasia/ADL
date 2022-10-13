import json
import pickle
from argparse import ArgumentParser, Namespace
from pathlib import Path
from typing import Dict

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import SeqClsDataset
from model import SeqClassifier
from utils import Vocab, AverageMeter, ClsMetrics as Metrics

TRAIN = "train"
DEV = "eval"
SPLITS = [TRAIN, DEV]

def trainOneEpoch(args, model, train_loader, optimizer):
    model.train()
    am = AverageMeter()
    m = Metrics()

    bar = tqdm(train_loader)
    for i, batch in enumerate(bar):
        batch['text'] = batch['text'].to(args.device)
        batch['intent'] = batch['intent'].to(args.device)
        
        output_dict = model(batch)
        #
        bar.set_postfix(iter=i, loss=output_dict['loss'].item(), lr=optimizer.param_groups[0]['lr'])
        #
        am.update(output_dict['loss'], n=batch['intent'].size(0))
        m.update(batch['intent'].detach().cpu(), output_dict['pred_labels'].detach().cpu())

        # get loss
        loss = output_dict['loss']
        # clear gradients
        optimizer.zero_grad()
        # calulate new gradient
        loss.backward()
        #
        if args.grad_clip > 0.0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.grad_clip)
        # update parameters
        optimizer.step()

    m.cal()
    print('Train Loss: {:6.4f} Acc: {:6.4f}'.format(am.avg, m.accuracy))
    return am.avg, m.accuracy

@torch.no_grad() 
def valOneEpoch(args, model, val_loader):
    model.eval()
    am = AverageMeter()
    m = Metrics()

    for batch in val_loader:
        batch['text'] = batch['text'].to(args.device)
        batch['intent'] = batch['intent'].to(args.device)

        output_dict = model(batch)

        #
        am.update(output_dict['loss'], n = batch['intent'].size(0))
        m.update(batch['intent'].detach().cpu(), output_dict['pred_labels'].detach().cpu())
    
    m.cal()
    print('Val Loss: {:6.4f} Acc: {:6.4f}'.format(am.avg, m.accuracy))
    return am.avg, m.accuracy

def saveModel(model, ckp_dir, epoch):
    ckp_path = ckp_dir / '{}-model.pth'.format(epoch + 1)
    best_ckp_path = ckp_dir / 'best-model.pth'
    torch.save(model.state_dict(), ckp_path)
    torch.save(model.state_dict(), best_ckp_path)
    print('Saved model checkpoints into {}...'.format(ckp_path))
    print('Saved best model into {}...'.format(best_ckp_path))

def main(args):
    # utils.Vocab
    with open(args.cache_dir / "vocab.pkl", "rb") as f:
        vocab: Vocab = pickle.load(f)

    ckpt_dir = args.ckpt_dir
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # str
    # cache/intent/intent2idx.json
    intent_idx_path = args.cache_dir / "intent2idx.json"
    # dict[str, int]
    # {'oil_change_how': 0, 'restaurant_suggestion': 1, 'new_card': 2 ... 'repeat': 149}
    intent2idx: Dict[str, int] = json.loads(intent_idx_path.read_text())

    # dict[str, str]
    # {'train': PosixPath('data/intent/train.json'), 'eval': PosixPath('data/intent/eval.json')}
    data_paths = {split: args.data_dir / f"{split}.json" for split in SPLITS}
    # dict[str, list[dict[str, str]]]
    # {'train': [{'text': '', 'intent': '', 'id': ''} ... {'text': '', 'intent': '', 'id': 'train-14999'}], 
    #  'eval': [{'text': '', 'intent': '', 'id': ''} ... {'text': '', 'intent': '', 'id': 'train-14999'}]}
    data = {split: json.loads(path.read_text()) for split, path in data_paths.items()}
    
    # dict
    # {'train': <dataset.SeqClsDataset object at 0x7ff34ccdc250>, 'eval': <dataset.SeqClsDataset object at 0x7ff34ccdc1c0>}
    datasets: Dict[str, SeqClsDataset] = {
        split: SeqClsDataset(split_data, vocab, intent2idx, args.max_len)
        for split, split_data in data.items()
    }

    # TODO: crecate DataLoader for train / dev datasets
    dataloaders: Dict[str, DataLoader] ={
        split: DataLoader(split_dataset, args.batch_size, shuffle=True, collate_fn=split_dataset.collate_fn) 
        for split, split_dataset in datasets.items()
    }

    embeddings = torch.load(args.cache_dir / "embeddings.pt")
    # TODO: init model and move model to target device(cpu / gpu)
    model = SeqClassifier(embeddings, args.hidden_size, args.num_layers, args.dropout, args.bidirectional, datasets[TRAIN].num_classes).to(args.device)

    # TODO: init optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    best_acc = 0.0

    for epoch in range(args.num_epoch): # 50
        # TODO: Training loop - iterate over train dataloader and update model weights
        print("EPOCH: %d" % (epoch))
        train_loss, train_acc = trainOneEpoch(args, model, dataloaders[TRAIN], optimizer)
        # TODO: Evaluation loop - calculate accuracy and save model weights
        val_loss, val_acc = valOneEpoch(args, model, dataloaders[DEV])
        # pass

        if val_acc > best_acc:
            best_acc = val_acc
            saveModel(model, ckpt_dir, epoch)

    # TODO: Inference on test set


def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument(
        "--data_dir",
        type=Path,
        help="Directory to the dataset.",
        default="./data/intent/",
    )
    parser.add_argument(
        "--cache_dir",
        type=Path,
        help="Directory to the preprocessed caches.",
        default="./cache/intent/",
    )
    parser.add_argument(
        "--ckpt_dir",
        type=Path,
        help="Directory to save the model file.",
        default="./ckpt/intent/",
    )
    # # model file
    # parser.add_argument('--name', default='', type=str, help='name for saving model')

    # data
    parser.add_argument("--max_len", type=int, default=128)

    # model
    parser.add_argument("--hidden_size", type=int, default=256)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--bidirectional", type=bool, default=True)

    # optimizer
    parser.add_argument("--lr", type=float, default=1e-3)

    # clipping
    parser.add_argument('--grad_clip', default = 5., type=float, help='max gradient norm')

    # data loader
    parser.add_argument("--batch_size", type=int, default=128)

    # training
    parser.add_argument(
        "--device", type=torch.device, help="cpu, cuda, cuda:0, cuda:1", default="cpu"
    )
    parser.add_argument("--num_epoch", type=int, default=10)

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    args.ckpt_dir.mkdir(parents=True, exist_ok=True)
    main(args)