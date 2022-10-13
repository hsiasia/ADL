import json
import pickle
from argparse import ArgumentParser, Namespace
from pathlib import Path
from typing import Dict

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import SeqTaggingClsDataset
from model import SeqTagger
from utils import Vocab, AverageMeter, TagMetrics as Metrics

TRAIN = "train"
DEV = "eval"
SPLITS = [TRAIN, DEV]

def trainOneEpoch(args, model, train_loader, optimizer):
    model.train()
    am = AverageMeter()
    m = Metrics()

    bar = tqdm(train_loader)
    for i, batch in enumerate(bar):
        batch['tokens'] = batch['tokens'].to(args.device)
        batch['tags'] = batch['tags'].to(args.device)
        batch['mask'] = batch['mask'].to(args.device)

        output_dict = model(batch)
        #
        bar.set_postfix(iter=i, loss=output_dict['loss'].item(), lr=optimizer.param_groups[0]['lr'])
        #
        am.update(output_dict['loss'], n=batch['tokens'].size(0))
        m.update(batch['tags'].cpu(), output_dict['pred_labels'].cpu(), batch['mask'].cpu())

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
    print('Train Loss: {:6.4f} Joint Acc: {:6.4f} ({}/{}) Token Acc: {:6.4f} ({}/{})'.format(am.avg, m.jointAccuracy, m.jointCorrect, m.jointCount, m.tokenAccuracy, m.tokenCorrect, m.tokenCount))
    return am.avg, m.jointAccuracy, m.tokenAccuracy

@torch.no_grad() 
def valOneEpoch(args, model, val_loader):
    model.eval()
    am = AverageMeter()
    m = Metrics()

    for batch in val_loader:
        batch['tokens'] = batch['tokens'].to(args.device)
        batch['tags'] = batch['tags'].to(args.device)
        batch['mask'] = batch['mask'].to(args.device)

        output_dict = model(batch)
        
        #
        am.update(output_dict['loss'], n = batch['tokens'].size(0))
        m.update(batch['tags'].cpu(), output_dict['pred_labels'].cpu(), batch['mask'].cpu())
    
    m.cal()
    print('Val Loss: {:6.4f} Joint Acc: {:6.4f} ({}/{}) Token Acc: {:6.4f} ({}/{})'.format(am.avg, m.jointAccuracy, m.jointCorrect, m.jointCount, m.tokenAccuracy, m.tokenCorrect, m.tokenCount))
    return am.avg, m.jointAccuracy, m.tokenAccuracy

def saveModel(model, ckp_dir, epoch):
    ckp_path = ckp_dir / '{}-model.pth'.format(epoch + 1)
    best_ckp_path = ckp_dir / 'best-model.pth'
    torch.save(model.state_dict(), ckp_path)
    torch.save(model.state_dict(), best_ckp_path)
    print('Saved model checkpoints into {}...'.format(ckp_path))
    print('Saved best model into {}...'.format(best_ckp_path))

def main(args):
    # TODO: implement main function
    # utils.Vocab
    with open(args.cache_dir / "vocab.pkl", "rb") as f:
        vocab: Vocab = pickle.load(f)

    ckpt_dir = args.ckpt_dir
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # str
    tag_idx_path = args.cache_dir / "tag2idx.json"
    # dict[str, int]
    tag2idx: Dict[str, int] = json.loads(tag_idx_path.read_text())

    # dict[str, str]
    data_paths = {split: args.data_dir / f"{split}.json" for split in SPLITS}
    # dict[str, list[dict[str, list[str]]]]
    data = {split: json.loads(path.read_text()) for split, path in data_paths.items()}

    # dict
    datasets: Dict[str, SeqTaggingClsDataset] = {
        split: SeqTaggingClsDataset(split_data, vocab, tag2idx, args.max_len)
        for split, split_data in data.items()
    }

    # TODO: crecate DataLoader for train / dev datasets
    dataloaders: Dict[str, DataLoader] ={
        split: DataLoader(split_dataset, args.batch_size, shuffle=True, collate_fn=split_dataset.collate_fn) 
        for split, split_dataset in datasets.items()
    }

    embeddings = torch.load(args.cache_dir / "embeddings.pt")
    # TODO: init model and move model to target device(cpu / gpu)
    model = SeqTagger(embeddings, args.hidden_size, args.num_layers, args.dropout, args.bidirectional, datasets[TRAIN].num_classes).to(args.device)

    # TODO: init optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    best_acc = 0.0

    # epoch_pbar = trange(args.num_epoch, desc="Epoch")
    for epoch in range(args.num_epoch):
        # TODO: Training loop - iterate over train dataloader and update model weights
        print("EPOCH: %d" % (epoch))
        train_loss, train_joi_acc, train_tok_acc = trainOneEpoch(args, model, dataloaders[TRAIN], optimizer)
        # TODO: Evaluation loop - calculate accuracy and save model weights
        val_loss, val_joi_acc, val_tok_acc = valOneEpoch(args, model, dataloaders[DEV])
        # pass

        if val_joi_acc > best_acc:
            best_acc = val_joi_acc
            saveModel(model, ckpt_dir, epoch)

    # TODO: Inference on test set


def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument(
        "--data_dir",
        type=Path,
        help="Directory to the dataset.",
        default="./data/slot/",
    )
    parser.add_argument(
        "--cache_dir",
        type=Path,
        help="Directory to the preprocessed caches.",
        default="./cache/slot/",
    )
    parser.add_argument(
        "--ckpt_dir",
        type=Path,
        help="Directory to save the model file.",
        default="./ckpt/slot/",
    )
    # # model file
    # parser.add_argument('--name', default='', type=str, help='name for saving model')

    # data
    parser.add_argument("--max_len", type=int, default=128)

    # model
    parser.add_argument("--hidden_size", type=int, default=512)
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
    parser.add_argument("--num_epoch", type=int, default=50)

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    args.ckpt_dir.mkdir(parents=True, exist_ok=True)
    main(args)