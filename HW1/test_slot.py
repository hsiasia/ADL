import json
import pickle
from argparse import ArgumentParser, Namespace
from pathlib import Path
from typing import Dict

import torch
from torch.utils.data import DataLoader

from dataset import SeqTaggingClsDataset
from model import SeqTagger
from utils import Vocab


def main(args):
    # TODO: implement main function
    with open(args.cache_dir / "vocab.pkl", "rb") as f:
        vocab: Vocab = pickle.load(f)

    tag_idx_path = args.cache_dir / "tag2idx.json"
    tag2idx: Dict[str, int] = json.loads(tag_idx_path.read_text())

    data = json.loads(args.test_file.read_text())
    dataset = SeqTaggingClsDataset(data, vocab, tag2idx, args.max_len)

    test_loader = DataLoader(dataset, args.batch_size, shuffle=False, collate_fn=dataset.collate_fn)

    embeddings = torch.load(args.cache_dir / "embeddings.pt")

    model = SeqTagger(embeddings, args.hidden_size, args.num_layers, args.dropout, args.bidirectional, dataset.num_classes).to(args.device)
    model.eval()

    ckpt = torch.load(args.ckpt_path)
    model.load_state_dict(ckpt)

    all_ids = []
    all_tags = []
    all_lens = []

    # TODO: predict dataset
    for batch in test_loader:
        batch['tokens'] = batch['tokens'].to(args.device)
        batch['tags'] = batch['tags'].to(args.device)
        batch['mask'] = batch['mask'].to(args.device)

        with torch.no_grad():
            output_dict = model(batch)

        #
        all_ids += batch['id']
        all_tags += output_dict['pred_labels'].cpu().tolist()
        all_lens += batch['mask'].sum(-1).long().cpu().tolist()

    # write prediction to file
    if args.pred_file.parent:
        args.pred_file.parent.mkdir(parents=True, exist_ok=True)

    with open(args.pred_file, 'w') as f:
        f.write('id,tags\n')
        for i, tags, seq_len in zip(all_ids, all_tags, all_lens):
            f.write("%s," % (i))
            for idx, tag in enumerate(tags):
                if idx < seq_len - 1:
                    f.write("%s " % (dataset.idx2label(tag)))
                else:
                    f.write("%s\n" % (dataset.idx2label(tag)))
                    break


def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument(
        "--test_file",
        type=Path,
        help="Path to the test file.",
        default="./data/slot/test.json"
    )
    parser.add_argument(
        "--cache_dir",
        type=Path,
        help="Directory to the preprocessed caches.",
        default="./cache/slot/",
    )
    parser.add_argument(
        "--ckpt_path",
        type=Path,
        help="Path to model checkpoint.",
        default="./ckpt/slot/best-model.pth",
        required=True
    )
    parser.add_argument("--pred_file", type=Path, default="pred.slot.csv")

    # data
    parser.add_argument("--max_len", type=int, default=128)

    # model
    parser.add_argument("--hidden_size", type=int, default=512)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--bidirectional", type=bool, default=True)

    # data loader
    parser.add_argument("--batch_size", type=int, default=128)

    parser.add_argument(
        "--device", type=torch.device, help="cpu, cuda, cuda:0, cuda:1", default="cpu"
    )
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    main(args)