from typing import Iterable, List


class Vocab:
    PAD = "[PAD]"
    UNK = "[UNK]"

    def __init__(self, vocab: Iterable[str]) -> None:
        self.token2idx = {
            Vocab.PAD: 0,
            Vocab.UNK: 1,
            **{token: i for i, token in enumerate(vocab, 2)},
        }

    @property
    def pad_id(self) -> int:
        return self.token2idx[Vocab.PAD]

    @property
    def unk_id(self) -> int:
        return self.token2idx[Vocab.UNK]

    @property
    def tokens(self) -> List[str]:
        return list(self.token2idx.keys())

    def token_to_id(self, token: str) -> int:
        return self.token2idx.get(token, self.unk_id)

    def encode(self, tokens: List[str]) -> List[int]:
        return [self.token_to_id(token) for token in tokens]

    def encode_batch(
        self, batch_tokens: List[List[str]], to_len: int = None
    ) -> List[List[int]]:
        batch_ids = [self.encode(tokens) for tokens in batch_tokens]
        to_len = max(len(ids) for ids in batch_ids) if to_len is None else to_len
        padded_ids = pad_to_len(batch_ids, to_len, self.pad_id)
        return padded_ids


def pad_to_len(seqs: List[List[int]], to_len: int, padding: int) -> List[List[int]]:
    paddeds = [seq[:to_len] + [padding] * max(0, to_len - len(seq)) for seq in seqs]
    return paddeds


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class ClsMetrics(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.correct = 0
        self.count = 0

    def update(self, target, pred):
        self.correct += pred.eq(target.view_as(pred)).sum().item()
        self.count += target.size(0)
    
    def cal(self):
        self.accuracy = self.correct / self.count


class TagMetrics(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.tokenCorrect = 0
        self.jointCorrect = 0
        self.tokenCount = 0
        self.jointCount = 0

    def update(self, target, pred, mask):
        mask = mask[:, :target.size(1)]
        batch_cor = (target.eq(pred.view_as(target)) * mask).sum(-1)
        seq_len = mask.sum(-1)
        
        self.tokenCorrect += batch_cor.sum().long().item()
        self.jointCorrect += batch_cor.eq(seq_len).sum().item()
        self.tokenCount += mask.sum().long().item()
        self.jointCount += len(target)
    
    def cal(self):
        self.tokenAccuracy = self.tokenCorrect / self.tokenCount
        self.jointAccuracy = self.jointCorrect / self.jointCount