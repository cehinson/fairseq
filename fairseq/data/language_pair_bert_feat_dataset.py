
import torch

from . import data_utils

from . import LanguagePairDataset

from ..modules import BertFeatEmbed


def collate(
    samples, pad_idx, eos_idx, left_pad_source=True, left_pad_target=False,
    input_feeding=True,
):
    if len(samples) == 0:
        return {}

    def my_stack(list_of_tensors, pad_idx):
        max_len = max([t.size(0) for t in list_of_tensors])
        result = torch.zeros(len(list_of_tensors), max_len, 4, 768)
        result.fill_(pad_idx)
        for i, t in enumerate(list_of_tensors):
            result[i] = t
        return result

    def merge(key, left_pad, move_eos_to_beginning=False):
        return data_utils.collate_tokens(
            [s[key] for s in samples],
            pad_idx, eos_idx, left_pad, move_eos_to_beginning,
        )

    id = torch.LongTensor([s['id'] for s in samples])
    src_tokens = merge('source', left_pad=left_pad_source)
    # sort by descending source length
    src_lengths = torch.LongTensor([s['source'].numel() for s in samples])
    src_lengths, sort_order = src_lengths.sort(descending=True)
    id = id.index_select(0, sort_order)
    src_tokens = src_tokens.index_select(0, sort_order)

    prev_output_tokens = None
    target = None
    if samples[0].get('target', None) is not None:
        target = merge('target', left_pad=left_pad_target)
        target = target.index_select(0, sort_order)
        ntokens = sum(len(s['target']) for s in samples)

        if input_feeding:
            # we create a shifted version of targets for feeding the
            # previous output token(s) into the next decoder step
            prev_output_tokens = merge(
                'target',
                left_pad=left_pad_target,
                move_eos_to_beginning=True,
            )
            prev_output_tokens = prev_output_tokens.index_select(0, sort_order)
    else:
        ntokens = sum(len(s['source']) for s in samples)

    bert_src = [s['bert_src'] for s in samples]
    bert_tgt = [s['bert_tgt'] for s in samples]

    # bert_src = torch.stack(bert_src)
    # bert_tgt = torch.stack(bert_tgt)
    bert_src = my_stack(bert_src, pad_idx)
    bert_tgt = my_stack(bert_tgt, pad_idx)
    batch = {
        'id': id,
        'nsentences': len(samples),
        'ntokens': ntokens,
        'net_input': {
            'src_tokens': src_tokens,
            'src_lengths': src_lengths,
            'bert_src': bert_src,
            'bert_tgt': bert_tgt
        },
        'target': target,
    }
    if prev_output_tokens is not None:
        batch['net_input']['prev_output_tokens'] = prev_output_tokens
    return batch


class LangPairBertFeatDataset(LanguagePairDataset):
    def __init__(
        self, src, src_sizes, src_dict,
        tgt=None, tgt_sizes=None, tgt_dict=None,
        bert_src=None, bert_src_sizes=None, bert_src_dict=None,
        bert_tgt=None, bert_tgt_sizes=None, bert_tgt_dict=None,
        left_pad_source=True, left_pad_target=False,
        max_source_positions=1024, max_target_positions=1024,
        shuffle=True, input_feeding=True,
        remove_eos_from_source=False, append_eos_to_target=False,
    ):
        super().__init__(
            src, src_sizes, src_dict,
            tgt, tgt_sizes, tgt_dict,
            left_pad_source, left_pad_target,
            max_source_positions, max_target_positions,
            shuffle, input_feeding, remove_eos_from_source, append_eos_to_target
        )
        self.bert_src = bert_src
        self.bert_tgt = bert_tgt
        self.bert_src_sizes = bert_src_sizes
        self.bert_tgt_sizes = bert_tgt_sizes
        self.bert_src_dict = bert_src_dict
        self.bert_tgt_dict = bert_tgt_dict
        self.feat_embedder = BertFeatEmbed()

    def __getitem__(self, index):
        tgt_item = self.tgt[index] if self.tgt is not None else None
        src_item = self.src[index]

        bert_tgt_item = self.bert_tgt[index]
        bert_src_item = self.bert_src[index]

        # Append EOS to end of tgt sentence if it does not have an EOS and remove
        # EOS from end of src sentence if it exists. This is useful when we use
        # use existing datasets for opposite directions i.e., when we want to
        # use tgt_dataset as src_dataset and vice versa
        if self.append_eos_to_target:
            eos = self.tgt_dict.eos() if self.tgt_dict else self.src_dict.eos()
            if self.tgt and self.tgt[index][-1] != eos:
                tgt_item = torch.cat(
                    [self.tgt[index], torch.LongTensor([eos])])

        if self.remove_eos_from_source:
            eos = self.src_dict.eos()
            if self.src[index][-1] == eos:
                src_item = self.src[index][:-1]

        # add bert feature embeddings
        src_tokens = [self.bert_src_dict[tok] for tok in bert_src_item]
        tgt_tokens = [self.bert_tgt_dict[tok] for tok in bert_tgt_item]
        # remove EOS
        eos = self.bert_tgt_dict[self.bert_tgt_dict.eos()] if self.bert_tgt_dict \
            else self.bert_src_dict[self.bert_src_dict.eos()]
        if src_tokens[-1] == eos:
            src_tokens = src_tokens[:len(src_tokens)-1]
        if tgt_tokens[-1] == eos:
            tgt_tokens = tgt_tokens[:len(tgt_tokens)-1]

        # remove PAD
        pad = self.bert_tgt_dict[self.bert_tgt_dict.pad()] if self.bert_tgt_dict \
            else self.bert_src_dict[self.bert_src_dict.pad()]
        if pad in src_tokens:
            raise RuntimeError
        if pad in tgt_tokens:
            raise RuntimeError

        bert_src = self.feat_embedder(src_tokens)
        bert_tgt = self.feat_embedder(tgt_tokens)

        return {
            'id': index,
            'source': src_item,
            'target': tgt_item,
            'bert_src': bert_src,
            'bert_tgt': bert_tgt,
        }

    def collater(self, samples):
        return collate(
            samples, pad_idx=self.src_dict.pad(), eos_idx=self.src_dict.eos(),
            left_pad_source=self.left_pad_source, left_pad_target=self.left_pad_target,
            input_feeding=self.input_feeding,
        )
