# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree. An additional grant of patent rights
# can be found in the PATENTS file in the same directory.

import itertools
import os

from fairseq import options
from fairseq.data import (
    ConcatDataset,
    data_utils,
    Dictionary,
    IndexedCachedDataset,
    IndexedDataset,
    IndexedRawTextDataset,
    LanguagePairDataset,
    LangPairBertFeatDataset
)

from . import FairseqTask, register_task


@register_task('translation')
class TranslationTask(FairseqTask):
    """
    Translate from one (source) language to another (target) language.

    Args:
        src_dict (Dictionary): dictionary for the source language
        tgt_dict (Dictionary): dictionary for the target language

    .. note::

        The translation task is compatible with :mod:`fairseq-train`,
        :mod:`fairseq-generate` and :mod:`fairseq-interactive`.

    The translation task provides the following additional command-line
    arguments:

    .. argparse::
        :ref: fairseq.tasks.translation_parser
        :prog:
    """

    @staticmethod
    def add_args(parser):
        """Add task-specific arguments to the parser."""
        # fmt: off
        parser.add_argument('data', nargs='+',
                            help='path(s) to data directorie(s)')
        parser.add_argument('-s', '--source-lang', default=None, metavar='SRC',
                            help='source language')
        parser.add_argument('-t', '--target-lang', default=None, metavar='TARGET',
                            help='target language')
        parser.add_argument('--lazy-load', action='store_true',
                            help='load the dataset lazily')
        parser.add_argument('--raw-text', action='store_true',
                            help='load raw text dataset')
        parser.add_argument('--left-pad-source', default='True', type=str, metavar='BOOL',
                            help='pad the source on the left')
        parser.add_argument('--left-pad-target', default='False', type=str, metavar='BOOL',
                            help='pad the target on the left')
        parser.add_argument('--max-source-positions', default=1024, type=int, metavar='N',
                            help='max number of tokens in the source sequence')
        parser.add_argument('--max-target-positions', default=1024, type=int, metavar='N',
                            help='max number of tokens in the target sequence')
        parser.add_argument('--upsample-primary', default=1, type=int,
                            help='amount to upsample primary dataset')
        # fmt: on
        parser.add_argument('--use_bert', default='False', type=str,
                            metavar='BOOL', help='use bert feature embeddings')
        parser.add_argument('--bert_data',
                            help='path to data that has NOT had BPE applied to it')

    def __init__(self, args, src_dict, tgt_dict, bert_src_dict=None, bert_tgt_dict=None):
        super().__init__(args)
        self.src_dict = src_dict
        self.tgt_dict = tgt_dict
        self.bert_src_dict = bert_src_dict
        self.bert_tgt_dict = bert_tgt_dict

    @classmethod
    def setup_task(cls, args, **kwargs):
        """Setup the task (e.g., load dictionaries).

        Args:
            args (argparse.Namespace): parsed command-line arguments
        """
        args.left_pad_source = options.eval_bool(args.left_pad_source)
        args.left_pad_target = options.eval_bool(args.left_pad_target)

        # upgrade old checkpoints
        if isinstance(args.data, str):
            args.data = [args.data]

        if isinstance(args.bert_data, str):
            args.bert_data = [args.bert_data]

        # find language pair automatically
        if args.source_lang is None or args.target_lang is None:
            args.source_lang, args.target_lang = data_utils.infer_language_pair(
                args.data[0])
        if args.source_lang is None or args.target_lang is None:
            raise Exception(
                'Could not infer language pair, please provide it explicitly')

        # find language pair automatically -- Bert data has NO BPE applied
        if args.use_bert:
            if args.source_lang is None or args.target_lang is None:
                args.source_lang, args.target_lang = data_utils.infer_language_pair(
                    args.bert_data[0])
            if args.source_lang is None or args.target_lang is None:
                raise Exception(
                    'Could not infer bert_data language pair, please provide it explicitly')

        # load dictionaries
        src_dict = cls.load_dictionary(os.path.join(
            args.data[0], 'dict.{}.txt'.format(args.source_lang)))
        tgt_dict = cls.load_dictionary(os.path.join(
            args.data[0], 'dict.{}.txt'.format(args.target_lang)))
        assert src_dict.pad() == tgt_dict.pad()
        assert src_dict.eos() == tgt_dict.eos()
        assert src_dict.unk() == tgt_dict.unk()
        print('| [{}] dictionary: {} types'.format(
            args.source_lang, len(src_dict)))
        print('| [{}] dictionary: {} types'.format(
            args.target_lang, len(tgt_dict)))

        # load bert data dictionaries
        bert_src_dict = None
        bert_tgt_dict = None
        if args.use_bert:
            bert_src_dict = cls.load_dictionary(os.path.join(
                args.bert_data[0], 'dict.{}.txt'.format(args.source_lang)))
            bert_tgt_dict = cls.load_dictionary(os.path.join(
                args.bert_data[0], 'dict.{}.txt'.format(args.target_lang)))
            assert bert_src_dict.pad() == bert_tgt_dict.pad()
            assert bert_src_dict.eos() == bert_tgt_dict.eos()
            assert bert_src_dict.unk() == bert_tgt_dict.unk()
            print('| [{}] bert dictionary: {} types'.format(
                args.source_lang, len(bert_src_dict)))
            print('| [{}] bert dictionary: {} types'.format(
                args.target_lang, len(bert_tgt_dict)))

        return cls(args, src_dict, tgt_dict, bert_src_dict, bert_tgt_dict)

    def load_dataset(self, split, combine=False, **kwargs):
        """Load a given dataset split.

        Args:
            split (str): name of the split (e.g., train, valid, test)
        """

        def split_exists(split, src, tgt, lang, data_path):
            filename = os.path.join(
                data_path, '{}.{}-{}.{}'.format(split, src, tgt, lang))
            if self.args.raw_text and IndexedRawTextDataset.exists(filename):
                return True
            elif not self.args.raw_text and IndexedDataset.exists(filename):
                return True
            return False

        def indexed_dataset(path, dictionary):
            if self.args.raw_text:
                return IndexedRawTextDataset(path, dictionary)
            elif IndexedDataset.exists(path):
                if self.args.lazy_load:
                    return IndexedDataset(path, fix_lua_indexing=True)
                else:
                    return IndexedCachedDataset(path, fix_lua_indexing=True)
            return None

        src_datasets = []
        tgt_datasets = []

        for dk, data_path in enumerate(self.args.data):
            for k in itertools.count():
                split_k = split + (str(k) if k > 0 else '')

                # infer langcode
                src, tgt = self.args.source_lang, self.args.target_lang
                if split_exists(split_k, src, tgt, src, data_path):
                    prefix = os.path.join(
                        data_path, '{}.{}-{}.'.format(split_k, src, tgt))
                elif split_exists(split_k, tgt, src, src, data_path):
                    prefix = os.path.join(
                        data_path, '{}.{}-{}.'.format(split_k, tgt, src))
                else:
                    if k > 0 or dk > 0:
                        break
                    else:
                        raise FileNotFoundError(
                            'Dataset not found: {} ({})'.format(split, data_path))

                src_datasets.append(indexed_dataset(
                    prefix + src, self.src_dict))
                tgt_datasets.append(indexed_dataset(
                    prefix + tgt, self.tgt_dict))

                print('| {} {} {} examples'.format(
                    data_path, split_k, len(src_datasets[-1])))

                if not combine:
                    break

        assert len(src_datasets) == len(tgt_datasets)

        if self.args.use_bert:
            bert_src_datasets = []
            bert_tgt_datasets = []

        for dk, data_path in enumerate(self.args.bert_data):
            for k in itertools.count():
                split_k = split + (str(k) if k > 0 else '')

                # infer langcode
                src, tgt = self.args.source_lang, self.args.target_lang
                if split_exists(split_k, src, tgt, src, data_path):
                    prefix = os.path.join(
                        data_path, '{}.{}-{}.'.format(split_k, src, tgt))
                elif split_exists(split_k, tgt, src, src, data_path):
                    prefix = os.path.join(
                        data_path, '{}.{}-{}.'.format(split_k, tgt, src))
                else:
                    if k > 0 or dk > 0:
                        break
                    else:
                        raise FileNotFoundError(
                            'Bert Dataset not found: {} ({})'.format(split, data_path))

                bert_src_datasets.append(indexed_dataset(
                    prefix + src, self.bert_src_dict))
                bert_tgt_datasets.append(indexed_dataset(
                    prefix + tgt, self.bert_tgt_dict))

                print('| {} {} {} examples'.format(
                    data_path, split_k, len(bert_src_datasets[-1])))

                if not combine:
                    break

        assert len(bert_src_datasets) == len(bert_tgt_datasets)

        if len(src_datasets) == 1:
            src_dataset, tgt_dataset = src_datasets[0], tgt_datasets[0]
        else:
            sample_ratios = [1] * len(bert_src_datasets)
            sample_ratios[0] = self.args.upsample_primary
            src_dataset = ConcatDataset(src_datasets, sample_ratios)
            tgt_dataset = ConcatDataset(tgt_datasets, sample_ratios)

        if len(bert_src_datasets) == 1:
            bert_src_dataset, bert_tgt_dataset = bert_src_datasets[0], bert_tgt_datasets[0]
        else:
            sample_ratios = [1] * len(src_datasets)
            sample_ratios[0] = self.args.upsample_primary
            bert_src_dataset = ConcatDataset(bert_src_datasets, sample_ratios)
            bert_tgt_dataset = ConcatDataset(bert_tgt_datasets, sample_ratios)

        if self.args.use_bert:
            self.datasets[split] = LangPairBertFeatDataset(
                src_dataset, src_dataset.sizes, self.src_dict,
                tgt_dataset, tgt_dataset.sizes, self.tgt_dict,
                bert_src_dataset, bert_src_dataset.sizes, self.bert_src_dict,
                bert_tgt_dataset, bert_tgt_dataset.sizes, self.bert_tgt_dict,
                left_pad_source=self.args.left_pad_source,
                left_pad_target=self.args.left_pad_target,
                max_source_positions=self.args.max_source_positions,
                max_target_positions=self.args.max_target_positions,
            )
        else:
            self.datasets[split] = LanguagePairDataset(
                src_dataset, src_dataset.sizes, self.src_dict,
                tgt_dataset, tgt_dataset.sizes, self.tgt_dict,
                left_pad_source=self.args.left_pad_source,
                left_pad_target=self.args.left_pad_target,
                max_source_positions=self.args.max_source_positions,
                max_target_positions=self.args.max_target_positions,
            )

    def build_dataset_for_inference(self, src_tokens, src_lengths):
        return LanguagePairDataset(src_tokens, src_lengths, self.source_dictionary)

    def max_positions(self):
        """Return the max sentence length allowed by the task."""
        return (self.args.max_source_positions, self.args.max_target_positions)

    @property
    def source_dictionary(self):
        """Return the source :class:`~fairseq.data.Dictionary`."""
        return self.src_dict

    @property
    def target_dictionary(self):
        """Return the target :class:`~fairseq.data.Dictionary`."""
        return self.tgt_dict
