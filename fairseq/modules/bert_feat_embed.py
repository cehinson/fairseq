import torch
from pytorch_pretrained_bert import BertTokenizer, BertModel

# FIXME this is a temporary workaround for now...
use_cuda = torch.cuda.is_available()


class BertFeatEmbed:

    def __init__(self, num_layers=4, emb_dim=768, bert_model='bert-base-cased'):
        self.num_layers = num_layers
        self.emb_dim = emb_dim

        assert(num_layers <= 12)  # BERT (base) has 12 layers
        assert(num_layers >= 1)

        # Load bert model & tokenizer
        self.model = BertModel.from_pretrained(bert_model)
        self.model.eval()

        if use_cuda:
            self.model.to('cuda')

        self.tokenizer = BertTokenizer.from_pretrained(
            bert_model,
            do_lower_case=False
        )

    def new_prep_sentence(self, sentence):
        '''Sentences are pre-tokenized'''

        word_starts = []  # indices where words are split into subwords
        tokens = ['[CLS]']
        for i, subword in enumerate(sentence, 1):
            tokens.append(subword)
            if '##' not in subword:
                word_starts.append(i)
        tokens.append('[SEP]')
        token_ids = self.tokenizer.convert_tokens_to_ids(tokens)

        return (tokens, token_ids, word_starts)

    def prepare_sentence(self, sentence):
        '''We have to tokenize the sentences ourselves...'''
        word_starts = [1]  # indices where words are split into subwords
        tokens = [['[CLS]']]

        for word in sentence:
            subwords = self.tokenizer.tokenize(word)
            tokens.append(subwords)
            word_starts.append(word_starts[-1] + len(subwords))
        tokens.append(['[SEP]'])

        word_starts = word_starts[:len(word_starts)-1]
        # flatten the list
        tokens = [item for sublist in tokens for item in sublist]
        token_ids = self.tokenizer.convert_tokens_to_ids(tokens)

        return (tokens, token_ids, word_starts)

    def __call__(self, sample, max_len=510):
        features = torch.zeros(
            len(sample),
            self.num_layers,       # Number of Layers
            self.emb_dim           # Hidden dimension
        )
        # chunk into chunks < max_len
        chunks = [sample[x:x+max_len] for x in range(0, len(sample), max_len)]
        chunk_lens = [len(sample[x:x+max_len])
                      for x in range(0, len(sample), max_len)]
        # indices where chunks start/stop
        cids = [sum(chunk_lens[:i]) for i in range(len(chunk_lens)+1)]

        breakpoint()
        for i, chunk in enumerate(chunks, 1):
            # (1) Tokenize using BERT tokenizer
            tokens, token_ids, word_starts = self.new_prep_sentence(chunk)
            # (2) Use BERT for ctx embedding
            with torch.no_grad():
                ids = torch.tensor([token_ids])
                if use_cuda:
                    ids = ids.cuda()
                layer_out, _ = self.model(ids)
                # take the last num_layers layers
                layer_out = torch.stack(
                    layer_out[-self.num_layers:]
                ).squeeze(1)
                # only take hidden states that count...
                # layer_out = layer_out[:, word_starts, :]
                # ignore [CLS] and [SEP] tags
                layer_out = layer_out[:, 1:len(tokens)-1, :]
                layer_out = layer_out.permute(1, 0, 2)

            features[cids[i-1]:cids[i], :, :] = layer_out

        return features
