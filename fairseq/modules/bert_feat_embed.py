import torch
from pytorch_pretrained_bert import BertTokenizer, BertModel

# FIXME this is a temporary workaround for now...
use_cuda = torch.cuda.is_available()


class BertFeatEmbed:

    def __init__(self, num_layers=4, emb_dim=768, bert_model='bert-base-uncased'):
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
            do_lower_case=True
        )

    def prepare_sentence(self, sentence):
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

    def __call__(self, tokens):
        features = torch.zeros(
            len(tokens),
            self.num_layers,            # Number of Layers
            self.emb_dim           # Hidden dimension
        )

        # (1) Tokenize using BERT tokenizer
        tokens, token_ids, word_starts = self.prepare_sentence(tokens)

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
            layer_out = layer_out[:, word_starts, :]

        features = layer_out.permute(1, 0, 2)
        return features
