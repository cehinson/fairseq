import torch
from pytorch_pretrained_bert import BertTokenizer, BertModel


class BERTFeatEmbedder:

    def __init__(self, n_ctx_embs=4, ctx_emb_dim=768, bert_model='bert-base-uncased'):
        self.n_ctx_embs = n_ctx_embs
        self.ctx_emb_dim = ctx_emb_dim
        assert(n_ctx_embs <= 12)  # BERT (base) has 12 layers
        # Load bert model & tokenizer
        self.model = BertModel.from_pretrained(bert_model)
        self.model.eval()
        self.model.to('cuda')
        self.tokenizer = BertTokenizer.from_pretrained(
            bert_model,
            do_lower_case=True
        )

    def prepare_sentence(self, sentence):
        word_start_idxs = [1]  # where words were split into subwords

        tokens = [['[CLS]']]
        for word in sentence:
            subwords = self.tokenizer.tokenize(word)
            tokens.append(subwords)
            word_start_idxs.append(word_start_idxs[-1] + len(subwords))
        tokens.append(['[SEP]'])

        word_start_idxs = word_start_idxs[:len(word_start_idxs)-1]
        # flatten the list
        tokens = [item for sublist in tokens for item in sublist]
        token_idxs = self.tokenizer.convert_tokens_to_ids(tokens)

        return (tokens, token_idxs, word_start_idxs)

    def __call__(self, sentences):
        bert_embed = torch.zeros(
            (len(sentences),             # Batch size
             max(map(len, sentences)),   # Max Sent. Length
             self.n_ctx_embs,            # Number of Layers
             self.ctx_emb_dim)           # Hidden dimension
        )

        # (1) Tokenize using BERT tokenizer
        sent_token_idxs = []
        sent_ws_idxs = []  # word start indicies
        for sent in sentences:
            tokens, token_idxs, ws_idxs = self.prepare_sentence(sent)
            sent_token_idxs.append(token_idxs)
            sent_ws_idxs.append(ws_idxs)

        # (2) Use BERT for ctx embedding
        batch_embedding = []
        for t, sent_idxs in enumerate(sent_token_idxs):
            with torch.no_grad():
                idxs = torch.tensor([sent_idxs], device='cuda')
                encoded_layers, _ = self.model(idxs)
                # take the last n_ctx_embs layers
                encoded_layers = torch.stack(
                    encoded_layers[-self.n_ctx_embs:]
                ).squeeze(1)
                # only take hidden states that count...
                encoded_layers = encoded_layers[:, sent_ws_idxs[t], :]
                batch_embedding.append(encoded_layers)

        # (3)
        for b, _ in enumerate(batch_embedding):
            arr = batch_embedding[b].transpose(0, 1)
            bert_embed[b, :arr.shape[0], :, :] = arr

        return bert_embed
