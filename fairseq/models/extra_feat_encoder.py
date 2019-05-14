
from .fairseq_encoder import FairseqEncoder


class ExtraFeatEncoder(FairseqEncoder):
    '''
    Encoder with extra input features
    '''

    def __init__(self, dictionary):
        super().__init__(dictionary)

    def forward(self, src_tokens, src_lengths, extra_feats):
        raise NotImplementedError
