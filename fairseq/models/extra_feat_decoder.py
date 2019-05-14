
from . import FairseqDecoder


class ExtraFeatDecoder(FairseqDecoder):
    '''
    Decoder with extra input features
    '''

    def __init__(self, dictionary):
        super().__init__(self, dictionary)

    def forward(self, prev_output_tokens, decoder_out, extra_feats):
        raise NotImplementedError
