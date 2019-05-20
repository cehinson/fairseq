
from . import FairseqIncrementalDecoder


class ExtraFeatDecoder(FairseqIncrementalDecoder):
    '''
    Decoder with extra input features
    '''

    def __init__(self, dictionary):
        super().__init__(dictionary)

    def forward(
        self, prev_output_tokens, encoder_out, extra_feats,
        incremental_state=None,
    ):
        raise NotImplementedError
