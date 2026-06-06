from .speech_projector import EncoderProjectorConcat

#* will look for the projector type in the config and look for the type contains linear
def build_speech_projector(config):
    projector_type = getattr(config, 'speech_projector_type', 'linear')
    if projector_type == 'linear':
        return EncoderProjectorConcat(config)

    raise ValueError(f'Unknown projector type: {projector_type}')
