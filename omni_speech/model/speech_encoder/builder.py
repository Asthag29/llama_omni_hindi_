from .speech_encoder import WhisperWrappedEncoder

#* look for the speech encoder type in the config and look for the type contains whisper
def build_speech_encoder(config):
    speech_encoder_type = getattr(config, 'speech_encoder_type', None)
    if "whisper" in speech_encoder_type.lower():
        return WhisperWrappedEncoder.load(config)

    raise ValueError(f'Unknown speech encoder: {speech_encoder_type}')
