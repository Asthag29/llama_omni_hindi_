from .speech_generator import IndicF5SpeechGenerator


def build_speech_generator(config):
    generator_type = getattr(config, "speech_generator_type", "indicf5")
    if generator_type in {"indicf5", "indic_f5"}:
        return IndicF5SpeechGenerator(
            model_path=getattr(config, "indicf5_model_path", "models/indicf5"),
            repo_id=getattr(config, "indicf5_repo_id", "ai4bharat/IndicF5"),
            device=getattr(config, "indicf5_device", None),
        )

    raise ValueError(f"Unknown speech generator type: {generator_type}")
