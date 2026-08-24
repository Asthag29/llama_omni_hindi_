CONTROLLER_HEART_BEAT_EXPIRATION = 30
WORKER_HEART_BEAT_INTERVAL = 15

LOGDIR = "."

# Model Constants
IGNORE_INDEX = -100
SPEECH_TOKEN_INDEX = -200
DEFAULT_SPEECH_TOKEN = "<speech>"

DEFAULT_SPEECH_PROMPT = (
    "<speech>\n"
    "आप हिंदी लामा मॉडल हैं। उपयोगकर्ता की आवाज़ सुनें और उनके प्रश्न का उत्तर हिंदी "
    "(देवनागरी लिपि) में दें। "
    "Roman/Latin अक्षरों (Hinglish) का उपयोग न करें।"
)