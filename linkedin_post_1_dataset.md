Happy to share Hindi-speech-instruct, a dataset of spoken Hindi questions paired with text answers, built for training speech-language models in Hindi.

I have been building a voice assistant for Hindi, and while looking for training data I realised that no dataset pairs spoken questions with text answers. What exists is either text-to-text or audio with its transcription. So I built Hindi-speech-instruct myself. Generating the dataset ended up taking more time than the modelling did.

What is inside: about 110,700 single-turn examples. Each one has a Hindi question as 16 kHz mono FLAC audio, the same question as text, and a Hindi text answer. The audio comes to 158 GB, with clips from under a second to several minutes. Train, validation and test splits are included, and it loads with one line of Hugging Face datasets.

The text comes from AI4Bharat's indic-instruct-data-v0.1, keeping the single-turn conversations from four of its subsets: Flan v2 (66.8k), LMSYS (34.8k), Anudesh (7.5k) and HH-RLHF (1.5k). Between them the questions cover a wide range: summarisation, translation and classification tasks, open questions and explanations, writing and emails, maths, coding, everyday advice, and human-written questions with Indian context from Anudesh. Every question was then spoken with Meta's MMS Hindi TTS (facebook/mms-tts-hin).

Beyond speech assistants, it should be useful for Hindi ASR, spoken question answering, and for measuring how well a speech model actually listens in Hindi. It already seems to be finding its way into the community, with more than 500 downloads in a few weeks.

Dataset: https://huggingface.co/datasets/Pastaaaaa2003/Hindi-speech-instruct

Happy to talk if you want to know more about how it was built.
