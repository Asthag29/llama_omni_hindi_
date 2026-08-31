import argparse
import datetime
import json
import os
import time
from pathlib import Path

import torch
import torchaudio

import gradio as gr
import numpy as np
import requests

from omni_speech.conversation import default_conversation, conv_templates
from omni_speech.constants import DEFAULT_SPEECH_PROMPT, LOGDIR
from omni_speech.train_utils import build_logger, server_error_msg
from omni_speech.model.speech_generator.speech_generator import IndicF5SpeechGenerator


logger = build_logger("gradio_web_server", "gradio_web_server.log")

speech_generator = None

DEFAULT_REFERENCE_AUDIO = Path(__file__).resolve().parents[2] / "data" / "inference.wav"
DEFAULT_REFERENCE_TEXT = "तुम कौन हो"

headers = {"User-Agent": "LLaMA-Omni Client"}


def get_conv_log_filename():
    t = datetime.datetime.now()
    name = os.path.join(LOGDIR, f"{t.year}-{t.month:02d}-{t.day:02d}-conv.json")
    return name


def get_model_list():
    ret = requests.post(args.controller_url + "/refresh_all_workers")
    assert ret.status_code == 200
    ret = requests.post(args.controller_url + "/list_models")
    models = ret.json()["models"]
    logger.info(f"Models: {models}")
    return models


get_window_url_params = """
function() {
    const params = new URLSearchParams(window.location.search);
    url_params = Object.fromEntries(params);
    console.log(url_params);
    return url_params;
    }
"""


def load_demo(url_params, request: gr.Request):
    logger.info(f"load_demo. ip: {request.client.host}. params: {url_params}")

    dropdown_update = gr.Dropdown(visible=True)
    if "model" in url_params:
        model = url_params["model"]
        if model in models:
            dropdown_update = gr.Dropdown(value=model, visible=True)

    state = default_conversation.copy()
    return state, dropdown_update


def load_demo_refresh_model_list(request: gr.Request):
    logger.info(f"load_demo. ip: {request.client.host}")
    models = get_model_list()
    state = default_conversation.copy()
    dropdown_update = gr.Dropdown(
        choices=models,
        value=models[0] if len(models) > 0 else ""
    )
    return state, dropdown_update


def clear_history(request: gr.Request):
    logger.info(f"clear_history. ip: {request.client.host}")
    state = default_conversation.copy()
    return (state, None, "", DEFAULT_REFERENCE_TEXT, None)


def normalize_audio(audio):
    audio = np.asarray(audio)
    if audio.ndim > 1:
        audio = audio.mean(axis=-1)
    if np.issubdtype(audio.dtype, np.integer):
        audio = audio.astype(np.float32) / np.iinfo(audio.dtype).max
    else:
        audio = audio.astype(np.float32)
        peak = np.max(np.abs(audio)) if audio.size else 0.0
        if peak > 1.0:
            audio = audio / 32768.0
    return audio


def get_default_reference() -> tuple[str, str]:
    if not DEFAULT_REFERENCE_AUDIO.is_file():
        raise FileNotFoundError(
            f"IndicF5 reference audio is missing: {DEFAULT_REFERENCE_AUDIO}"
        )
    return str(DEFAULT_REFERENCE_AUDIO), DEFAULT_REFERENCE_TEXT


def synthesize_with_indicf5(text, ref_audio_path, ref_text):
    if speech_generator is None:
        return None
    try:
        return speech_generator.synthesize(text, ref_audio_path, ref_text)
    except Exception as exc:
        logger.exception(f"IndicF5 synthesis failed: {exc}")
        return None


def add_speech(state, speech, request: gr.Request):
    text = (DEFAULT_SPEECH_PROMPT, speech)
    state = default_conversation.copy()
    state.append_message(state.roles[0], text)
    state.append_message(state.roles[1], None)
    state.skip_next = False
    return (state)


def http_bot(state, model_selector, temperature, top_p, max_new_tokens, request: gr.Request):
    logger.info(f"http_bot. ip: {request.client.host}")
    start_tstamp = time.time()
    model_name = model_selector

    if state.skip_next:
        # This generate call is skipped due to invalid inputs
        yield (state, "", "", None)
        return

    if len(state.messages) == state.offset + 2:
        # First round of conversation
        template_name = "llama_3"
        new_state = conv_templates[template_name].copy()
        new_state.append_message(new_state.roles[0], state.messages[-2][1])
        new_state.append_message(new_state.roles[1], None)
        state = new_state

    # Query worker address
    controller_url = args.controller_url
    ret = requests.post(controller_url + "/get_worker_address",
            json={"model": model_name})
    worker_addr = ret.json()["address"]
    logger.info(f"model_name: {model_name}, worker_addr: {worker_addr}")

    # No available worker
    if worker_addr == "":
        state.messages[-1][-1] = server_error_msg
        yield (state, "", "", None)
        return

    # Construct prompt
    prompt = state.get_prompt()

    sr, audio = state.messages[0][1][1]
    if speech_generator is not None:
        ref_audio_path, ref_text = get_default_reference()
        logger.info(f"Using IndicF5 default reference: {ref_audio_path}")
    else:
        ref_audio_path, ref_text = None, ""

    resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=16000)
    audio = torch.tensor(normalize_audio(audio)).unsqueeze(0)
    audio = resampler(audio).squeeze(0).numpy()
    audio = audio.tolist()
    # Make requests
    pload = {
        "model": model_name,
        "prompt": prompt,
        "temperature": float(temperature),
        "top_p": float(top_p),
        "max_new_tokens": min(int(max_new_tokens), 1500),
        "stop": state.sep2,
        "audio": audio,
    }

    yield (state, "", "", None)

    cur_dir = os.path.dirname(os.path.abspath(__file__))

    try:
        # Stream output
        response = requests.post(worker_addr + "/worker_generate_stream",
            headers=headers, json=pload, stream=True, timeout=10)
        output = ""
        for chunk in response.iter_lines(decode_unicode=False, delimiter=b"\0"):
            if chunk:
                data = json.loads(chunk.decode())
                if data["error_code"] == 0:
                    output = data["text"][len(prompt):].strip()
                    state.messages[-1][-1] = output

                    yield (state, output, ref_text, None)
                else:
                    output = data["text"] + f" (error_code: {data['error_code']})"
                    state.messages[-1][-1] = output
                    yield (state, "", "", None)
                    return
                time.sleep(0.03)
    except requests.exceptions.RequestException as e:
        state.messages[-1][-1] = server_error_msg
        yield (state, "", "", None)
        return

    return_value = synthesize_with_indicf5(output, ref_audio_path, ref_text)
    yield (state, output, ref_text, return_value)

    finish_tstamp = time.time()
    logger.info(f"{output}")
    logger.info(f"IndicF5 reference transcript: {ref_text}")


title_markdown = ("""
# 🎧 LLaMA-Omni: Seamless Speech Interaction with Large Language Models
""")

block_css = """

#buttons button {
    min-width: min(120px,100%);
}

"""

def build_demo(embed_mode, cur_dir=None, concurrency_count=10):
    with gr.Blocks(title="LLaMA-Omni Speech Chatbot", theme=gr.themes.Default(), css=block_css) as demo:
        state = gr.State()

        if not embed_mode:
            gr.Markdown(title_markdown)

        with gr.Row(elem_id="model_selector_row"):
            model_selector = gr.Dropdown(
                choices=models,
                value=models[0] if len(models) > 0 else "",
                interactive=True,
                show_label=False,
                container=False)

        with gr.Row():
            audio_input_box = gr.Audio(sources=["upload", "microphone"], label="Speech Input")
            with gr.Accordion("Parameters", open=True) as parameter_row:
                temperature = gr.Slider(minimum=0.0, maximum=1.0, value=0.0, step=0.1, interactive=True, label="Temperature",)
                top_p = gr.Slider(minimum=0.0, maximum=1.0, value=0.7, step=0.1, interactive=True, label="Top P",)
                max_output_tokens = gr.Slider(minimum=0, maximum=1024, value=512, step=64, interactive=True, label="Max Output Tokens",)

        if cur_dir is None:
            cur_dir = os.path.dirname(os.path.abspath(__file__))
        gr.Examples(examples=[
            [f"{cur_dir}/examples/example1.wav"],
            [f"{cur_dir}/examples/example2.wav"],
        ], inputs=[audio_input_box])

        with gr.Row():
            submit_btn = gr.Button(value="Send", variant="primary")
            clear_btn = gr.Button(value="Clear")

        text_output_box = gr.Textbox(label="Text Output", type="text")
        reference_text_box = gr.Textbox(
            label="Default Reference Transcript",
            value=DEFAULT_REFERENCE_TEXT,
            interactive=False,
            type="text",
        )
        audio_output_box = gr.Audio(label="Speech Output")

        url_params = gr.JSON(visible=False)

        submit_btn.click(
            add_speech,
            [state, audio_input_box],
            [state]
        ).then(
            http_bot,
            [state, model_selector, temperature, top_p, max_output_tokens],
            [state, text_output_box, reference_text_box, audio_output_box],
            concurrency_limit=concurrency_count
        )

        clear_btn.click(
            clear_history,
            None,
            [state, audio_input_box, text_output_box, reference_text_box, audio_output_box],
            queue=False
        )

        if args.model_list_mode == "once":
            demo.load(
                load_demo,
                [url_params],
                [state, model_selector],
                js=get_window_url_params
            )
        elif args.model_list_mode == "reload":
            demo.load(
                load_demo_refresh_model_list,
                None,
                [state, model_selector],
                queue=False
            )
        else:
            raise ValueError(f"Unknown model list mode: {args.model_list_mode}")

    return demo


def build_speech_output_backend(args):
    global speech_generator
    get_default_reference()
    speech_generator = IndicF5SpeechGenerator(
        model_path=args.indicf5_model_path,
        repo_id=args.indicf5_repo_id,
        device=args.indicf5_device,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int)
    parser.add_argument("--controller-url", type=str, default="http://localhost:21001")
    parser.add_argument("--concurrency-count", type=int, default=16)
    parser.add_argument("--model-list-mode", type=str, default="once",
        choices=["once", "reload"])
    parser.add_argument("--share", action="store_true")
    parser.add_argument("--moderate", action="store_true")
    parser.add_argument("--embed", action="store_true")
    parser.add_argument("--indicf5-model-path", type=str, default="models/indicf5")
    parser.add_argument("--indicf5-repo-id", type=str, default="ai4bharat/IndicF5")
    parser.add_argument("--indicf5-device", type=str, default=None)
    args = parser.parse_args()
    logger.info(f"args: {args}")

    models = get_model_list()
    build_speech_output_backend(args)

    logger.info(args)
    demo = build_demo(args.embed, concurrency_count=args.concurrency_count)
    demo.queue(
        api_open=False
    ).launch(
        server_name=args.host,
        server_port=args.port,
        share=args.share
    )