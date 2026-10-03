import argparse
import html
import json
import os
import time

import gradio as gr
import requests

from omni_speech.conversation import default_conversation, conv_templates
from omni_speech.constants import DEFAULT_MAX_NEW_TOKENS, DEFAULT_SPEECH_PROMPT
from omni_speech.serve.utils import build_logger, server_error_msg
from omni_speech.train_utils import load_audio_16k
from omni_speech.tts.indicf5 import (
    DEFAULT_REFERENCE_AUDIO,
    DEFAULT_REFERENCE_TEXT,
    IndicF5SpeechGenerator,
)


logger = build_logger("gradio_web_server", "gradio_web_server.log")

speech_generator = None

headers = {"User-Agent": "Hindi LLaMA-Omni Client"}


def get_model_list():
    try:
        ret = requests.post(args.controller_url + "/refresh_all_workers")
        if ret.status_code != 200:
            raise SystemExit(
                f"Controller at {args.controller_url} returned status {ret.status_code} "
                "for /refresh_all_workers."
            )
        ret = requests.post(args.controller_url + "/list_models")
    except requests.exceptions.RequestException:
        raise SystemExit(
            f"Cannot reach the controller at {args.controller_url}. "
            "Start `python -m omni_speech.serve.controller` first."
        )
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


IDLE_STATUS = "Ask a question to get a spoken answer. · सवाल पूछें, जवाब यहाँ आएगा।"
WORKER_DOWN_MSG = "The model worker is not responding. Check that it is running, then try again."
UNEXPECTED_ERROR_MSG = "Something went wrong while answering. Please try again."


def render_status(message, kind="progress"):
    """Status line HTML; kind is one of "idle", "progress", "error"."""
    if not message:
        return ""
    return f'<div class="status status-{kind}" role="status">{html.escape(message)}</div>'


def ask_button(enabled):
    return gr.Button(interactive=enabled)


def start_request():
    return (ask_button(False), render_status("Uploading your question…"), "", None)


def clear_history(request: gr.Request):
    logger.info(f"clear_history. ip: {request.client.host}")
    state = default_conversation.copy()
    return (state, None, render_status(IDLE_STATUS, "idle"), "", None, ask_button(True))


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
    state.skip_next = speech is None
    return (state)


def http_bot(state, model_selector, temperature, top_p, max_new_tokens, request: gr.Request):
    """Yields (state, status_html, answer_text, answer_audio, ask_button_update).

    Every terminal yield re-enables the Ask button.
    """
    logger.info(f"http_bot. ip: {request.client.host}")
    try:
        yield from _http_bot_stream(state, model_selector, temperature, top_p, max_new_tokens)
    except Exception as exc:
        logger.exception(f"http_bot failed: {exc}")
        yield (state, render_status(UNEXPECTED_ERROR_MSG, "error"), gr.skip(), None, ask_button(True))


def _http_bot_stream(state, model_selector, temperature, top_p, max_new_tokens):
    model_name = model_selector
    busy = gr.skip()

    if state.skip_next:
        # This generate call is skipped due to invalid inputs
        yield (state, render_status("Please record or upload a Hindi question first.", "error"),
               "", None, ask_button(True))
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
    try:
        ret = requests.post(controller_url + "/get_worker_address",
                json={"model": model_name}, timeout=10)
        worker_addr = ret.json()["address"]
    except (requests.exceptions.RequestException, ValueError, KeyError) as e:
        logger.error(f"Controller request failed ({controller_url}): {e}")
        worker_addr = ""
    logger.info(f"model_name: {model_name}, worker_addr: {worker_addr}")

    # No available worker
    if worker_addr == "":
        state.messages[-1][-1] = server_error_msg
        yield (state, render_status(WORKER_DOWN_MSG, "error"), "", None, ask_button(True))
        return

    # Construct prompt
    prompt = state.get_prompt()

    audio_path = state.messages[0][1][1]
    if speech_generator is not None:
        ref_audio_path, ref_text = get_default_reference()
        logger.info(f"Using IndicF5 default reference: {ref_audio_path}")
    else:
        ref_audio_path, ref_text = None, ""

    # Same loader as inference.py, so the demo and the script give the same answer.
    audio = load_audio_16k(audio_path).tolist()
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

    thinking_status = render_status("Listening and thinking…")
    yield (state, thinking_status, "", None, busy)

    try:
        # Stream output
        response = requests.post(worker_addr + "/worker_generate_stream",
            headers=headers, json=pload, stream=True, timeout=(10, 120))
        output = ""
        for chunk in response.iter_lines(decode_unicode=False, delimiter=b"\0"):
            if chunk:
                data = json.loads(chunk.decode())
                if data["error_code"] == 0:
                    output = data["text"][len(prompt):].strip()
                    state.messages[-1][-1] = output

                    yield (state, thinking_status, output, None, busy)
                else:
                    output = data["text"] + f" (error_code: {data['error_code']})"
                    state.messages[-1][-1] = output
                    logger.error(f"Worker returned error_code {data['error_code']}: {data['text']}")
                    yield (state, render_status(
                        f"The model could not answer this question (error code {data['error_code']}). "
                        "Please try again.", "error"), "", None, ask_button(True))
                    return
                time.sleep(0.03)
    except requests.exceptions.RequestException as e:
        logger.error(f"Worker request failed: {e}")
        state.messages[-1][-1] = server_error_msg
        yield (state, render_status(WORKER_DOWN_MSG, "error"), "", None, ask_button(True))
        return

    yield (state, render_status("Generating speech…"), output, None, busy)

    return_value = synthesize_with_indicf5(output, ref_audio_path, ref_text)
    if return_value is None:
        yield (state, render_status(
            "The text answer is ready, but speech could not be generated.", "error"),
            output, None, ask_button(True))
    else:
        yield (state, "", output, return_value, ask_button(True))

    logger.info(f"{output}")
    logger.info(f"IndicF5 reference transcript: {ref_text}")


title_html = """
<div id="header">
  <h1>🦙🎧 Hindi LLaMA-Omni</h1>
  <p>Record or upload a Hindi question and get a spoken Hindi answer.</p>
  <p lang="hi">हिंदी में सवाल पूछें, जवाब आवाज़ में पाएँ।</p>
  <nav class="links">
    <a href="https://huggingface.co/Pastaaaaa2003/hindi-llama-omni-model" target="_blank" rel="noopener">Model</a>
    <a href="https://huggingface.co/datasets/Pastaaaaa2003/Hindi-speech-instruct" target="_blank" rel="noopener">Dataset</a>
    <a href="https://github.com/Asthag29/llama_omni_hindi_" target="_blank" rel="noopener">Code</a>
    <a href="https://github.com/ictnlp/LLaMA-Omni" target="_blank" rel="noopener">Base model: LLaMA-Omni</a>
  </nav>
</div>
"""

footer_html = """
<div id="footer">
  Built on LLaMA-Omni, Whisper large-v3, and IndicF5.
  Answers are generated by a model and may be wrong.
</div>
"""

block_css = """
.gradio-container { max-width: 1100px !important; margin: 0 auto !important; }
#header { text-align: center; padding: 12px 0 4px; }
#header h1 { margin: 0 0 6px; font-size: 2rem; color: var(--body-text-color); }
#header p { margin: 2px 0; color: var(--body-text-color-subdued); }
#header .links { display: flex; flex-wrap: wrap; justify-content: center; gap: 8px; margin-top: 12px; }
#header .links a {
    padding: 3px 12px; border: 1px solid var(--border-color-primary); border-radius: 999px;
    background: var(--background-fill-secondary); color: var(--body-text-color);
    font-size: var(--text-sm); text-decoration: none;
}
#header .links a:hover { border-color: var(--color-accent); color: var(--color-accent); }
.panel-heading h3 { margin: 0 !important; }
#buttons button { min-width: min(120px, 100%); }
.status {
    padding: 8px 12px; border-radius: var(--radius-lg); font-size: var(--text-md);
    border: 1px solid var(--border-color-primary); border-left: 4px solid var(--color-accent);
    background: var(--background-fill-secondary); color: var(--body-text-color);
}
.status-idle { border-left-color: var(--border-color-primary); color: var(--body-text-color-subdued); }
.status-error {
    border-color: var(--error-border-color); border-left-color: var(--error-text-color);
    background: var(--error-background-fill); color: var(--error-text-color);
}
#answer textarea { font-size: 1.2rem; line-height: 1.8; }
#footer { text-align: center; font-size: var(--text-sm); color: var(--body-text-color-subdued); padding: 8px 0; }
"""


def build_demo(embed_mode, cur_dir=None, concurrency_count=10):
    with gr.Blocks(title="Hindi LLaMA-Omni", theme=gr.themes.Soft(), css=block_css) as demo:
        state = gr.State()

        if not embed_mode:
            gr.HTML(title_html)

        if cur_dir is None:
            cur_dir = os.path.dirname(os.path.abspath(__file__))

        with gr.Row(equal_height=False):
            with gr.Column(scale=1, min_width=320, variant="panel"):
                gr.Markdown("### 1 · Ask your question", elem_classes="panel-heading")
                audio_input_box = gr.Audio(
                    sources=["upload", "microphone"],
                    type="filepath",
                    label="Your question (Hindi)",
                )
                gr.Examples(
                    examples=[
                        [f"{cur_dir}/examples/example1.wav"],
                        [f"{cur_dir}/examples/example2.wav"],
                    ],
                    inputs=[audio_input_box],
                    label="Or try an example, then press Ask",
                )
                with gr.Row(elem_id="buttons"):
                    submit_btn = gr.Button(value="Ask", variant="primary")
                    clear_btn = gr.Button(value="Clear")
                with gr.Accordion("Advanced", open=False):
                    model_selector = gr.Dropdown(
                        choices=models,
                        value=models[0] if models else "",
                        label="Model",
                        interactive=True,
                    )
                    temperature = gr.Slider(minimum=0.0, maximum=1.0, value=0.0, step=0.1, interactive=True, label="Temperature")
                    top_p = gr.Slider(minimum=0.0, maximum=1.0, value=0.7, step=0.1, interactive=True, label="Top P")
                    max_output_tokens = gr.Slider(minimum=64, maximum=1024, value=DEFAULT_MAX_NEW_TOKENS, step=64, interactive=True, label="Max Output Tokens")
                    gr.Textbox(
                        label="IndicF5 reference transcript (fixed voice)",
                        value=DEFAULT_REFERENCE_TEXT,
                        interactive=False,
                    )

            with gr.Column(scale=1, min_width=320, variant="panel"):
                gr.Markdown("### 2 · Answer", elem_classes="panel-heading")
                status_box = gr.HTML(render_status(IDLE_STATUS, "idle"), elem_id="status")
                text_output_box = gr.Textbox(
                    label="Answer (text)", lines=6, interactive=False,
                    show_copy_button=True, elem_id="answer",
                )
                audio_output_box = gr.Audio(label="Answer (speech)", autoplay=True, interactive=False)

        if not embed_mode:
            gr.HTML(footer_html)

        url_params = gr.JSON(visible=False)

        submit_btn.click(
            start_request,
            None,
            [submit_btn, status_box, text_output_box, audio_output_box],
            queue=False
        ).then(
            add_speech,
            [state, audio_input_box],
            [state]
        ).then(
            http_bot,
            [state, model_selector, temperature, top_p, max_output_tokens],
            [state, status_box, text_output_box, audio_output_box, submit_btn],
            concurrency_limit=concurrency_count
        ).then(
            # Safety net: .then runs even if a previous step failed.
            lambda: ask_button(True),
            None,
            [submit_btn],
            queue=False
        )

        clear_btn.click(
            clear_history,
            None,
            [state, audio_input_box, status_box, text_output_box, audio_output_box, submit_btn],
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