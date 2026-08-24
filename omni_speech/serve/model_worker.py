"""
A model worker executes the model.
"""
import argparse
import asyncio
import json
import time
import threading
import uuid
from pathlib import Path

from fastapi import FastAPI, Request, BackgroundTasks
from fastapi.responses import StreamingResponse
import requests
import torch
import uvicorn
import whisper
import numpy as np
from functools import partial

from omni_speech.constants import WORKER_HEART_BEAT_INTERVAL
from omni_speech.train_utils import (build_logger, server_error_msg,
    pretty_print_semaphore)
from omni_speech.model.builder import load_pretrained_model
from omni_speech.datasets.preprocess import tokenizer_speech_token
from omni_speech.infer.inference import load_inference_cfg, load_module_from_checkpoint
from transformers import TextIteratorStreamer
from threading import Thread


GB = 1 << 30

worker_id = str(uuid.uuid4())[:6]
logger = build_logger("model_worker", f"model_worker_{worker_id}.log")
global_counter = 0

model_semaphore = None


def heart_beat_worker(controller):

    while True:
        time.sleep(WORKER_HEART_BEAT_INTERVAL)
        controller.send_heart_beat()


def load_speech(audio, input_type, mel_size, speech_normalize):
    speech = np.array(audio, dtype=np.float32)
    if input_type == "raw":
        speech = torch.from_numpy(speech)
        if speech_normalize:
            speech = torch.nn.functional.layer_norm(speech, speech.shape)
    elif input_type == "mel":
        speech = whisper.pad_or_trim(speech)
        speech = whisper.log_mel_spectrogram(speech, n_mels=mel_size).permute(1, 0)
    return speech


class ModelWorker:
    def __init__(self, controller_addr, worker_addr,
                 worker_id, no_register,
                 model_path, model_base, model_name,
                 load_8bit, load_4bit, device, input_type, mel_size, is_lora,
                 checkpoint_path=None, config_path=None, use_flash_attn=False):
        self.controller_addr = controller_addr
        self.worker_addr = worker_addr
        self.worker_id = worker_id
        self.device = device
        self.model_name = model_name
        self.input_type = input_type
        self.mel_size = mel_size
        if checkpoint_path:
            config_path = config_path or "configs/stage_2.yaml"
            cfg = load_inference_cfg(Path(config_path).expanduser().resolve())
            module, loaded_device = load_module_from_checkpoint(
                Path(checkpoint_path).expanduser().resolve(),
                cfg,
                requested_device=self.device,
            )
            self.device = str(loaded_device)
            self.tokenizer = module.tokenizer
            self.model = module.model
            self.context_len = getattr(self.model.config, "max_sequence_length", 2048)
        else:
            self.tokenizer, self.model, self.context_len = load_pretrained_model(
                model_path, model_base, is_lora=is_lora, load_8bit=load_8bit,
                load_4bit=load_4bit, device=self.device,
                use_flash_attn=use_flash_attn)

        if not no_register:
            self.register_to_controller()
            self.heart_beat_thread = threading.Thread(
                target=heart_beat_worker, args=(self,), daemon=True)
            self.heart_beat_thread.start()

    def register_to_controller(self):
        logger.info("Register to controller")

        url = self.controller_addr + "/register_worker"
        data = {
            "worker_name": self.worker_addr,
            "check_heart_beat": True,
            "worker_status": self.get_status()
        }
        r = requests.post(url, json=data)
        assert r.status_code == 200

    def send_heart_beat(self):
        logger.info(f"Send heart beat. Models: {[self.model_name]}. "
                    f"Semaphore: {pretty_print_semaphore(model_semaphore)}. "
                    f"global_counter: {global_counter}")

        url = self.controller_addr + "/receive_heart_beat"

        while True:
            try:
                ret = requests.post(url, json={
                    "worker_name": self.worker_addr,
                    "queue_length": self.get_queue_length()}, timeout=5)
                exist = ret.json()["exist"]
                break
            except requests.exceptions.RequestException as e:
                logger.error(f"heart beat error: {e}")
            time.sleep(5)

        if not exist:
            self.register_to_controller()

    def get_queue_length(self):
        if model_semaphore is None:
            return 0
        else:
            return args.limit_model_concurrency - model_semaphore._value + (len(
                model_semaphore._waiters) if model_semaphore._waiters is not None else 0)

    def get_status(self):
        return {
            "model_names": [self.model_name],
            "speed": 1,
            "queue_length": self.get_queue_length(),
        }

    @torch.inference_mode()
    def generate_stream(self, params):
        tokenizer, model = self.tokenizer, self.model

        prompt = params["prompt"]
        ori_prompt = prompt
        audio = params.get("audio", None)
        if audio is not None and len(audio) > 0:
            speech = load_speech(audio, self.input_type, self.mel_size, self.model.config.speech_normalize)
            speech_length = torch.LongTensor([speech.shape[0]]).unsqueeze(0).to(self.device)
            speech_tensor = speech.unsqueeze(0).to(self.device, dtype=torch.float16)
            speech_args = {"speech": speech_tensor, "speech_lengths": speech_length}
        else:
            speech = None
            speech_args = {}

        temperature = float(params.get("temperature", 1.0))
        top_p = float(params.get("top_p", 1.0))
        max_new_tokens = min(int(params.get("max_new_tokens", 256)), 1024)
        stop_str = params.get("stop", None)
        do_sample = True if temperature > 0.001 else False

        input_ids = tokenizer_speech_token(prompt, tokenizer, return_tensors='pt').unsqueeze(0).to(self.device)
        streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True, timeout=15)

        # max_new_tokens = min(max_new_tokens, max_context_length - input_ids.shape[-1] - num_image_tokens)

        if max_new_tokens < 1:
            yield json.dumps({"text": ori_prompt + "Exceeds max token length. Please start a new conversation, thanks.", "error_code": 0}).encode() + b"\0"
            return

        thread = Thread(target=model.generate, kwargs=dict(
            inputs=input_ids,
            do_sample=do_sample,
            temperature=temperature,
            top_p=top_p,
            max_new_tokens=max_new_tokens,
            streamer=streamer,
            use_cache=True,
            **speech_args
        ))
        thread.start()

        generated_text = ori_prompt
        for new_text in streamer:
            generated_text += new_text
            if stop_str and generated_text.endswith(stop_str):
                generated_text = generated_text[:-len(stop_str)]
            yield json.dumps({"text": generated_text, "error_code": 0}).encode() + b"\0"

    def generate_stream_gate(self, params):
        try:
            for x in self.generate_stream(params):
                yield x
        except ValueError as e:
            print("Caught ValueError:", e)
            ret = {
                "text": server_error_msg,
                "error_code": 1,
            }
            yield json.dumps(ret).encode() + b"\0"
        except torch.cuda.CudaError as e:
            print("Caught torch.cuda.CudaError:", e)
            ret = {
                "text": server_error_msg,
                "error_code": 1,
            }
            yield json.dumps(ret).encode() + b"\0"
        except Exception as e:
            print("Caught Unknown Error", e)
            ret = {
                "text": server_error_msg,
                "error_code": 1,
            }
            yield json.dumps(ret).encode() + b"\0"


app = FastAPI()


def release_model_semaphore(fn=None):
    model_semaphore.release()
    if fn is not None:
        fn()


@app.post("/worker_generate_stream")
async def generate_stream(request: Request):
    global model_semaphore, global_counter
    global_counter += 1
    params = await request.json()

    if model_semaphore is None:
        model_semaphore = asyncio.Semaphore(args.limit_model_concurrency)
    await model_semaphore.acquire()
    worker.send_heart_beat()
    generator = worker.generate_stream_gate(params)
    background_tasks = BackgroundTasks()
    background_tasks.add_task(partial(release_model_semaphore, fn=worker.send_heart_beat))
    return StreamingResponse(generator, background=background_tasks)


@app.post("/worker_get_status")
async def get_status(request: Request):
    return worker.get_status()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="localhost")
    parser.add_argument("--port", type=int, default=21002)
    parser.add_argument("--worker-address", type=str,
        default="http://localhost:21002")
    parser.add_argument("--controller-address", type=str,
        default="http://localhost:21001")
    parser.add_argument("--model-path", type=str, default="facebook/opt-350m")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--model-name", type=str)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--limit-model-concurrency", type=int, default=5)
    parser.add_argument("--stream-interval", type=int, default=1)
    parser.add_argument("--no-register", action="store_true")
    parser.add_argument("--load-8bit", action="store_true")
    parser.add_argument("--load-4bit", action="store_true")
    parser.add_argument("--use-flash-attn", action="store_true")
    parser.add_argument("--input-type", type=str, default="mel")
    parser.add_argument("--mel-size", type=int, default=128)
    parser.add_argument("--is-lora", action="store_true", default=False)
    args = parser.parse_args()
    logger.info(f"args: {args}")

    worker = ModelWorker(args.controller_address,
                         args.worker_address,
                         worker_id,
                         args.no_register,
                         args.model_path,
                         args.model_base,
                         args.model_name,
                         args.load_8bit,
                         args.load_4bit,
                         args.device,
                         args.input_type,
                         args.mel_size,
                         args.is_lora,
                         use_flash_attn=args.use_flash_attn)
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")