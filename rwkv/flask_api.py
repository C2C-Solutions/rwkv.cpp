import argparse
import copy
import json
import os
import pathlib
from typing import Dict, List, Optional

import tokenizers
import torch
from flask import Flask, jsonify, request
from flask_sock import Sock

import rwkv_cpp_model
import rwkv_cpp_shared_library
import sampling

END_OF_LINE_TOKEN: int = 187
DOUBLE_END_OF_LINE_TOKEN: int = 535
END_OF_TEXT_TOKEN: int = 0
LANGUAGE: str = "English"
PROMPT_TYPE: str = "Chat"

parser = argparse.ArgumentParser(
    description="Provide terminal-based chat interface for RWKV model"
)
parser.add_argument("model_path", help="Path to RWKV model in ggml format")
args = parser.parse_args()

script_dir: pathlib.Path = pathlib.Path(os.path.abspath(__file__)).parent

print("Loading 20B tokenizer")
tokenizer_path = script_dir / "20B_tokenizer.json"
tokenizer = tokenizers.Tokenizer.from_file(str(tokenizer_path))

library = rwkv_cpp_shared_library.load_rwkv_shared_library()
print(f"System info: {library.rwkv_get_system_info_string()}")

print("Loading RWKV model")
model = rwkv_cpp_model.RWKVModel(library, args.model_path)

processed_tokens: List[int] = []
logits: Optional[torch.Tensor] = None
state: Optional[torch.Tensor] = None


def process_tokens(_tokens: List[int], new_line_logit_bias: float = 0.0) -> None:
    global processed_tokens, logits, state

    processed_tokens = _tokens

    for _token in _tokens:
        logits, state = model.eval(_token, state, state, logits)

    logits[END_OF_LINE_TOKEN] += new_line_logit_bias


state_by_thread: Dict[str, Dict] = {}


def save_thread_state(_thread: str) -> None:
    state_by_thread[_thread] = {
        "tokens": copy.deepcopy(processed_tokens),
        "logits": copy.deepcopy(logits),
        "state": copy.deepcopy(state),
    }


def load_thread_state(_thread: str) -> None:
    global processed_tokens, logits, state

    thread_state = state_by_thread[_thread]

    processed_tokens = copy.deepcopy(thread_state["tokens"])
    logits = copy.deepcopy(thread_state["logits"])
    state = copy.deepcopy(thread_state["state"])


def split_last_end_of_line(tokens):
    if len(tokens) > 0 and tokens[-1] == DOUBLE_END_OF_LINE_TOKEN:
        tokens = tokens[:-1] + [END_OF_LINE_TOKEN, END_OF_LINE_TOKEN]

    return tokens


save_thread_state("chat")

app = Flask(__name__)
sock = Sock(app)


@app.route("/api/v1/model")
def api():
    return jsonify({"result": "rwkv"})


@sock.route("/api/v1/stream")
def generate_stream(ws):
    content = json.loads(ws.receive())

    MAX_GENERATION_LENGTH: int = content["max_new_tokens"]
    TEMPERATURE: float = content["temperature"]
    TOP_P: float = content["top_p"]
    PRESENCE_PENALTY: float = 0.5
    FREQUENCY_PENALTY: float = 0.5

    stopping = content["stopping_strings"]

    for i in range(len(stopping)):
        stopping[i] = stopping[i].replace("\n", "")

    prompt = content["prompt"]

    msg = prompt.replace("\\n", "\n").strip()

    temperature = TEMPERATURE
    top_p = TOP_P

    if "-temp=" in msg:
        temperature = float(msg.split("-temp=")[1].split(" ")[0])

        msg = msg.replace("-temp=" + f"{temperature:g}", "")

        if temperature <= 0.2:
            temperature = 0.2

        if temperature >= 5:
            temperature = 5

    if "-top_p=" in msg:
        top_p = float(msg.split("-top_p=")[1].split(" ")[0])

        msg = msg.replace("-top_p=" + f"{top_p:g}", "")

        if top_p <= 0:
            top_p = 0

    # + reset --> reset chat
    if msg == "+reset":
        load_thread_state("chat_init")
        save_thread_state("chat")
        return
    elif (
        msg[:5].lower() == "+gen "
        or msg[:3].lower() == "+i "
        or msg[:4].lower() == "+qa "
        or msg[:4].lower() == "+qq "
        or msg.lower() == "+++"
        or msg.lower() == "++"
    ):
        # ++ --> retry last free generation (only for +gen / +i)
        if msg.lower() == "++":
            try:
                load_thread_state("gen_0")
            except Exception as e:
                print(e)
                return
        thread = "gen_1"
    else:
        load_thread_state("chat")
        process_tokens(tokenizer.encode(prompt).ids, new_line_logit_bias=-999999999)
        save_thread_state("chat_pre")

        thread = "chat"

        # Print bot response

    start_index: int = len(processed_tokens)
    accumulated_tokens: List[int] = []
    token_counts: Dict[int, int] = {}

    result_string = ""

    generating = True

    for i in range(MAX_GENERATION_LENGTH):
        if not generating:
            break

        for n in token_counts:
            logits[n] -= PRESENCE_PENALTY + token_counts[n] * FREQUENCY_PENALTY

        token: int = sampling.sample_logits(logits, temperature, top_p)

        if token == END_OF_TEXT_TOKEN:
            print()
            break

        if token not in token_counts:
            token_counts[token] = 1
        else:
            token_counts[token] += 1

        process_tokens([token])

        # Avoid UTF-8 display issues
        accumulated_tokens += [token]

        decoded: str = tokenizer.decode(accumulated_tokens)

        if "\uFFFD" not in decoded:
            print(decoded, end="")
            result_string += decoded
            for stopper in stopping:
                if result_string.rfind(stopper) != -1:
                    result_string = result_string.replace(stopper, "")
                    generating = False
                    break
            accumulated_tokens = []
            ws.send({"event": "text_stream", "text": decoded})

        if thread == "chat":
            if "\n\n" in tokenizer.decode(processed_tokens[start_index:]):
                break

        if i == MAX_GENERATION_LENGTH - 1:
            print()

    save_thread_state(thread)

    ws.send({"event": "stream_end"})

    ws.close()


@app.route("/api/v1/generate", methods=["GET", "POST"])
def generate():
    content = request.json

    MAX_GENERATION_LENGTH: int = content["max_new_tokens"]
    TEMPERATURE: float = content["temperature"]
    TOP_P: float = content["top_p"]
    PRESENCE_PENALTY: float = 0.5
    FREQUENCY_PENALTY: float = 0.5

    stopping = content["stopping_strings"]

    for i in range(len(stopping)):
        stopping[i] = stopping[i].replace("\n", "")

    prompt = content["prompt"]

    msg = prompt.replace("\\n", "\n").strip()

    temperature = TEMPERATURE
    top_p = TOP_P

    if "-temp=" in msg:
        temperature = float(msg.split("-temp=")[1].split(" ")[0])

        msg = msg.replace("-temp=" + f"{temperature:g}", "")

        if temperature <= 0.2:
            temperature = 0.2

        if temperature >= 5:
            temperature = 5

    if "-top_p=" in msg:
        top_p = float(msg.split("-top_p=")[1].split(" ")[0])

        msg = msg.replace("-top_p=" + f"{top_p:g}", "")

        if top_p <= 0:
            top_p = 0

    # + reset --> reset chat
    if msg == "+reset":
        load_thread_state("chat_init")
        save_thread_state("chat")
        return
    elif (
        msg[:5].lower() == "+gen "
        or msg[:3].lower() == "+i "
        or msg[:4].lower() == "+qa "
        or msg[:4].lower() == "+qq "
        or msg.lower() == "+++"
        or msg.lower() == "++"
    ):
        # ++ --> retry last free generation (only for +gen / +i)
        if msg.lower() == "++":
            try:
                load_thread_state("gen_0")
            except Exception as e:
                print(e)
                return
        thread = "gen_1"
    else:
        load_thread_state("chat")
        process_tokens(tokenizer.encode(prompt).ids, new_line_logit_bias=-999999999)
        save_thread_state("chat_pre")

        thread = "chat"

        # Print bot response

    start_index: int = len(processed_tokens)
    accumulated_tokens: List[int] = []
    token_counts: Dict[int, int] = {}

    result_string = ""

    generating = True

    for i in range(MAX_GENERATION_LENGTH):
        if not generating:
            break

        for n in token_counts:
            logits[n] -= PRESENCE_PENALTY + token_counts[n] * FREQUENCY_PENALTY

        token: int = sampling.sample_logits(logits, temperature, top_p)

        if token == END_OF_TEXT_TOKEN:
            print()
            break

        if token not in token_counts:
            token_counts[token] = 1
        else:
            token_counts[token] += 1

        process_tokens([token])

        # Avoid UTF-8 display issues
        accumulated_tokens += [token]

        decoded: str = tokenizer.decode(accumulated_tokens)

        if "\uFFFD" not in decoded:
            print(decoded, end="")
            result_string += decoded
            for stopper in stopping:
                if result_string.rfind(stopper) != -1:
                    result_string = result_string.replace(stopper, "")
                    generating = False
                    break
            accumulated_tokens = []

        if thread == "chat":
            if "\n\n" in tokenizer.decode(processed_tokens[start_index:]):
                break

        if i == MAX_GENERATION_LENGTH - 1:
            print()

    save_thread_state(thread)

    return jsonify({"results": [{"text": result_string}]})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=1111, debug=False)
