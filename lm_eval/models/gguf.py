import logging
import re
import time

import requests
import transformers
from requests.exceptions import RequestException
from tqdm import tqdm

from lm_eval.api.model import LM
from lm_eval.api.registry import register_model


logger = logging.getLogger(__name__)

def normalize(orig_str) -> bytes:
    norm_str = re.sub(r"\s+", " ", orig_str)
    norm_str_enc = norm_str.encode("ascii", errors="ignore") # Remove non-ascii characters because Llama.cpp returns them as �

    return norm_str_enc

def get_result(logprobs: dict, prompt_len: int, continuation: str):

    # Truncate to prompt_len to remove generated token(s) since we only care about the loglikelihood of the prompt tokens
    token_logprobs = logprobs["token_logprobs"][:prompt_len]
    top_logprobs = logprobs["top_logprobs"][:prompt_len]
    tokens = logprobs["tokens"][:prompt_len]

    is_greedy = True
    cont_start_idx = None

    cont_norm = normalize(continuation)

    # Calculate start position of the continuation by reconstructing it with the tokens from the end to the beginning
    for idx in range (len(tokens)-1, 0, -1):
        reconstructed_cont = "".join(tokens[idx:])
        reconst_norm = normalize(reconstructed_cont)
        if reconst_norm == cont_norm:
            cont_start_idx = idx
            break

        if len(reconst_norm) >= len(cont_norm):
            break

    if cont_start_idx is None:
        raise RuntimeError(f"Failed to identify continuation tokens in GGUF model response for continuation: \"{continuation}.\" Returned tokens list: {tokens}")

    # Start continuation from the previous token if possible (aligned with HF model implementation)
    if cont_start_idx != 0:
        cont_start_idx = cont_start_idx-1

    # Remove first item of the token lists if it's None
    if token_logprobs and token_logprobs[0] is None:
        tokens = tokens[1:]
        token_logprobs = token_logprobs[1:]
        top_logprobs = top_logprobs[1:]

    # Sum up the logprobs of the continuation tokens
    continuation_logprobs = sum(token_logprobs[cont_start_idx:])

    # Iterate over cont tokens to check if they were the preferred token in order to define is_greedy
    for idx in range(cont_start_idx, len(tokens)):
        token = tokens[idx]
        top_tokens = top_logprobs[idx]

        top_token = max(top_tokens.keys(), key=lambda x: top_tokens[x])
        if top_token != token:
            is_greedy = False
            break

    return continuation_logprobs, is_greedy


@register_model("gguf", "ggml")
class GGUFLM(LM):
    def __init__(self, base_url=None, gguf_model=None, hf_tokenizer=None, max_length=2048, **kwargs):
        super().__init__()
        self.base_url = base_url
        assert self.base_url, "must pass `base_url` to use GGUF LM!"
        self.model = gguf_model
        self.logprobs = 10
        self.temperature = 0.0
        self.max_length = max_length

        if hf_tokenizer:
            tokenizer = transformers.AutoTokenizer.from_pretrained(hf_tokenizer)
            self.hf_tokenizer = tokenizer
        else:
            self.hf_tokenizer = None

    @property
    def tokenizer_name(self) -> str:
        return self.hf_tokenizer

    def gguf_completion(
        self, prompt, echo=False, stop=None, retries=3, delay=5, **kwargs
    ):
        for _ in range(retries):
            try:
                request = {
                    "prompt": prompt,
                    "logprobs": self.logprobs,
                    "temperature": self.temperature,
                    "model": self.model,
                    "top_logprobs": 0,
                }

                if echo:
                    request.update({"max_tokens": 0, "echo": True})

                if stop is not None:
                    request["stop"] = stop

                response = requests.post(
                    f"{self.base_url}/v1/completions", json=request
                )
                response.raise_for_status()
                return response.json()

            except RequestException as e:
                logger.error(f"RequestException: {e}")
                time.sleep(delay)  # wait before retrying
        else:
            raise RuntimeError(
                f"Failed to get a valid response after {retries} retries."
            )

    def apply_chat_template(
        self, messages: list[dict[str, str]], add_generation_prompt: bool = True
    ) -> str:
        if not self.hf_tokenizer:
            raise ValueError("In order to use chat template (`--apply-chat-template`), --model_args must include valid `hf_tokenizer` path.")

        chat_templated = self.hf_tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=add_generation_prompt,
            continue_final_message=not add_generation_prompt,
        )

        return chat_templated

    def loglikelihood(self, requests, disable_tqdm: bool = False):
        if not requests:
            return []
        res = []
        for context, continuation in tqdm(
            [req.args for req in requests], disable=disable_tqdm
        ):
            echo = False

            if continuation:
                prompt = context + continuation
                echo = True

            response = self.gguf_completion(prompt=prompt, echo=echo)

            if response and "choices" in response and response["choices"] and "usage" in response and response["usage"]:
                choice = response["choices"][0]
                logprobs = choice.get("logprobs")
                if (
                    logprobs
                    and "token_logprobs" in logprobs
                    and logprobs["token_logprobs"]
                ):
                    prompt_len = response["usage"]["prompt_tokens"]
                    logprob, is_greedy = get_result(logprobs, prompt_len=prompt_len, continuation=continuation)
                    res.append((logprob, is_greedy))
                else:
                    logger.warning(
                        "Invalid logprobs data. Expected 'logprobs' to contain 'token_logprobs' list."
                    )
            else:
                logger.error(
                    f"Invalid response for loglikelihood. Response: {response}"
                )
                assert False
        return res

    def generate_until(self, requests, disable_tqdm: bool = False):
        if not requests:
            return []

        res = []
        for request in tqdm([req.args for req in requests], disable=disable_tqdm):
            inp = request[0]
            request_args = request[1]
            until = request_args.get("until", ["</s>"])
            response = self.gguf_completion(prompt=inp, stop=until)
            if response and "choices" in response and response["choices"]:
                choice = response["choices"][0]
                if "text" in choice:
                    generated_text = choice["text"].strip()
                    res.append(generated_text)
                else:
                    logger.error(
                        f"Invalid response for greedy_until. Response: {response}"
                    )
                    res.append(None)
            else:
                logger.error(f"Invalid response for greedy_until. Response: {response}")
                res.append(None)
        return res

    def loglikelihood_rolling(self, requests, disable_tqdm: bool = False):
        raise NotImplementedError(
            "loglikelihood_rolling not yet supported for GGUF models"
        )

