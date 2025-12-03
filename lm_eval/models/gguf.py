import logging
import time

import requests
import transformers
from requests.exceptions import RequestException
from tqdm import tqdm

from lm_eval.api.model import LM
from lm_eval.api.registry import register_model


logger = logging.getLogger(__name__)

def get_gen_prompt(tokenizer):
    dummy_with_gen_prompt = tokenizer.apply_chat_template([{"role": "user", "content": ""}], add_generation_prompt=True, tokenize=False)
    dummy_without_gen_prompt = tokenizer.apply_chat_template([{"role": "user", "content": ""}], add_generation_prompt=False, tokenize=False)

    gen_prompt = dummy_with_gen_prompt[len(dummy_without_gen_prompt):]

    return gen_prompt

def get_result(logprobs, continuation):
    is_greedy = True
    offsets = logprobs["text_offset"]
    tokens = logprobs["tokens"]
    tokens_logprobs = logprobs["token_logprobs"]
    top_logprobs = logprobs["top_logprobs"]

    # Remove first item if it's None because of Llama.cpp issue
    # https://github.com/EleutherAI/lm-evaluation-harness/issues/3385
    if tokens_logprobs and tokens_logprobs[0] is None:
        tokens = tokens[1:]
        tokens_logprobs = tokens_logprobs[1:]
        top_logprobs = top_logprobs[1:]

    idx = 0
    # while offsets[idx] < context_length:
        # idx += 1

    # continuation_logprobs = sum(tokens_logprobs[idx:-1])

    for idx in range(-1, -100, -1):
        if continuation in tokens[idx]:
            break

    continuation_logprobs = tokens_logprobs[idx]

    for i in range(idx, len(tokens)):
        token = tokens[i]
        top_tokens = top_logprobs[i]
        top_token = max(top_tokens.keys(), key=lambda x: top_tokens[x])
        if top_token != token:
            is_greedy = False
            break

    return continuation_logprobs, is_greedy


@register_model("gguf", "ggml")
class GGUFLM(LM):
    def __init__(self, base_url=None, gguf_model=None, hf_model=None, max_length=2048, **kwargs):
        super().__init__()
        self.base_url = base_url
        assert self.base_url, "must pass `base_url` to use GGUF LM!"
        self.model = gguf_model # TODO
        self.logprobs = 10
        self.temperature = 0.0
        self.max_length = max_length

        if hf_model:
            tokenizer = transformers.AutoTokenizer.from_pretrained(hf_model)
            self.hf_tokenizer = tokenizer
            self.gen_prompt = get_gen_prompt(tokenizer)
            logger.debug(f"Generation prompt section of the chat template: {self.gen_prompt}")
        else:
            self.hf_tokenizer = None
            self.gen_prompt = None


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
                }

                if echo:
                    request.update({"max_tokens": 0, "echo": True})

                if stop is not None:
                    request["stop"] = stop

                print(f"[gguf_completion] {request=}") # rm!
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

    def loglikelihood(self, requests, disable_tqdm: bool = False):
        if not requests:
            return []
        res = []
        for context, continuation in tqdm(
            [req.args for req in requests], disable=disable_tqdm
        ):

            if self.hf_tokenizer and self.hf_tokenizer.chat_template:
                messages = [{"role": "user", "content": context}]
                if continuation:
                    continuation = continuation.strip()
                    messages.append({"role": "assistant", "content": continuation})
                prompt = self.hf_tokenizer.apply_chat_template(messages, tokenize=False)
            else:
                prompt = context + (continuation or "")

            response = self.gguf_completion(prompt=prompt, echo=bool(continuation))

            if response and "choices" in response and response["choices"]:
                choice = response["choices"][0]
                logprobs = choice.get("logprobs")
                if (
                    logprobs
                    and "token_logprobs" in logprobs
                    and logprobs["token_logprobs"]
                ):
                    # logprob, is_greedy = get_result(logprobs, continuation)
                    logprob = self.get_cont_logprobs(prompt, continuation, logprobs)
                    is_greedy = False # TODO
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
            # TODO: apply chat template
            inp = request[0]
            request_args = request[1]
            until = request_args.get("until", ["</s>"])
            response = self.gguf_completion(context=inp, stop=until)
            if response and "choices" in response and response["choices"]:
                choice = response["choices"][0]
                if "text" in choice:
                    generated_text = choice["text"].strip()
                    res.append(generated_text)
                else:
                    logger.error(
                        f"Invalid response for greedy_until. Response: {response}"
                    )
                    res.append(None)  # Add default value in case of error
            else:
                logger.error(f"Invalid response for greedy_until. Response: {response}")
                res.append(None)  # Add default value in case of error
        return res

    def loglikelihood_rolling(self, requests, disable_tqdm: bool = False):
        raise NotImplementedError(
            "loglikelihood_rolling not yet supported for GGUF models"
        )

    def get_cont_logprobs(self, prompt: str, continuation: str, logprobs: dict) -> float:

        tokens = logprobs["tokens"]
        token_logprobs = logprobs["token_logprobs"]
        text_offsets = logprobs["text_offset"]

        for i in reversed(range(len(tokens))):
            if tokens[i] == "sí" or tokens[i] == "no": # TEST
                return token_logprobs[i] # TEST

        breakpoint()
        return 0 # TEST
