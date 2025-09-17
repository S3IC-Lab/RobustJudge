import openai
import time
import os
from loguru import logger
# API setting constants
API_MAX_RETRY = 16
API_RETRY_SLEEP = 10
API_ERROR_OUTPUT = "$ERROR$"

def set_openai_api(base_url, api_key):
    """
    Set the OpenAI API key and base URL.
    """
    for _ in range(API_MAX_RETRY):
        try:
            client = openai.OpenAI(
                base_url = base_url,
                api_key = api_key
                )
            break
        except openai.OpenAIError as e:
            print(type(e), e)
            time.sleep(API_RETRY_SLEEP)
    return client
    

def _chat_completion_openai(model, conv, temperature, max_tokens, base_url, api_key, num_return_sequences=1):
    # logger.info(f"conv: {conv}")
    client = set_openai_api(base_url, api_key)
    # logger.info(f"client: {client}")
    output = API_ERROR_OUTPUT
    for _ in range(API_MAX_RETRY):
        try:
            messages = conv.to_openai_api_messages()
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                n=num_return_sequences,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            if num_return_sequences == 1:
                output = response.choices[0].message.content
            else:
                output = [item.message.content.strip() for item in response.choices]
            # output = response
            break
        except openai.OpenAIError as e:
            print(type(e), e)
            time.sleep(API_RETRY_SLEEP)

    return output


def chat_completion_openai_demo(model_api, conv, temperature, max_new_token, base_url, api_key, num_return_sequences):
    return _chat_completion_openai(model_api, conv, temperature, max_new_token, base_url, api_key, num_return_sequences)

def chat_completion_openai_with_logprobs(model, conv, temperature, max_tokens, base_url, api_key, num_return_sequences=1,
                            top_p=1.0, logprobs=True, top_logprobs=20, seeds=0):
    client = set_openai_api(base_url, api_key)
    output = API_ERROR_OUTPUT
    for _ in range(API_MAX_RETRY):
        try:
            messages = conv.to_openai_api_messages()
            completion = client.chat.completions.create(
                model=model,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                logprobs=logprobs,
                top_logprobs=top_logprobs,
                seed=seeds
            )
            if num_return_sequences == 1:
                output = completion.choices[0].message.content
                top_logprobs = completion.choices[0].logprobs.content[0].top_logprobs
            else:
                output = [item.message.content.strip() for item in completion.choices]
                top_logprobs = [item.logprobs.content[0].top_logprobs for item in completion.choices]
            break
        except openai.OpenAIError as e:
            print(type(e), e)
            time.sleep(API_RETRY_SLEEP)

    return output, top_logprobs