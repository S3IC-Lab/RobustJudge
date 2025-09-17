import time
from loguru import logger
from openai import OpenAI

client = OpenAI(base_url="...", 
                api_key="...")

def generate_text(*args, **kwargs):
    for _ in range(30):
        try:
            response = client.chat.completions.create(*args, **kwargs)
            return response
        except Exception as e:
            logger.info(e)
            time.sleep(30)
    exit(-1)
