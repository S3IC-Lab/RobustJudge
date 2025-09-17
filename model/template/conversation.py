"""
Conversation prompt templates.
Adapted from https://github.com/lm-sys/FastChat/blob/main/fastchat/conversation.py.
"""
import base64
import dataclasses
from enum import auto, IntEnum
from io import BytesIO
import os
from typing import List, Any, Dict, Union, Tuple

class SeparatorStyle(IntEnum):
    """Separator styles."""

    ADD_COLON_SINGLE = auto()
    ADD_COLON_TWO = auto()
    ADD_NEW_LINE_SINGLE = auto()
    NO_COLON_SINGLE = auto()
    LLAMA2 = auto()
    LLAMA3 = auto()
    CHATML = auto()
    DEEPSEEK_CHAT = auto()
    CHATGLM = auto()
    CHATGLM3 = auto()
    FALCON_CHAT = auto()
    DEFAULT = auto()

IMAGE_PLACEHOLDER_STR = "$$<image>$$"

@dataclasses.dataclass
class Conversation:
    """A class that manages prompt templates and keeps all conversation history."""

    # The name of this template
    name: str
    # The template of the system prompt
    system_template: str = "{system_message}"
    # The system message
    system_message: str = ""
    system_message_vision: str = ""
    # The names of two roles
    roles: Tuple[str] = ("USER", "ASSISTANT")
    # All messages. Each item is (role, message).
    # Each message is either a string or a tuple of (string, List[image_url]).
    messages: List[List[str]] = ()
    # The number of few shot examples
    offset: int = 0
    # The separator style and configurations
    sep_style: SeparatorStyle = SeparatorStyle.ADD_COLON_SINGLE
    sep: str = "\n"
    sep2: str = None
    # Stop criteria (the default one is EOS token)
    stop_str: Union[str, List[str]] = None
    # Stops generation if meeting any token in this list
    stop_token_ids: List[int] = None
    # The maximum image size in megabytes that this model takes in. None means we do not resize the image.
    max_image_size_mb: int = None

    def get_prompt(self) -> str:
        """Get the prompt for generation."""
        system_prompt = self.system_template.format(system_message=self.system_message)
        if self.sep_style == SeparatorStyle.ADD_COLON_SINGLE:
            ret = system_prompt + self.sep
            for role, message in self.messages:
                if message:
                    if type(message) is tuple:
                        message, images = message
                        message = IMAGE_PLACEHOLDER_STR * len(images) + message
                    ret += role + ": " + message + self.sep
                else:
                    ret += role + ":"
            return ret
        elif self.sep_style == SeparatorStyle.ADD_COLON_TWO:
            seps = [self.sep, self.sep2]
            ret = system_prompt + seps[0]
            for i, (role, message) in enumerate(self.messages):
                if message:
                    if type(message) is tuple:
                        message, images = message
                        message = IMAGE_PLACEHOLDER_STR * len(images) + message
                    ret += role + ": " + message + seps[i % 2]
                else:
                    ret += role + ":"
            return ret
        elif self.sep_style == SeparatorStyle.ADD_NEW_LINE_SINGLE:
            ret = "" if system_prompt == "" else system_prompt + self.sep
            for role, message in self.messages:
                if message:
                    ret += role + "\n" + message + self.sep
                else:
                    ret += role + "\n"
            return ret
        elif self.sep_style == SeparatorStyle.NO_COLON_SINGLE:
            ret = system_prompt
            for role, message in self.messages:
                if message:
                    ret += role + message + self.sep
                else:
                    ret += role
            return ret
        elif self.sep_style == SeparatorStyle.LLAMA2:
            seps = [self.sep, self.sep2]
            if self.system_message:
                ret = system_prompt
            else:
                ret = "[INST] "
            for i, (role, message) in enumerate(self.messages):
                tag = self.roles[i % 2]
                if message:
                    if i == 0:
                        ret += message + " "
                    else:
                        ret += tag + " " + message + seps[i % 2]
                else:
                    ret += tag
            return ret
        elif self.sep_style == SeparatorStyle.LLAMA3:
            ret = "<|begin_of_text|>"
            if self.system_message:
                ret += system_prompt
            else:
                ret += ""
            for i, (role, message) in enumerate(self.messages):
                if message:
                    ret += f"<|start_header_id|>{role}<|end_header_id|>\n\n"
                    ret += f"{message.strip()}<|eot_id|>"
                else:
                    ret += f"<|start_header_id|>{role}<|end_header_id|>\n\n"
            return ret
        elif self.sep_style == SeparatorStyle.CHATML:
            ret = "" if system_prompt == "" else system_prompt + self.sep + "\n"
            for role, message in self.messages:
                if message:
                    if type(message) is tuple:
                        message, images = message
                        message = IMAGE_PLACEHOLDER_STR * len(images) + message
                    ret += role + "\n" + message + self.sep + "\n"
                else:
                    ret += role + "\n"
            return ret
        elif self.sep_style == SeparatorStyle.DEEPSEEK_CHAT:
            seps = [self.sep, self.sep2]
            ret = system_prompt
            for i, (role, message) in enumerate(self.messages):
                if message:
                    ret += role + ": " + message + seps[i % 2]
                else:
                    ret += role + ":"
            return ret
        elif self.sep_style == SeparatorStyle.CHATGLM:
            # source: https://huggingface.co/THUDM/chatglm-6b/blob/1d240ba371910e9282298d4592532d7f0f3e9f3e/modeling_chatglm.py#L1302-L1308
            # source2: https://huggingface.co/THUDM/chatglm2-6b/blob/e186c891cf64310ac66ef10a87e6635fa6c2a579/modeling_chatglm.py#L926
            round_add_n = 1 if self.name == "chatglm2" else 0
            if system_prompt:
                ret = system_prompt + self.sep
            else:
                ret = ""

            for i, (role, message) in enumerate(self.messages):
                if i % 2 == 0:
                    ret += f"[Round {i//2 + round_add_n}]{self.sep}"

                if message:
                    ret += f"{role}：{message}{self.sep}"
                else:
                    ret += f"{role}："
            return ret
        elif self.sep_style == SeparatorStyle.CHATGLM3:
            ret = ""
            if self.system_message:
                ret += system_prompt
            for role, message in self.messages:
                if message:
                    ret += role + "\n" + message
                else:
                    ret += role
            return ret
        elif self.sep_style == SeparatorStyle.FALCON_CHAT:
            ret = ""
            if self.system_message:
                ret += system_prompt + self.sep
            for role, message in self.messages:
                if message:
                    ret += role + ": " + message + self.sep
                else:
                    ret += role + ":"
            return ret
        elif self.sep_style == SeparatorStyle.DEFAULT:
            ret = system_prompt + "\n"
            for role, message in self.messages:
                if message:
                    if type(message) is tuple:
                        message, images = message
                    ret += role + ": " + message + "\n"
                else:
                    ret += role + ":"
            return ret
        else:
            raise ValueError(f"Invalid style: {self.sep_style}")

    def set_system_message(self, system_message: str):
        """Set the system message."""
        self.system_message = system_message

    def get_system_message(self, is_vision=False):
        """return the system message."""
        if is_vision and self.system_message_vision:
            return self.system_message_vision
        return self.system_message

    def append_message(self, role: str, message: str):
        """Append a new message."""
        self.messages.append([role, message])

    def update_last_message(self, message: str):
        """Update the last output.

        The last message is typically set to be None when constructing the prompt,
        so we need to update it in-place after getting the response from a model.
        """
        self.messages[-1][1] = message
    def to_openai_vision_api_messages(self, is_mistral=False):
        """Convert the conversation to OpenAI vision api completion format"""
        if self.system_message == "":
            ret = []
        else:
            ret = [
                {
                    "role": "system",
                    "content": self.system_message,
                }
            ]

        for i, (_, msg) in enumerate(self.messages[self.offset :]):
            if i % 2 == 0:
                if type(msg) is tuple:
                    content_list = [{"type": "text", "text": msg[0]}]
                    image_urls = msg[1]
                    for image in image_urls:
                        image_url = image.to_openai_image_format()
                        content = {}
                        if is_mistral:
                            content = {"type": "image_url", "image_url": image_url}
                        else:
                            content = {
                                "type": "image_url",
                                "image_url": {"url": image_url},
                            }
                        content_list.append(content)

                    ret.append({"role": "user", "content": content_list})
                else:
                    ret.append({"role": "user", "content": msg})
            else:
                if msg is not None:
                    ret.append(
                        {
                            "role": "assistant",
                            "content": msg,
                        }
                    )
        return ret

    def to_openai_api_messages(self):
        """Convert the conversation to OpenAI chat completion format."""
        if self.system_message == "":
            ret = []
        else:
            ret = [{"role": "system", "content": self.system_message}]

        for i, (_, msg) in enumerate(self.messages[self.offset :]):
            if i % 2 == 0:
                ret.append({"role": "user", "content": msg})
            else:
                if msg is not None:
                    ret.append({"role": "assistant", "content": msg})
        return ret

    def copy(self):
        return Conversation(
            name=self.name,
            system_template=self.system_template,
            system_message=self.system_message,
            system_message_vision=self.system_message_vision,
            roles=self.roles,
            messages=[[x, y] for x, y in self.messages],
            offset=self.offset,
            sep_style=self.sep_style,
            sep=self.sep,
            sep2=self.sep2,
            stop_str=self.stop_str,
            stop_token_ids=self.stop_token_ids,
            max_image_size_mb=self.max_image_size_mb,
        )

    def dict(self):
        return {
            "template_name": self.name,
            "system_message": self.system_message,
            "roles": self.roles,
            "messages": self.extract_text_and_image_hashes_from_messages(),
            "offset": self.offset,
        }


# A global registry for all conversation templates
conv_templates: Dict[str, Conversation] = {}

def register_conv_template(template: Conversation, override: bool = False):
    """Register a new conversation template."""
    if not override:
        assert (
            template.name not in conv_templates
        ), f"{template.name} has been registered."

    conv_templates[template.name] = template

def get_conv_template(name: str) -> Conversation:
    """Get a conversation template."""
    return conv_templates[name].copy()

# An empty template for raw conversation.
register_conv_template(
    Conversation(
        name="raw",
        system_message="",
        roles=("", ""),
        sep_style=SeparatorStyle.NO_COLON_SINGLE,
        sep="",
    )
)

# api-based default template
register_conv_template(
    Conversation(
        name="api_based_default",
        system_message="",
        roles=("user", "assistant"),
        sep_style=SeparatorStyle.DEFAULT,
        sep=None,
    )
)

# Vicuna v1.1 template
register_conv_template(
    Conversation(
        name="vicuna_v1.1",
        system_message="A chat between a curious user and an artificial intelligence assistant. "
        "The assistant gives helpful, detailed, and polite answers to the user's questions.",
        roles=("USER", "ASSISTANT"),
        sep_style=SeparatorStyle.ADD_COLON_TWO,
        sep=" ",
        sep2="</s>",
    )
)

# A template with a one-shot conversation example
register_conv_template(
    Conversation(
        name="one_shot",
        system_message="A chat between a curious human and an artificial intelligence assistant. "
        "The assistant gives helpful, detailed, and polite answers to the human's questions.",
        roles=("Human", "Assistant"),
        messages=(
            (
                "Human",
                "Got any creative ideas for a 10 year old’s birthday?",
            ),
            (
                "Assistant",
                """Of course! Here are some creative ideas for a 10-year-old's birthday party:
                1. Treasure Hunt: Organize a treasure hunt in your backyard or nearby park. Create clues and riddles for the kids to solve, leading them to hidden treasures and surprises.
                2. Science Party: Plan a science-themed party where kids can engage in fun and interactive experiments. You can set up different stations with activities like making slime, erupting volcanoes, or creating simple chemical reactions.
                3. Outdoor Movie Night: Set up a backyard movie night with a projector and a large screen or white sheet. Create a cozy seating area with blankets and pillows, and serve popcorn and snacks while the kids enjoy a favorite movie under the stars.
                4. DIY Crafts Party: Arrange a craft party where kids can unleash their creativity. Provide a variety of craft supplies like beads, paints, and fabrics, and let them create their own unique masterpieces to take home as party favors.
                5. Sports Olympics: Host a mini Olympics event with various sports and games. Set up different stations for activities like sack races, relay races, basketball shooting, and obstacle courses. Give out medals or certificates to the participants.
                6. Cooking Party: Have a cooking-themed party where the kids can prepare their own mini pizzas, cupcakes, or cookies. Provide toppings, frosting, and decorating supplies, and let them get hands-on in the kitchen.
                7. Superhero Training Camp: Create a superhero-themed party where the kids can engage in fun training activities. Set up an obstacle course, have them design their own superhero capes or masks, and organize superhero-themed games and challenges.
                8. Outdoor Adventure: Plan an outdoor adventure party at a local park or nature reserve. Arrange activities like hiking, nature scavenger hunts, or a picnic with games. Encourage exploration and appreciation for the outdoors.
                Remember to tailor the activities to the birthday child's interests and preferences. Have a great celebration!""",
            ),
        ),
        offset=2,
        sep_style=SeparatorStyle.ADD_COLON_SINGLE,
        sep="\n### ",
        stop_str="###",
    )
)

# Qwen-7b-chat template
# source: https://huggingface.co/Qwen/Qwen-7B-Chat/blob/main/qwen_generation_utils.py#L130
register_conv_template(
    Conversation(
        name="qwen-7b-chat",
        system_template="<|im_start|>system\n{system_message}",
        system_message="You are a helpful assistant.",
        roles=("<|im_start|>user", "<|im_start|>assistant"),
        sep_style=SeparatorStyle.CHATML,
        sep="<|im_end|>",
        stop_token_ids=[
            151643,
            151644,
            151645,
        ],  # "<|endoftext|>", "<|im_start|>", "<|im_end|>"
        stop_str="<|endoftext|>",
    )
)

# llama2 template
# reference: https://huggingface.co/blog/codellama#conversational-instructions
# reference: https://github.com/facebookresearch/llama/blob/1a240688810f8036049e8da36b073f63d2ac552c/llama/generation.py#L212
register_conv_template(
    Conversation(
        name="llama-2",
        system_template="[INST] <<SYS>>\n{system_message}\n<</SYS>>\n\n",
        roles=("[INST]", "[/INST]"),
        sep_style=SeparatorStyle.LLAMA2,
        sep=" ",
        sep2=" </s><s>",
    )
)

# llama3 template
# reference: https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct/blob/main/tokenizer_config.json
# reference: https://github.com/meta-llama/llama3/blob/0cee08ec68f4cfc0c89fe4a9366d82679aaa2a66/llama/tokenizer.py#L222
# LLAMA3-8B
register_conv_template(
    Conversation(
        name="llama-3",
        system_template="<|start_header_id|>system<|end_header_id|>\n\n{system_message}<|eot_id|>",
        roles=("user", "assistant"),
        sep_style=SeparatorStyle.LLAMA3,
        sep="",
        stop_str="<|eot_id|>",
        stop_token_ids=[128001, 128009],
    )
)

# llama3.1 template
register_conv_template(
    Conversation(
        name="meta-llama-3.1",
        system_message=(
            """Cutting Knowledge Date: December 2023
Today Date: {{currentDateTimev2}}"""
        ),
        roles=("user", "assistant"),
        sep_style=SeparatorStyle.DEFAULT,
        sep=None,
    )
)

# llama-3.1-sp template
register_conv_template(
    Conversation(
        name="meta-llama-3.1-sp",
        system_message=(
            """Cutting Knowledge Date: December 2023
Today Date: {{currentDateTimev2}}

Carefully read the user prompt. Your responses are comprehensive and easy to understand. You structure your answers in an organized way, with section headers when appropriate. You use consistent formatting in your responses. You follow user instructions. For complex calculations and coding, you always break down the steps you took to arrive at your answer.

Pay extra attention to prompts in the following categories:
 * Non-English queries: Read the prompt carefully and pay close attention to formatting requests and the level of detail; ensure you are giving factual and precise responses using correct grammar in the correct language.
 * Coding queries: You prioritize code organization and documentation. Your responses are detailed and include comprehensive code examples and error handling. Include comments to explain the code's purpose and behavior. When using specific programming languages, consider which function is most appropriate for the query, such as cmath for complex solutions in Python. Check for errors.
 * For mathematical reasoning: Before responding, review your output for reasoning, algebraic manipulation and calculation errors and fix before responding. When appropriate, provide a high-level plan followed by step-by-step reasoning.

Remember your instructions."""
        ),
        roles=("user", "assistant"),
        sep_style=SeparatorStyle.DEFAULT,
        sep=None,
    )
)

# Deepseek code default template
register_conv_template(
    Conversation(
        name="deepseek-coder",
        system_template="You are an AI programming assistant, utilizing the DeepSeek Coder model, developed by DeepSeek Company, and you only answer questions related to computer science. For politically sensitive questions, security and privacy issues, and other non-computer science questions, you will refuse to answer.",
        roles=("### Instruction:", "### Response:"),
        sep="\n",
        stop_str="<|EOT|>",
        sep_style=SeparatorStyle.ADD_NEW_LINE_SINGLE,
    )
)

# Deepseek-chat template
# reference: https://huggingface.co/deepseek-ai/deepseek-llm-67b-chat/blob/main/tokenizer_config.json
register_conv_template(
    Conversation(
        name="deepseek-chat",
        system_message="<｜begin▁of▁sentence｜>",  # must add a bos token before first message
        roles=("User", "Assistant"),
        sep_style=SeparatorStyle.DEEPSEEK_CHAT,
        sep="\n\n",
        sep2="<｜end▁of▁sentence｜>",
        stop_str="<｜end▁of▁sentence｜>",
    )
)

# ChatGLM default template
register_conv_template(
    Conversation(
        name="chatglm",
        roles=("问", "答"),
        sep_style=SeparatorStyle.CHATGLM,
        sep="\n",
    )
)

# ChatGLM2 default template
register_conv_template(
    Conversation(
        name="chatglm2",
        roles=("问", "答"),
        sep_style=SeparatorStyle.CHATGLM,
        sep="\n\n",
    )
)

# ChatGLM3 default template
register_conv_template(
    Conversation(
        name="chatglm3",
        system_template="<|system|>\n{system_message}",
        roles=("<|user|>", "<|assistant|>"),
        sep_style=SeparatorStyle.CHATGLM3,
        stop_token_ids=[
            64795,
            64797,
            2,
        ],  # "<|user|>", "<|observation|>", "</s>"
    )
)

# OpenChat 3.5 default template
register_conv_template(
    Conversation(
        name="openchat_3.5",
        roles=("GPT4 Correct User", "GPT4 Correct Assistant"),
        sep_style=SeparatorStyle.FALCON_CHAT,
        sep="<|end_of_turn|>",
    )
)

# ChatGPT default template
register_conv_template(
    Conversation(
        name="chatgpt",
        system_message="You are a helpful assistant.",
        roles=("user", "assistant"),
        sep_style=SeparatorStyle.DEFAULT,
        sep=None,
        max_image_size_mb=None,  # OpenAI does auto-resizing
    )
)

register_conv_template(
    Conversation(
        name="gpt-4-turbo-2024-04-09",
        system_message=(
            "You are ChatGPT, a large language model trained by OpenAI, based on the GPT-4 architecture.\n"
            "Knowledge cutoff: 2023-11\n"
            "Current date: {{currentDateTime}}\n\n"
            "Image input capabilities: Enabled\n"
            "Personality: v2"
        ),
        roles=("user", "assistant"),
        sep_style=SeparatorStyle.DEFAULT,
        sep=None,
    )
)

register_conv_template(
    Conversation(
        name="gpt-mini",
        system_message=(
            "You are ChatGPT, a large language model trained by OpenAI, based on the GPT-4 architecture.\n"
            "Current date: {{currentDateTime}}\n\n"
            "Image input capabilities: Enabled\n"
            "Personality: v2"
        ),
        roles=("user", "assistant"),
        sep_style=SeparatorStyle.DEFAULT,
        sep=None,
    )
)

# Mistral template
# source: https://docs.mistral.ai/llm/mistral-instruct-v0.1#chat-template
register_conv_template(
    Conversation(
        name="mistral",
        system_template="[INST] {system_message}\n",
        roles=("[INST]", "[/INST]"),
        sep_style=SeparatorStyle.LLAMA2,
        sep=" ",
        sep2="</s>",
    )
)
