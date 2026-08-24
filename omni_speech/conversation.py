# Adopted from https://github.com/haotian-liu/LLaVA. Below is the original copyright:
#    Copyright 2023 Haotian Liu
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

import dataclasses
from enum import auto, Enum
from typing import List, Any, Union


class SeparatorStyle(Enum):
    """Different separator style."""
    LLAMA_3 = auto()


@dataclasses.dataclass
class Conversation:
    """A class that keeps all conversation history."""
    system: str
    roles: List[str]
    messages: List[List[str]]
    offset: int
    sep_style: SeparatorStyle = SeparatorStyle.LLAMA_3
    sep: str = "###"
    sep2: str = None
    version: str = "Unknown"

    tokenizer_id: str = ""
    tokenizer: Any = None
    # Stop criteria (the default one is EOS token)
    stop_str: Union[str, List[str]] = None
    # Stops generation if meeting any token in this list
    stop_token_ids: List[int] = None

    skip_next: bool = False

#* dataset structure (messages)looks like this:
#* [
#*     [
#*         role ,user,
#*         content, "Hello, how are you?"
#*     ],
#*     [
#*         role ,assistant,
#*         content, "I'm good, thank you!"
#*     ]
#* ]

    def get_prompt(self):
        messages = self.messages

        if self.sep_style != SeparatorStyle.LLAMA_3:
            raise ValueError(f"Invalid style: {self.sep_style}")

        wrap_sys = (
            lambda msg: (
                f"<|start_header_id|>system<|end_header_id|>\n\n{msg}<|eot_id|>"
                if len(msg) > 0
                else msg
            )
        )
        ret = "<|begin_of_text|>" + wrap_sys(self.system)
        for role, message in messages:
            if message:
                if isinstance(message, tuple):
                    message = message[0]
                ret += f"<|start_header_id|>{role}<|end_header_id|>\n\n"
                ret += message.strip() + self.sep2
            else:
                ret += f"<|start_header_id|>{role}<|end_header_id|>\n\n"
        return ret

## todo: maybe for adding new message
    def append_message(self, role, message):
        self.messages.append([role, message])
    
    def to_gradio_chatbot(self):
        ret = []
        for i, (role, msg) in enumerate(self.messages[self.offset:]):
            if i % 2 == 0:
                if type(msg) is tuple:
                    msg, speech = msg
                    ret.append([msg, None])
                else:
                    ret.append([msg, None])
            else:
                ret[-1][-1] = msg
        return ret

    def copy(self):
        return Conversation(
            system=self.system,
            roles=self.roles,
            messages=[[x, y] for x, y in self.messages],
            offset=self.offset,
            sep_style=self.sep_style,
            sep=self.sep,
            sep2=self.sep2,
            version=self.version)

    def dict(self):
        return {
            "system": self.system,
            "roles": self.roles,
            "messages": [
                [role, message[0] if isinstance(message, tuple) else message]
                for role, message in self.messages
            ],
            "offset": self.offset,
            "sep": self.sep,
            "sep2": self.sep2,
        }

conv_llama_3 = Conversation(
    system="आप एक सहायक भाषा और वाणी सहायक हैं। "
    "उपयोगकर्ता की बात समझकर प्राकृतिक भाषा में सहायता करें।",
    roles=("user", "assistant"),
    version="llama_v3",
    messages=[],
    offset=0,
    sep_style=SeparatorStyle.LLAMA_3,
    sep="",
    sep2="<|eot_id|>"
)

default_conversation = conv_llama_3
conv_templates = {
    "llama_3": conv_llama_3,
}


