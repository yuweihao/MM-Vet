"""
This is a demo for using CogAgent and CogVLM in CLI
Make sure you have installed vicuna-7b-v1.5 tokenizer model (https://huggingface.co/lmsys/vicuna-7b-v1.5), full checkpoint of vicuna-7b-v1.5 LLM is not required.
In this demo, We us chat template, you can use others to replace such as 'vqa'.
Strongly suggest to use GPU with bfloat16 support, otherwise, it will be slow.
Mention that only one picture can be processed at one conversation, which means you can not replace or insert another picture during the conversation.
"""

import argparse
import torch
import json
import os
from PIL import Image
from transformers import AutoModelForCausalLM, LlamaTokenizer
import pandas as pd
from accelerate import (
    init_empty_weights,
    infer_auto_device_map,
    load_checkpoint_and_dispatch,
)
from utils import evaluate_on_mmvetv2, process_images_for_question


class CogAgent:
    def __init__(
        self,
        model_name="THUDM/cogagent-chat-hf",
        tokenizer_name="",
        image_first=False,
        system_message="You are a helpful assistant, dedicated to delivering comprehensive and meticulous responses.",
        chat_format=True,
    ):
        self.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

        self.tokenizer = LlamaTokenizer.from_pretrained(tokenizer_name)
        if args.bf16:
            self.torch_type = torch.bfloat16
        else:
            self.torch_type = torch.float16

        print(
            "========Use torch type as:{} with device:{}========\n\n".format(
                self.torch_type, self.DEVICE
            )
        )
        # tokenizer = LlamaTokenizer.from_pretrained('lmsys/vicuna-7b-v1.5')
        with init_empty_weights():
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=self.torch_type,
                low_cpu_mem_usage=True,
                trust_remote_code=True,
            )
        device_map = infer_auto_device_map(
            model,
            max_memory={0: "20GiB", 1: "20GiB"},
            no_split_module_classes=["CogAgentDecoderLayer"],
        )
        path = "~/.cache/huggingface/hub/models--THUDM--cogagent-chat-hf/snapshots/balabala"  # typical, '~/.cache/huggingface/hub/models--THUDM--cogagent-chat-hf/snapshots/balabala'
        model = load_checkpoint_and_dispatch(
            model,
            path,
            device_map=device_map,
        )
        self.model = model.eval()
        self.system_message = system_message
        self.chat_format = chat_format

    def get_response(self, image_folder, prompt="What's in this image?") -> str:
        images = []
        text_queries = []
        queries = prompt.split("<IMG>")
        for query in queries:
            query = query.strip()
            if query.endswith((".jpg", ".png", ".jpeg")):
                images.append(os.path.join(image_folder, query))
                text_queries.append("<IMAGE>")
            else:
                text_queries.append(query)
        text_query = "".join(text_queries)
        image = process_images_for_question(images).convert("RGB")
        input_by_model = self.model.build_conversation_input_ids(
            self.tokenizer, query=text_query, history=None, images=[image]
        )
        inputs = {
            "input_ids": input_by_model["input_ids"].unsqueeze(0).to(self.DEVICE),
            "token_type_ids": input_by_model["token_type_ids"]
            .unsqueeze(0)
            .to(self.DEVICE),
            "attention_mask": input_by_model["attention_mask"]
            .unsqueeze(0)
            .to(self.DEVICE),
            "images": [
                [input_by_model["images"][0].to(self.DEVICE).to(self.torch_type)]
            ],
        }
        if "cross_images" in input_by_model and input_by_model["cross_images"]:
            inputs["cross_images"] = [
                [input_by_model["cross_images"][0].to(self.DEVICE).to(self.torch_type)]
            ]

        # add any transformers params here.
        gen_kwargs = {"max_length": 2048, "temperature": 0.9, "do_sample": False}
        with torch.no_grad():
            outputs = self.model.generate(**inputs, **gen_kwargs)
            outputs = outputs[:, inputs["input_ids"].shape[1] :]
            response = self.tokenizer.decode(outputs[0])
            response = response.split("</s>")[0]
        output_text = response
        return output_text


def arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--quant", choices=[4], type=int, default=None, help="quantization bits"
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="THUDM/cogagent-chat-hf",
        help="pretrained ckpt",
    )
    parser.add_argument(
        "--local_tokenizer",
        type=str,
        default="lmsys/vicuna-7b-v1.5",
        help="tokenizer path",
    )
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument(
        "--mmvetv2_path",
        type=str,
        default="/path/to/mm-vet-v2",
        help="Download mm-vet.zip and `unzip mm-vet.zip` and change the path here",
    )
    parser.add_argument(
        "--result_path",
        type=str,
        default="results",
    )
    parser.add_argument(
        "--image_first",
        action="store_true",
        help="whether <image>text",
    )
    parser.add_argument(
        "--chat_format",
        action="store_true",
        help="whether to use chat format",
    )
    args = parser.parse_args()
    return args


# path = "/home/abc/.cache/huggingface/hub/models--THUDM--cogvlm-chat-hf/snapshots/e29dc3ba206d524bf8efbfc60d80fc4556ab0e3c"
if __name__ == "__main__":
    args = arg_parser()

    model = CogAgent(
        args.model_name, args.local_tokenizer, image_first=args.image_first
    )
    if args.image_first:
        args.model_name = args.model_name + "-image-first"
    if args.chat_format:
        args.model_name = args.model_name + "-chat-format"
    print(args)
    evaluate_on_mmvetv2(args, model)
