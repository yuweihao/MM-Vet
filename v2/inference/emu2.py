from PIL import Image
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from accelerate import (
    init_empty_weights,
    infer_auto_device_map,
    load_checkpoint_and_dispatch,
)
import os
import argparse
from utils import evaluate_on_mmvetv2


class Emu2:
    def __init__(
        self,
        model_name="BAAI/Emu2-Chat",
        image_first=False,
        system_message="You are a helpful assistant, dedicated to delivering comprehensive and meticulous responses.",
        chat_format=True,
    ):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)  # "BAAI/Emu2-Chat"
        # self.model = AutoModelForCausalLM.from_pretrained(
        #     model_name,
        #     torch_dtype=torch.bfloat16,
        #     low_cpu_mem_usage=True,
        #     trust_remote_code=True).to('cuda').eval()
        with init_empty_weights():
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.bfloat16,
                low_cpu_mem_usage=True,
                trust_remote_code=True,
            )
        device_map = infer_auto_device_map(
            model,
            max_memory={0: "16GIB", 1: "20GIB", 2: "20GIB", 3: "20GIB"},
            no_split_module_classes=["Block", "LlamaDecoderLayer"],
        )
        device_map["model.decoder.lm.lm_head"] = 0
        self.image_first = image_first
        self.model = load_checkpoint_and_dispatch(
            model,
            "/home/abc/.cache/huggingface/hub/models--BAAI--Emu2-Chat/snapshots/20ea30b04f8fee599cf97535e655c200df728501",
            device_map=device_map,
        ).eval()
        self.system_message = system_message
        self.chat_format = chat_format

    def get_response(self, image_folder, prompt="What's in this image?") -> str:
        images = []
        text_queries = []
        queries = prompt.split("<IMG>")
        for query in queries:
            query = query.strip()
            if query.endswith((".jpg", ".png", ".jpeg")):
                image_path = os.path.join(image_folder, query)
                images.append(Image.open(image_path).convert("RGB"))
                text_queries.append("[<IMG_PLH>]")
            else:
                text_queries.append(query)

        if self.image_first:
            for i in range(1, len(text_queries)):
                if text_queries[i] == "[<IMG_PLH>]" and (
                    text_queries[i - 1] != "[<IMG_PLH>]"
                ):
                    tmp = text_queries[i - 1]
                    text_queries[i - 1] = text_queries[i]
                    text_queries[i] = tmp
        text_query = "".join(text_queries)
        if self.chat_format:
            text_query = f"{self.system_message} [USER]: {text_query} [ASSISTANT]:"
        print(text_query)
        inputs = self.model.build_input_ids(
            text=[text_query], tokenizer=self.tokenizer, image=images
        )

        with torch.no_grad():
            outputs = self.model.generate(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                image=inputs["image"].to(torch.bfloat16),
                max_new_tokens=512,
                length_penalty=-1,
            )
        output_text = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
        return output_text


def arg_parser():
    parser = argparse.ArgumentParser()
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
        "--model_name",
        type=str,
        default="BAAI/Emu2-Chat",
        help="pretrained ckpt",
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


if __name__ == "__main__":
    args = arg_parser()

    model = Emu2(args.model_name, image_first=args.image_first)
    if args.image_first:
        args.model_name = args.model_name + "-image-first"
    if args.chat_format:
        args.model_name = args.model_name + "-chat-format"
    evaluate_on_mmvetv2(args, model)
