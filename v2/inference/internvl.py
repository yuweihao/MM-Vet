import random
import torch
from transformers import AutoModel, AutoTokenizer, CLIPImageProcessor
import os
from accelerate import init_empty_weights, infer_auto_device_map
import argparse
from utils import evaluate_on_mmvetv2, process_images_for_question


def disable_torch_init():
    """
    Disable the redundant torch default initialization to accelerate model creation.
    """
    import torch

    setattr(torch.nn.Linear, "reset_parameters", lambda self: None)
    setattr(torch.nn.LayerNorm, "reset_parameters", lambda self: None)


class Internvl:
    def __init__(
        self,
        model_name="OpenGVLab/InternVL-Chat-V1-2",  # OpenGVLab/InternVL-Chat-V1-2
        image_first=False,
        system_message="You are a helpful assistant, dedicated to delivering comprehensive and meticulous responses.",
        chat_format=True,
    ):
        random.seed(args.seed)
        if args.bf16:
            self.torch_type = torch.bfloat16
        else:
            self.torch_type = torch.float16
        self.model = (
            AutoModel.from_pretrained(
                model_name,
                torch_dtype=self.torch_type,
                low_cpu_mem_usage=True,
                trust_remote_code=True,
                device_map="auto",
            )
            .eval()
            .cuda()
        )

        self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
        self.temperature = 0.0
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
        image = image.resize((448, 448))
        image_processor = CLIPImageProcessor.from_pretrained(self.model_name)

        pixel_values = image_processor(images=image, return_tensors="pt").pixel_values
        pixel_values = pixel_values.to(self.torch_type).cuda()

        generation_config = dict(
            num_beams=1,
            max_new_tokens=1024,
            do_sample=True if self.temperature > 0 else False,
            temperature=self.temperature,
            length_penalty=1.0,
            repetition_penalty=1.2,
        )

        response = model.chat(
            self.tokenizer, pixel_values, text_query, generation_config
        )
        return response


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
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument(
        "--model_name",
        type=str,
        default="OpenGVLab/InternVL-Chat-V1-2",
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
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = arg_parser()

    model = Internvl(args.model_name, image_first=args.image_first)
    # model = None
    if args.image_first:
        args.model_name = args.model_name + "-image-first"
    if args.chat_format:
        args.model_name = args.model_name + "-chat-format"
    evaluate_on_mmvetv2(args, model)
