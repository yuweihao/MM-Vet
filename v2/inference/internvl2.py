import math
import random
import torch
from transformers import AutoModel, AutoTokenizer, CLIPImageProcessor
import os
from accelerate import init_empty_weights, infer_auto_device_map
import argparse
from utils import evaluate_on_mmvetv2, process_images_for_question
import torchvision.transforms as T
from PIL import Image, ImageDraw
from torchvision.transforms.functional import InterpolationMode

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def split_model(model_name):
    device_map = {}
    world_size = torch.cuda.device_count()
    num_layers = {
        "InternVL2-1B": 24,
        "InternVL2-2B": 24,
        "InternVL2-4B": 32,
        "InternVL2-8B": 32,
        "InternVL2-26B": 48,
        "InternVL2-40B": 60,
        "InternVL2-Llama3-76B": 80,
    }[model_name]
    # Since the first GPU will be used for ViT, treat it as half a GPU.
    num_layers_per_gpu = math.ceil(num_layers / (world_size - 0.5))
    num_layers_per_gpu = [num_layers_per_gpu] * world_size
    num_layers_per_gpu[0] = math.ceil(num_layers_per_gpu[0] * 0.5)
    layer_cnt = 0
    for i, num_layer in enumerate(num_layers_per_gpu):
        for j in range(num_layer):
            device_map[f"language_model.model.layers.{layer_cnt}"] = i
            layer_cnt += 1
    device_map["vision_model"] = 0
    device_map["mlp1"] = 0
    device_map["language_model.model.tok_embeddings"] = 0
    device_map["language_model.model.embed_tokens"] = 0
    device_map["language_model.output"] = 0
    device_map["language_model.model.norm"] = 0
    device_map["language_model.lm_head"] = 0
    device_map[f"language_model.model.layers.{num_layers - 1}"] = 0

    return device_map


def build_transform(input_size):
    MEAN, STD = IMAGENET_MEAN, IMAGENET_STD
    transform = T.Compose(
        [
            T.Lambda(lambda img: img.convert("RGB") if img.mode != "RGB" else img),
            T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
            T.ToTensor(),
            T.Normalize(mean=MEAN, std=STD),
        ]
    )
    return transform


def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    best_ratio_diff = float("inf")
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio


def dynamic_preprocess(
    image, min_num=1, max_num=6, image_size=448, use_thumbnail=False
):
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height

    # calculate the existing image aspect ratio
    target_ratios = set(
        (i, j)
        for n in range(min_num, max_num + 1)
        for i in range(1, n + 1)
        for j in range(1, n + 1)
        if i * j <= max_num and i * j >= min_num
    )
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

    # find the closest aspect ratio to the target
    target_aspect_ratio = find_closest_aspect_ratio(
        aspect_ratio, target_ratios, orig_width, orig_height, image_size
    )

    # calculate the target width and height
    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

    # resize the image
    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size,
        )
        # split the image
        split_img = resized_img.crop(box)
        processed_images.append(split_img)
    assert len(processed_images) == blocks
    if use_thumbnail and len(processed_images) != 1:
        thumbnail_img = image.resize((image_size, image_size))
        processed_images.append(thumbnail_img)
    return processed_images


def load_image(image_file, input_size=448, max_num=6):
    if isinstance(image_file, str):
        image = Image.open(image_file).convert("RGB")
    else:
        image = image_file
    transform = build_transform(input_size=input_size)
    images = dynamic_preprocess(
        image, image_size=input_size, use_thumbnail=True, max_num=max_num
    )
    pixel_values = [transform(image) for image in images]
    pixel_values = torch.stack(pixel_values)
    return pixel_values


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
        model_name="OpenGVLab/InternVL2-40B",  # OpenGVLab/InternVL-Chat-V1-5 OpenGVLab/InternVL2-40B OpenGVLab/InternVL2-Llama3-76B
        image_first=False,
        system_message="You are a helpful assistant, dedicated to delivering comprehensive and meticulous responses.",
        chat_format=True,
    ):
        random.seed(args.seed)
        if args.bf16:
            self.torch_type = torch.bfloat16
        else:
            self.torch_type = torch.float16
        os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
        try:
            self.model = AutoModel.from_pretrained(
                pretrained_model_name_or_path=model_name,
                torch_dtype=self.torch_type,
                low_cpu_mem_usage=True,
                trust_remote_code=True,
                device_map="auto",
            ).eval()
        except Exception:
            device_map = split_model(
                model_name.split("/")[-1]
            )  # "InternVL2-Llama3-76B"
            self.model = AutoModel.from_pretrained(
                model_name,
                torch_dtype=self.torch_type,
                low_cpu_mem_usage=True,
                trust_remote_code=True,
                device_map=device_map,
            ).eval()
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name, use_fast=False, trust_remote_code=True
        )

    def get_response(self, image_folder, prompt="What's in this image?") -> str:
        images = []
        text_queries = []
        queries = prompt.split("<IMG>")
        pixel_values = []
        num_patches_list = []
        for query in queries:
            query = query.strip()
            if query.endswith((".jpg", ".png", ".jpeg")):
                images.append(os.path.join(image_folder, query))
                pixel_values.append(
                    load_image(os.path.join(image_folder, query), max_num=6)
                    .to(self.torch_type)
                    .cuda()
                )
                num_patches_list.append(pixel_values[-1].size(0))
            else:
                text_queries.append(query)
        text_query = "".join(text_queries)
        if args.unique:
            question = ""
            for i in range(len(pixel_values)):
                idx = i + 1
                question += f"Image-{idx}: <image>\n"
            question += text_query
        else:
            question = f"<image>\n{text_query}"
        pixel_values = torch.cat(pixel_values, dim=0)
        generation_config = dict(
            num_beams=1,
            max_new_tokens=512,
            min_new_tokens=1,
            do_sample=True if args.temperature > 0 else False,
            temperature=args.temperature,
            length_penalty=1.0,
            repetition_penalty=1.2,
        )
        try:
            response, history = model.chat(
                self.tokenizer,
                pixel_values,
                question,
                generation_config,
                num_patches_list=num_patches_list if args.unique else None,
                history=None,
                return_history=True,
            )
        except Exception as e:
            combined_images = process_images_for_question(images).convert("RGB")
            pixel_values = (
                load_image(combined_images, max_num=6).to(self.torch_type).cuda()
            )
            response, history = model.chat(
                self.tokenizer,
                pixel_values,
                question,
                generation_config,
                history=None,
                return_history=True,
            )
            print(f"found error: {e}, combine images to save space")
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
        default="OpenGVLab/InternVL2-40B",
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
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--unique", action="store_true")

    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = arg_parser()

    model = Internvl(args.model_name, image_first=args.image_first)
    if args.image_first:
        args.model_name = args.model_name + "-image-first"
    if args.chat_format:
        args.model_name = args.model_name + "-chat-format"
    evaluate_on_mmvetv2(args, model)
