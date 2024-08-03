import torch
from transformers import AutoModel, AutoTokenizer
import argparse

import os
import torch
import torchvision
from PIL import Image
from utils import evaluate_on_mmvetv2, process_images_for_question


def auto_configure_device_map(num_gpus):
    # visual_encoder 算4层
    # internlm_model.model.embed_tokens 占用1层
    # norm 和 lm_head 占用1层
    # transformer.layers 占用 32 层
    # 总共34层分配到num_gpus张卡上
    num_trans_layers = 32
    per_gpu_layers = 38 / num_gpus

    device_map = {
        "vit": 0,
        "vision_proj": 0,
        "model.tok_embeddings": 0,
        "model.norm": num_gpus - 1,
        "output": num_gpus - 1,
    }

    used = 3
    gpu_target = 0
    for i in range(num_trans_layers):
        if used >= per_gpu_layers:
            gpu_target += 1
            used = 0
        assert gpu_target < num_gpus
        device_map[f"model.layers.{i}"] = gpu_target
        used += 1

    return device_map


def model_gen_single_img(model, text, images, need_bos=True, padding=False):
    pt1 = 0
    embeds = []
    im_mask = []
    images = images
    images_loc = [0]
    for i, pts in enumerate(images_loc + [len(text)]):
        subtext = text[pt1:pts]
        if need_bos or len(subtext) > 0:
            text_embeds = model.encode_text(subtext, add_special_tokens=need_bos)
            embeds.append(text_embeds)
            im_mask.append(torch.zeros(text_embeds.shape[:2]).cuda())
            need_bos = False
        if i < len(images):
            try:
                image = Image.open(images[i]).convert("RGB")
            except:
                image = images[i].convert("RGB")
            if padding:
                image = __padding__(image)
            image = model.vis_processor(image).unsqueeze(0).half().cuda()
            image_embeds = model.encode_img(image)
            embeds.append(image_embeds)
            im_mask.append(torch.ones(image_embeds.shape[:2]).cuda())
        pt1 = pts
    embeds = torch.cat(embeds, dim=1)
    im_mask = torch.cat(im_mask, dim=1)
    im_mask = im_mask.bool()

    outputs = model.generate(
        inputs_embeds=embeds,
        im_mask=im_mask,
        temperature=1.0,
        max_new_tokens=4096,
        num_beams=3,
        do_sample=False,
        repetition_penalty=1.0,
    )

    output_token = outputs[0]
    if output_token[0] == 0 or output_token[0] == 1:
        output_token = output_token[1:]
    output_text = model.tokenizer.decode(output_token, add_special_tokens=False)
    output_text = output_text.split("[UNUSED_TOKEN_145]")[0].strip()
    return output_text


def model_gen_multi_img(model, text, images, need_bos=True, padding=False):
    embeds = []
    im_mask = []
    images = images
    for i, pts in enumerate(text):
        text_embeds = model.encode_text(
            pts, add_special_tokens=need_bos if i == 0 else False
        )
        embeds.append(text_embeds)
        im_mask.append(torch.zeros(text_embeds.shape[:2]).cuda())
        if i < len(images):
            assert os.path.exists(images[i])
            try:
                image = Image.open(images[i]).convert("RGB")
            except:
                image = images[i].convert("RGB")
            if padding:
                image = __padding__(image)
            image = model.vis_processor(image).unsqueeze(0).cuda()
            image_embeds = model.encode_img(image)
            embeds.append(image_embeds)
            im_mask.append(torch.ones(image_embeds.shape[:2]).cuda())
    embeds = torch.cat(embeds, dim=1)
    im_mask = torch.cat(im_mask, dim=1)
    im_mask = im_mask.bool()
    outputs = model.generate(
        inputs_embeds=embeds,
        im_mask=im_mask,
        temperature=1.0,
        max_new_tokens=4096,
        num_beams=3,
        do_sample=False,
        repetition_penalty=1.0,
    )
    output_token = outputs[0]
    if output_token[0] == 0 or output_token[0] == 1:
        output_token = output_token[1:]
    output_text = model.tokenizer.decode(output_token, add_special_tokens=False)
    output_text = output_text.split("[UNUSED_TOKEN_145]")[0].strip()
    return output_text


def __padding__(image):
    width, height = image.size
    tar = max(width, height)
    top_padding = int((tar - height) / 2)
    bottom_padding = tar - height - top_padding
    left_padding = int((tar - width) / 2)
    right_padding = tar - width - left_padding
    image = torchvision.transforms.functional.pad(
        image, [left_padding, top_padding, right_padding, bottom_padding]
    )
    return image


class InternLM_XComposer2_VL:
    def __init__(
        self,
        model_name="internlm/internlm-xcomposer2-vl-7b",
        image_first=False,
        system_message="You are a helpful assistant, dedicated to delivering comprehensive and meticulous responses.",
        chat_format=True,
    ):
        self.model = AutoModel.from_pretrained(
            model_name, trust_remote_code=True
        ).eval()

        if args.dtype == "fp16":
            self.model.half().cuda()
        elif args.dtype == "fp32":
            self.model.cuda()

        if args.num_gpus > 1:
            from accelerate import dispatch_model

            device_map = auto_configure_device_map(args.num_gpus)
            self.model = dispatch_model(self.model, device_map=device_map)

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name, trust_remote_code=True
        )
        self.model.tokenizer = self.tokenizer
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
        if args.combine_imgs:
            text_query = "".join(text_queries)
            text = "[UNUSED_TOKEN_146]system\n{}[UNUSED_TOKEN_145]\n[UNUSED_TOKEN_146]user\n{}Answer this question in detail.[UNUSED_TOKEN_145]\n[UNUSED_TOKEN_146]assistant\n".format(
                self.system_message, text_query
            )
            image = [process_images_for_question(images)]
            response = model_gen_single_img(
                model=self.model,
                text=text,
                images=image,
            )
        else:
            text_query = (
                [
                    "[UNUSED_TOKEN_146]system\n{}[UNUSED_TOKEN_145]\n[UNUSED_TOKEN_146]user\n".format(
                        self.system_message
                    )
                ]
                + text_queries
                + [
                    "{}Answer this question in detail.[UNUSED_TOKEN_145]\n[UNUSED_TOKEN_146]assistant\n"
                ]
            )
            with torch.cuda.amp.autocast():
                response = model_gen_multi_img(
                    model=self.model, text=text_query, images=images
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
    parser.add_argument(
        "--model_name",
        type=str,
        default="internlm/internlm-xcomposer2-vl-7b",
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
    parser.add_argument(
        "--combine_imgs",
        action="store_true",
        help="whether to use chat format",
    )
    parser.add_argument("--num_gpus", default=2, type=int)
    parser.add_argument("--dtype", default="fp16", type=str)
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = arg_parser()
    meta_instruction = """You are an AI assistant whose name is InternLM-XComposer (浦语·灵笔).
            - InternLM-XComposer (浦语·灵笔) is a multi-modality conversational language model that is developed by Shanghai AI Laboratory (上海人工智能实验室). It is designed to be helpful, honest, and harmless.
            - InternLM-XComposer (浦语·灵笔) can understand and communicate fluently in the language chosen by the user such as English and 中文.
            - InternLM-XComposer (浦语·灵笔) is capable of comprehending and articulating responses effectively based on the provided image."""

    model = InternLM_XComposer2_VL(
        args.model_name, image_first=args.image_first, system_message=meta_instruction
    )
    if args.image_first:
        args.model_name = args.model_name + "-image-first"
    if args.chat_format:
        args.model_name = args.model_name + "-chat-format"
    evaluate_on_mmvetv2(args, model)
