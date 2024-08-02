from open_flamingo import create_model_and_transforms
from PIL import Image
import torch
import os
import argparse
from utils import evaluate_on_mmvetv2


class OpenFlamingo:
    def __init__(self, model_name='open-flamingo-9b'):
        if model_name == 'open-flamingo-9b':
            clip_vision_encoder_path="ViT-L-14"
            clip_vision_encoder_pretrained="openai"
            lang_encoder_path="anas-awadalla/mpt-7b"
            tokenizer_path="anas-awadalla/mpt-7b"
            cross_attn_every_n_layers=4
        self.model, self.image_processor, self.tokenizer = create_model_and_transforms(
            clip_vision_encoder_path=clip_vision_encoder_path,
            clip_vision_encoder_pretrained=clip_vision_encoder_pretrained,
            lang_encoder_path=lang_encoder_path,
            tokenizer_path=tokenizer_path,
            cross_attn_every_n_layers=cross_attn_every_n_layers,
            )
        
        self.tokenizer.padding_side = "left" # For generation padding tokens should be on the left

    def get_response(self, image_folder, prompt="What's in this image?") -> str:
        vision_x = []
        text_query = ""
        queries = prompt.split("<IMG>")
        for query in queries:
            query = query.strip()
            if query.endswith((".jpg", ".png", ".jpeg")):
                image_path = os.path.join(image_folder, query)
                image = Image.open(image_path).convert('RGB')
                vision_x.append(self.image_processor(image).unsqueeze(0))
                text_query += "<image>"
            else:
                text_query += query

        vision_x = torch.cat(vision_x, dim=0)
        vision_x = vision_x.unsqueeze(1).unsqueeze(0)

        lang_x = self.tokenizer(
            [text_query],
            return_tensors="pt",
        )

        generated_text = self.model.generate(
        vision_x=vision_x,
        lang_x=lang_x["input_ids"],
        attention_mask=lang_x["attention_mask"],
        max_new_tokens=512,
        num_beams=3,
        )

        response_text = self.tokenizer.decode(generated_text[0])
        return response_text.strip()
    

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
        default="open-flamingo-9b",
        help="Open Flamingo model name",
    )
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = arg_parser()

    model = OpenFlamingo(args.model_name)
    evaluate_on_mmvetv2(args, model)
