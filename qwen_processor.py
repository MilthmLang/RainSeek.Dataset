from config import config
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
import logging
from s3 import S3ClientWrapper
from datetime import datetime
import json
import os

class ImageDescriptionGenerator:
    def __init__(self, model_path: str, device: str = "cuda"):
        self.logger = logging.getLogger(__name__)
        self.model_path = model_path
        self.device = device
        self.model = None
        self.processor = None

    def load_model_and_processor(self):
        """Load model and processor"""

        self.logger.info("Loading model and processor")
        self.logger.info(f"model: {self.model_path}")
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            self.model_path, torch_dtype="auto", device_map="auto",
        )
        self.processor = AutoProcessor.from_pretrained(self.model_path)
        self.logger.info("Model and processor loaded successfully")

    def generate_prompt(self, image_url: str) -> list:
        """Generate prompt for the image URL"""
        messages = [
        {
            "role": "system",
            "content": [
                {
                    "type": "text",
                    "text": """The following is a image, try to use tags to describe it for text-based search engine.

Your description should include key elements such as color, objects, actions, people, settings, atmosphere, etc.

Each tags as short as possible.

Think step by step and output the tags separated by commas at the end.

Example:
    
It's a beautiful spring day with cherry blossoms gently floating in the air.
A girl wearing a white dress is sitting on the grass, holding a bottle of drink and enjoying the sunshine.
In the background, many children are playing on the open grassland.

tags: spring, park, grass, cherry blossom, drink, children, outdoor, girl, green, white dress
    """}
            ],
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": image_url,
                },
            ],
        }
    ]
        return messages

    def generate(self, url: list):
        """Generate descriptions for the list of image URLs"""
        self.logger.info("Start generating descriptions for images")
        self.logger.info("url: " + url)

        messages = self.generate_prompt(url)

        # Preparation for inference
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        self.logger.info("apply_chat_template")

        image_inputs, video_inputs = process_vision_info(messages)
        self.logger.info("process_vision_info")

        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt"
        )
        inputs = inputs.to(self.device)

        # Inference: Generation of the output
        # Accelerator handles the model and inputs across multiple GPUs
        generated_ids = self.model.generate(**inputs,
            temperature=0.8,  
            top_p=0.95,
            do_sample=True,  
            max_new_tokens=4096*3,
            num_beams=5,
            no_repeat_ngram_size=2,  
            early_stopping=True,  
        )
        self.logger.info("generate")

        generated_ids_trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = self.processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )

        for item in output_text:
            self.logger.info(item)

        return output_text

if __name__ == '__main__':
    logger = logging.getLogger(__name__)

    generator = ImageDescriptionGenerator(config.qwen_model_path)
    generator.load_model_and_processor()
    
    s3 = S3ClientWrapper()
    keys = s3.list_files_with_prefix(prefix=config.s3_list_prefix)

    # start_time = datetime.now().strftime("%Y%m%d-%H%M%S")
    os.makedirs("output", exist_ok=True)

    for key in keys:
        try:
            url = s3.generate_presigned_url(key)
            output_text = generator.generate(url)

            # 提取 key 的最后一部分（文件名）
            filename = key.split("/")[-1].split(".")[0]
            output_filename = f"output/{filename}.json"

            # 保存为 JSON 文件
            with open(output_filename, "w", encoding="utf-8") as f:
                json.dump({"key": key, "filename": filename, "id": "illustration_"+filename, "description": output_text, "tags": output_text[0]}, f, ensure_ascii=False, indent=2)

        except Exception as e:
            logger.error(f"Error processing {key}: {e}")
            continue
