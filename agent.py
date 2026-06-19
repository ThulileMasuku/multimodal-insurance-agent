import os
import torch
import pandas as pd
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info

class LocalVisionAgent:
    def __init__(self):
        print("🧠 Loading local Qwen2-VL-2B model onto GPU...")
        self.model = Qwen2VLForConditionalGeneration.from_pretrained(
            "Qwen/Qwen2-VL-2B-Instruct", 
            torch_dtype=torch.float16, 
            device_map="auto"
        )
        self.processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-2B-Instruct")
        print("✅ Local Model successfully loaded into VRAM!")

    def evaluate_claim(self, row, history_data, requirements_text):
        image_paths = row['image_paths'].split(';') if pd.notna(row['image_paths']) else []
        clean_paths = [p.strip() for p in image_paths if p.strip()]
        
        valid_images = []
        for p in clean_paths:
            full_path = p if p.startswith("hackerrank") else f"hackerrank-orchestrate-june26/dataset/{p}"
            if os.path.exists(full_path):
                valid_images.append(full_path)

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": f"Analyze claim for {row['claim_object']}. Dialogue: {row['user_claim']}."},
                ]
            }
        ]
        
        for img_path in valid_images:
            messages[0]["content"].append({"type": "image", "image": img_path})

        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self.processor(text=[text], images=image_inputs, videos=video_inputs, padding=True, return_tensors="pt")
        inputs = inputs.to("cuda")

        with torch.no_grad():
            generated_ids = self.model.generate(**inputs, max_new_tokens=200)
            generated_ids_trimmed = [out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)]
            output_text = self.processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True)[0]

        return output_text
