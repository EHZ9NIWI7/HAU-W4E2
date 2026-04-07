import re

from qwen_vl_utils import process_vision_info
from transformers import (AutoProcessor, Qwen2_5_VLForConditionalGeneration,
                          Qwen2VLForConditionalGeneration,
                          Qwen3OmniMoeForConditionalGeneration,
                          Qwen3OmniMoeProcessor,
                          Qwen3VLMoeForConditionalGeneration)


class Qwen(object):
    def __init__(self, model_path, **kwargs):
        self.model_path = model_path
        self.dtype = "auto"
        self.max_frames = 128

        self.init_model()
    
    def init_model(self):
        register = {
            "Qwen2-VL": {
                "model": lambda: Qwen2VLForConditionalGeneration.from_pretrained(
                    self.model_path, dtype=self.dtype, device_map="auto", trust_remote_code=True, attn_implementation="flash_attention_2"),
                "processor": lambda: AutoProcessor.from_pretrained(self.model_path, use_fast=True, trust_remote_code=True)
            }, 
            
            "Qwen2.5-VL": {
                "model": lambda: Qwen2_5_VLForConditionalGeneration.from_pretrained(
                    self.model_path, dtype=self.dtype, device_map="auto", trust_remote_code=True, attn_implementation="flash_attention_2"),
                "processor": lambda: AutoProcessor.from_pretrained(self.model_path, use_fast=True, trust_remote_code=True)
            }, 

            "Qwen3-VL": {
                "model": lambda: Qwen3VLMoeForConditionalGeneration.from_pretrained(
                    self.model_path, dtype=self.dtype, device_map="auto", trust_remote_code=True, attn_implementation="flash_attention_2"),
                "processor": lambda: AutoProcessor.from_pretrained(self.model_path, use_fast=True, trust_remote_code=True)
            },
            
            "Qwen3-Omni": {
                "model": lambda: Qwen3OmniMoeForConditionalGeneration.from_pretrained(
                    self.model_path, dtype=self.dtype, device_map="auto", trust_remote_code=True, attn_implementation="flash_attention_2"),
                "processor": lambda: Qwen3OmniMoeProcessor.from_pretrained(self.model_path, trust_remote_code=True),
            },
            
        }

        for n in register.keys():
            if n in self.model_path:
                for k, v in register[n].items():
                    setattr(self, k, v())
                break
            
        self.model.disable_talker() if "Qwen3-Omni" in self.model_path else None
    
    def converse(self, video_path, question):
        messages = []
        messages.append({
            "role": "system", 
            "content": [{"type": "text", "text": "You are an intelligent agent capable of understanding and analyzing human actions.\n"}]
        })
        messages.append({
            "role": "user", 
            "content": [
                {"type": "text", "text": question}, 
                {
                    "type": "video", 
                    "video": video_path,
                    "min_frames": 16, 
                    "max_frames": self.max_frames,
                    "fps": 2.0,
                    "max_pixels": 360 * 420
                    # "min_pixels": 4 * 32 * 32,
                    # "max_pixels": 256 * 32 * 32,
                    # "total_pixels": 20480 * 32 * 32,
                }
            ]
        })
        
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        
        if "Qwen3-VL" in self.model_path:
            images, videos, video_kwargs = process_vision_info(messages, image_patch_size=16, return_video_kwargs=True, return_video_metadata=True)
            if videos is not None:
                videos, video_metadatas = zip(*videos)
                videos, video_metadatas = list(videos), list(video_metadatas)
            else:
                video_metadatas = None
                
            inputs = self.processor(
                text=[text], images=images, videos=videos, video_metadata=video_metadatas, return_tensors="pt", do_resize=False, **video_kwargs
            )
            inputs = inputs.to(self.model.device).to(self.model.dtype)
            
        else:
            image_inputs, video_inputs = process_vision_info(messages)
            inputs = self.processor(text=[text], images=image_inputs, videos=video_inputs, padding=True, return_tensors="pt")
            inputs = inputs.to(self.model.device).to(self.model.dtype)
            
        if "Qwen3-Omni" in self.model_path:
            output_ids, audio = self.model.generate(**inputs, return_audio=False)
        else:
            output_ids = self.model.generate(**inputs, max_new_tokens=512)
        
        output_trimmed = output_ids[:, inputs.input_ids.shape[1]:] # output_ids.sequences[]
        
        response = self.processor.batch_decode(output_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        response = re.sub(r'\b\d+\.\d+\s*seconds\b|<|>', '', response)
        
        return response
     