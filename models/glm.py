import base64

from transformers import (AutoModelForImageTextToText, AutoProcessor,
                          Glm4vMoeForConditionalGeneration)
from zai import ZhipuAiClient


class GLM(object):
    def __init__(self, model_path, **kwargs):
        self.model_path = model_path
        self.dtype = "auto"
        self.max_new_tokens = 512
        
        self.init_model()
        
    def init_model(self):
        register = {
            "GLM-4.1V": {
                "model": lambda: AutoModelForImageTextToText.from_pretrained(self.model_path, dtype=self.dtype, device_map="auto", trust_remote_code=True), 
                "processor": lambda: AutoProcessor.from_pretrained(self.model_path, use_fast=True, trust_remote_code=True)
            },
            "GLM-4.5V": {
                "model": lambda: AutoModelForImageTextToText.from_pretrained(self.model_path, dtype=self.dtype, device_map="auto", trust_remote_code=True), 
                "processor": lambda: AutoProcessor.from_pretrained(self.model_path, use_fast=True, trust_remote_code=True)
            },
            "GLM-4.6V": {
                "model": lambda: Glm4vMoeForConditionalGeneration.from_pretrained(self.model_path, dtype=self.dtype, device_map="auto", trust_remote_code=True), 
                "processor": lambda: AutoProcessor.from_pretrained(self.model_path, use_fast=True, trust_remote_code=True)
            }
        }
        
        for n in register.keys():
            if n in self.model_path:
                for k, v in register[n].items():
                    setattr(self, k, v())
                break
    
    def converse(self, video_path, question):
        messages = []
        messages.append({"role": "user", "content": [{"type": "text", "text": question}, {"type": "video", "url": video_path}]})
    
        inputs = self.processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt")
        inputs = inputs.to(self.model.device) # .to(self.model.dtype)
        
        if "GLM-4.6V" in self.model_path:
            inputs.pop("token_type_ids", None)
        
        output_ids = self.model.generate(**inputs, max_new_tokens=self.max_new_tokens)
        response = self.processor.decode(output_ids[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True)
        # response = re.sub(r'<\|user\|>|<\|begin_of_box\|>|<\|end_of_box\|>|</answer>|<|>', '', response)
        # match = re.search(r"<answer>(.*?)</answer>", response, re.DOTALL)
        # response = match.group(1).strip() if match else ""
            
        return response


class GLM_API(object):
    def __init__(self, model_name, api_key, **kwargs):
        self.client = ZhipuAiClient(api_key=api_key)
        self.model_name = model_name

    def converse(self, video_path, question):
        video_bytes = open(video_path, 'rb').read()
        video_base = base64.b64encode(video_bytes).decode("utf-8")

        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[{
                "role": "user",
                "content": [
                    {"type": "text", "text": question},
                    {"type": "video_url", "video_url": {"url": video_base}},
                ]
                }],
            thinking={"type": "disabled"}
        )
        
        return response.choices[0].message.content

