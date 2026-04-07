import base64
import io
import re

import numpy as np
import openai
from decord import VideoReader, cpu
from PIL import Image


class GPT:
    def __init__(self, model_name, api_key, **kwargs):
        self.client = openai.OpenAI(api_key=api_key,)# base_url="https://api.zhizengzeng.com/")
        self.model_name = model_name
        
    def load_video(self, video_path, fps: float = 1.0, min_frames: int = 16, jpeg_quality: int = 90):
        vr = VideoReader(video_path, ctx=cpu(0))
        total_frames = len(vr)
        orig_fps = vr.get_avg_fps()
        
        step = orig_fps / fps
        frame_indices = np.arange(0, total_frames, step).astype(int)
        
        frame_indices = np.unique(frame_indices)
        frame_indices = frame_indices[frame_indices < total_frames]
        
        if len(frame_indices) < min_frames:
            frame_indices = np.linspace(0, total_frames - 1, min_frames).astype(int)
        
        frames = vr.get_batch(frame_indices).asnumpy()
        
        base64_frames = []
        for frame in frames:
            img = Image.fromarray(frame)
            buffer = io.BytesIO()
            img.save(buffer, format="JPEG", quality=jpeg_quality)
            encoded = base64.b64encode(buffer.getvalue()).decode("utf-8")
            base64_frames.append(encoded)
        
        return base64_frames
        
    def converse(self, video_path, question):
        video_frames = self.load_video(video_path)
        
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[{
                "role": "user",
                "content": 
                    [{"type": "text", "text": question}] + \
                    [{"type": "image_url", "image_url": {'url': f'data:image/jpeg;base64,{f}'}} for f in video_frames]
            }]
        )
        
        response = response.choices[0].message.content
        response = re.sub(r'<|>', '', response)
        response = re.sub(r'\u2014', ' ', response)

        return response
