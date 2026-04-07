import mimetypes

from google import genai
from google.genai import types


class Gemini:
    def __init__(self, model_name, api_key, **kwargs):
        self.client = genai.Client(api_key=api_key,)# http_options=types.HttpOptions(base_url="https://api.zhizengzeng.com/google/"))
        self.model_name = model_name
        
    def converse(self, video_path, question):
        mime_type, _ = mimetypes.guess_type(video_path)
        video_bytes = open(video_path, 'rb').read()

        response = self.client.models.generate_content(
            model=f'models/{self.model_name}',
            contents=types.Content(
                parts=[
                    types.Part(inline_data=types.Blob(data=video_bytes, mime_type=mime_type), video_metadata=types.VideoMetadata(fps=2)),
                    types.Part(text=question)
                ]
            )
        )
        
        return response.text