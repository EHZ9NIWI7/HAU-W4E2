from .gemini import Gemini
from .glm import GLM, GLM_API
from .gpt import GPT
from .intern_vl import InternVL
from .qwen import Qwen

_MODELS = {
    "Qwen": Qwen,
    "InternVL": InternVL,
    "GLM": GLM,
    "glm": GLM_API,
    "gemini": Gemini,
    "gpt": GPT,
}

def get_model(model_path, **kwargs):
    for key, cls in _MODELS.items():
        if key in model_path:
            return cls(model_path, **kwargs)
    raise ValueError(f"Wrong model path: {model_path}")
