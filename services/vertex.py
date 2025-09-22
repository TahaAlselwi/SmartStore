import vertexai
from vertexai.generative_models import GenerativeModel
from vertexai.vision_models import MultiModalEmbeddingModel, Image

_model = None

def init_vertex():
    vertexai.init()

def get_model(model_name: str) -> GenerativeModel:
    global _model
    if _model is None:
        init_vertex()
        _model = GenerativeModel(model_name)
    return _model

def draft_reply(model_name: str, prompt: str) -> str:
    model = get_model(model_name)
    resp = model.generate_content(prompt)
    return getattr(resp, "text", "").strip()


_mm_model = None  # cached multimodal embedding model

def get_mm_model() -> MultiModalEmbeddingModel:
    """Return a cached MultiModalEmbeddingModel instance."""
    global _mm_model
    if _mm_model is None:
        init_vertex()
        _mm_model = MultiModalEmbeddingModel.from_pretrained("multimodalembedding@001")
    return _mm_model

def get_image_embedding(image_bytes: bytes) -> list[float]:
    """
    Compute an image embedding from raw bytes (no GCS upload).
    Returns: list[float].
    """
    model = get_mm_model()
    img = Image(image_bytes=image_bytes)
    resp = model.get_embeddings(image=img)
    return [float(x) for x in resp.image_embedding]
