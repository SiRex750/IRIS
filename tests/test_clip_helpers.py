import inspect
import iris.pipeline as p
import iris._clip as c

LIFTED = [
    "get_blip_model",
    "get_generative_caption",
    "get_zero_shot_caption",
    "get_semantic_and_clip_caption",
    "get_clip_model",
    "get_frame_clip_embedding",
    "get_clip_embedding_from_pil",
]


def test_all_helpers_present_and_callable():
    for name in LIFTED:
        fn = getattr(c, name)
        assert callable(fn), name


def test_helpers_are_verbatim_copies():
    for name in LIFTED:
        src_pipeline = inspect.getsource(getattr(p, name))
        src_clip = inspect.getsource(getattr(c, name))
        assert src_pipeline == src_clip, f"{name} diverged from pipeline.py"
