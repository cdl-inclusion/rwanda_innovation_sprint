# Deploys a FASTAPI endpoint for Whisper models (via FasterWhisper) on Modal GPUs.
# Serves BOTH CDLI Rwandan fine-tunes (English + Kinyarwanda) from a single
# deployed class. Each container loads both models at startup; the `language`
# form field of the request picks the model (see LANGUAGES):
#   language=ky -> Kinyarwanda model, decoded with Whisper's `sw` setting
#   language=en -> Rwandan English model, decoded with `en`
#
# To deploy:
#   modal deploy whisper_endpoint.py
#
# To test (Kinyarwanda):
#   curl -X POST "https://cdli-hackathons--whisper-endpoint-transcriber-transcribe.modal.run" \
#     -F "wav=@some_kinyarwanda_clip.wav" -F "language=ky"
#
# To test (Rwandan English):
#   curl -X POST "https://cdli-hackathons--whisper-endpoint-transcriber-transcribe.modal.run" \
#     -F "wav=@jfk_asknot.wav" -F "language=en"
#
# Requires: huggingface-secret configured in Modal!

import modal
from pathlib import Path
import numpy as np
from fastapi import File, Form

MODAL_APP_NAME = "whisper_endpoint"
SAMPLE_RATE = 16000
BEAM_SIZE = 5
MODEL_MOUNT_DIR = Path("/models")
MODEL_DOWNLOAD_DIR = Path("downloads")
GPU = 'L4'
SCALEDOWN = 60 * 2  # seconds

# --- Language registry -------------------------------------------------
# API language code -> (HF model repo, language code Whisper decodes with).
# Kinyarwanda has no Whisper language code; the model was fine-tuned and must
# be decoded with `sw` (the closest language Whisper supports).
# Add a new language here (one line) rather than a new class/file.
LANGUAGES = {
    "ky": ("cdli/whisper-large-v3_finetuned_kinyarwanda_nonstandard_speech_v1.0", "sw"),
    "en": ("cdli/whisper-large-v3_finetuned_rwandan_english_nonstandard_speech_v1.0", "en"),
}


def maybe_download_and_convert_model(model_storage_dir, model_id):
    """Download and convert model to CTranslate2 format if not available locally.

    Checks cache first to avoid re-converting on every endpoint restart.
    A directory is only trusted as "cached" if it actually contains
    model.bin and tokenizer.json — a folder that exists but is missing either
    means a previous conversion was interrupted (or predates tokenizer.json),
    and gets removed and redone.
    Conversion happens in a temp dir first, then is atomically renamed into
    place, so a crash mid-conversion can never leave a broken folder at the
    final cache path again.
    """
    import ctranslate2
    import shutil
    import tempfile

    print(f"Checking for cached model in storage directory: {model_storage_dir}, model ID: {model_id}")
    subdir = model_id.replace("/", "_") + "_ct2"
    model_path = model_storage_dir / subdir
    print(f"Looking for model at: {model_path}")

    # A cache entry is only complete with both the weights and tokenizer.json
    # (entries converted before tokenizer.json was added get redone).
    if not ((model_path / "model.bin").exists() and (model_path / "tokenizer.json").exists()):
        if model_path.exists():
            print(f"Found incomplete/broken cached directory -- removing: {model_path}")
            shutil.rmtree(model_path)

        print(f"Model not found in cache -- downloading from HuggingFace: {model_id}")
        model_storage_dir.mkdir(parents=True, exist_ok=True)
        tmp_path = Path(tempfile.mkdtemp(dir=str(model_storage_dir)))
        converter = ctranslate2.converters.TransformersConverter(
            model_name_or_path=model_id,
            copy_files=["preprocessor_config.json"],
        )
        print(f"Converting to CTranslate2 format (float16)...")
        converter.convert(str(tmp_path), quantization="float16", force=True)
        # faster-whisper needs tokenizer.json next to the model. The fine-tuned
        # repos don't ship one, and without it faster-whisper falls back to
        # openai/whisper-tiny's tokenizer, whose special-token ids are off by one
        # for large-v3 (e.g. <|transcribe|> resolves to large-v3's <|translate|>).
        # Build it from the model's own tokenizer files instead.
        from transformers import WhisperTokenizerFast
        WhisperTokenizerFast.from_pretrained(model_id).save_pretrained(str(tmp_path))
        print(f"Saved tokenizer.json built from {model_id}")
        tmp_path.rename(model_path)
        print(f"Model successfully converted and saved to {model_path}")

        volume.commit()
    else:
        print(f"Found cached model at {model_path}")

    return str(model_path)


def transcribe_with_fasterwhisper(
        fasterwhisper_model,
        audio_array: np.ndarray,
        language,
        get_transcript_only=False,
        use_word_timestamps=False):
    """Actual transcription logic."""
    import numpy as np
    import time

    t1 = time.time()
    task = 'transcribe'
    print(f"Running Whisper transcription...")
    print(f"Language: {language if language else 'auto-detect'}")
    print(f"Audio length: {len(audio_array)} samples ({len(audio_array)/SAMPLE_RATE:.2f}s)")

    segments, _ = fasterwhisper_model.transcribe(
        audio_array,
        beam_size=BEAM_SIZE,
        language=language,
        task=task,
        condition_on_previous_text=False,
        vad_filter=True,
        word_timestamps=use_word_timestamps,
    )

    transcription = ''
    segment_texts = []
    words = []
    confidences = []
    compression_ratios = []

    print("[SEGMENTS] Processing transcribed segments...")
    segment_count = 0
    for segment in segments:
        segment_count += 1
        transcription += segment.text + ' '
        if get_transcript_only:
            continue
        segment_texts.append(segment.text)
        confidence = np.exp(segment.avg_logprob)
        confidences.append(confidence)
        compression_ratios.append(segment.compression_ratio)
        print(f"[SEGMENT {segment_count}] [{segment.start:.2f}s - {segment.end:.2f}s] "
              f"confidence: {confidence:.3f}, compression: {segment.compression_ratio:.3f}")
        print(f"[SEGMENT {segment_count}] Text: {segment.text}")
        if segment.words:
            for word in segment.words:
                words.append(word)

    transcription = transcription.strip()
    if get_transcript_only:
        return transcription

    elapsed_time = time.time() - t1
    avg_confidence = np.mean(confidences) if confidences else 0
    avg_compression_ratio = np.mean(compression_ratios) if compression_ratios else 0

    print(f"Transcription completed in {elapsed_time:.2f} seconds")
    print(f"Total segments: {segment_count}")
    print(f"Average confidence: {avg_confidence:.3f}")
    print(f"Average compression ratio: {avg_compression_ratio:.3f}")
    print(f"Final transcription: {transcription}")

    return {
        'result': "success",
        'transcription': transcription,
        'segments': segment_texts,
        'confidences': confidences,
        'compression_ratios': compression_ratios,
        'words': words
    }


#############################################
# Modal service with transcription endpoint
#############################################

# We need CUDA 12 + cuDNN 9 for faster-whisper (ctranslate2 depends on it).
# Versions are pinned below because unpinned installs are the #1 cause of
# "works locally, breaks on Modal" errors with this exact stack:
#   - ctranslate2 unpinned can drift past the cuDNN-9 ABI this base image ships.
#   - transformers unpinned needs a newer huggingface_hub than 0.26.2, which
#     throws an ImportError on container start.
#   - torch unpinned pulls a CUDA build with its own bundled cuDNN, which can
#     clash with ctranslate2's cuDNN at runtime. Torch here is only used by
#     the HF -> CTranslate2 converter (a one-time, CPU-fine step), so we pull
#     the CPU-only wheel instead of a full CUDA build to sidestep the clash.
cuda_image = (
    modal.Image.from_registry("nvidia/cuda:12.3.2-cudnn9-runtime-ubuntu22.04", add_python="3.11")
    .apt_install("git")
    .pip_install(
        "torch==2.4.1",
        extra_index_url="https://download.pytorch.org/whl/cpu",
    )
    .pip_install(
        "fastapi[standard]",
        "numpy",
        "librosa",
        "huggingface_hub[hf_transfer]==0.27.1",
        "ctranslate2==4.5.0",
        "faster_whisper==1.1.0",
        "transformers==4.47.1",
    )
)

app = modal.App(MODAL_APP_NAME)
volume = modal.Volume.from_name(MODAL_APP_NAME, create_if_missing=True)

with cuda_image.imports():
    from fastapi import File, Form, HTTPException
    import librosa
    import io
    from pathlib import Path
    import numpy as np
    from faster_whisper import WhisperModel


@app.cls(
    image=cuda_image,
    secrets=[modal.Secret.from_name("huggingface-secret")],
    gpu=GPU,
    scaledown_window=SCALEDOWN,
    enable_memory_snapshot=True,
    volumes={MODEL_MOUNT_DIR: volume})
@modal.concurrent(max_inputs=10)
class Transcriber:
    """Whisper transcriber serving all models in LANGUAGES.

    Every container loads all models at startup (two large-v3 models in
    float16 fit comfortably on one L4), and each request picks its model via
    the `language` form field.
    """

    @modal.enter()
    def enter(self):
        model_dir = MODEL_MOUNT_DIR / MODEL_DOWNLOAD_DIR
        self.whisper_models = {}
        for language, (model_id, whisper_language) in LANGUAGES.items():
            print(f"Loading model for language '{language}': {model_id} "
                  f"(Whisper decode language: {whisper_language})")
            model_path = maybe_download_and_convert_model(model_dir, model_id)
            self.whisper_models[language] = WhisperModel(model_path, device="cuda", compute_type="float16")
        print(f"FasterWhisper models loaded and ready: {list(self.whisper_models)}")

    @modal.fastapi_endpoint(docs=True, method="POST")
    def transcribe(self, wav: bytes = File(), language: str = Form(),
                   use_word_timestamps: bool = Form(default=False)):
        language = language.strip().lower()
        if language not in LANGUAGES:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported language '{language}'. Use one of: {sorted(LANGUAGES)}")
        model_id, whisper_language = LANGUAGES[language]
        audio_array, _ = librosa.load(io.BytesIO(wav), sr=SAMPLE_RATE)
        result = transcribe_with_fasterwhisper(
            self.whisper_models[language], audio_array, whisper_language,
            get_transcript_only=False, use_word_timestamps=use_word_timestamps)
        result['language'] = language
        result['model'] = model_id
        return result
