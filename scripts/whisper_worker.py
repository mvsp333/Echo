import base64
import gc
import json
import os
import sys
import traceback
import wave

import numpy as np
import torch
import whisper

# faster-whisper (CTranslate2) is ~4x faster than openai-whisper on CPU with
# int8 weights. It's optional: when installed we prefer it, otherwise we keep
# using openai-whisper exactly as before.
try:
    from faster_whisper import WhisperModel as FasterWhisperModel
except Exception:  # pragma: no cover - optional dependency
    FasterWhisperModel = None


model = None
model_name = None
model_dir = None
model_device = None
model_backend = None
model_effective_backend = None

# A decoded segment is discarded when Whisper itself rates it probably not
# speech and has low confidence in the words (typical of hallucinations).
NO_SPEECH_DROP = 0.5
LOGPROB_DROP = -0.7

# openai-whisper model names that faster-whisper publishes under another id.
FASTER_MODEL_ALIASES = {
    'large': 'large-v3',
    'turbo': 'large-v3-turbo',
}


def send(payload):
    sys.stdout.write(json.dumps(payload, ensure_ascii=False) + '\n')
    sys.stdout.flush()


def unload_model():
    global model, model_name, model_dir, model_device, model_backend, model_effective_backend
    model = None
    model_name = None
    model_dir = None
    model_device = None
    model_backend = None
    model_effective_backend = None
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def resolve_backend(requested):
    requested = (requested or 'auto').lower()
    if requested == 'openai':
        return 'openai'
    if requested == 'faster':
        if FasterWhisperModel is None:
            raise RuntimeError('faster-whisper backend requested but the faster-whisper package is not installed')
        return 'faster'
    return 'faster' if FasterWhisperModel is not None else 'openai'


def load_faster_model(name, device, download_root):
    compute_type = 'float16' if device == 'cuda' else 'int8'
    faster_name = FASTER_MODEL_ALIASES.get(name, name)
    faster_root = os.path.join(download_root, 'faster-whisper') if download_root else None
    return FasterWhisperModel(
        faster_name,
        device=device,
        compute_type=compute_type,
        download_root=faster_root,
        cpu_threads=max(1, (os.cpu_count() or 4)),
    )


def ensure_model(request):
    global model, model_name, model_dir, model_device, model_backend, model_effective_backend
    requested_name = request.get('model') or 'small'
    requested_dir = request.get('model_dir') or None
    requested_device = request.get('device') or 'auto'
    if requested_device == 'auto':
        requested_device = 'cuda' if torch.cuda.is_available() else 'cpu'

    if requested_device == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError(
            'CUDA was requested but this Python environment has no CUDA-enabled PyTorch build'
        )

    requested_backend = resolve_backend(request.get('backend'))

    if (
        model is None
        or model_name != requested_name
        or model_dir != requested_dir
        or model_device != requested_device
        or model_backend != requested_backend
    ):
        unload_model()
        loaded_backend = requested_backend
        if requested_backend == 'faster':
            try:
                model = load_faster_model(requested_name, requested_device, requested_dir)
            except Exception as error:
                # Model download blocked/offline or unsupported name: keep the
                # app working by falling back to openai-whisper.
                sys.stderr.write(f'faster-whisper load failed, using openai-whisper: {error}\n')
                sys.stderr.flush()
                loaded_backend = 'openai'
        if loaded_backend == 'openai':
            model = whisper.load_model(
                requested_name,
                device=requested_device,
                download_root=requested_dir,
            )
        model_name = requested_name
        model_dir = requested_dir
        model_device = requested_device
        # Keyed on what was *requested* so a fallback doesn't reload every call.
        model_backend = requested_backend
        model_effective_backend = loaded_backend

    return model, requested_device, model_effective_backend


def pcm16_to_float(raw):
    return np.frombuffer(raw, dtype='<i2').astype(np.float32) / 32768.0


def load_audio_input(request):
    """Return float32 16 kHz mono samples (or a path for non-PCM files)."""
    if request.get('audio_b64'):
        # Raw 16 kHz mono PCM16 sent inline: no temp file, no disk I/O.
        return pcm16_to_float(base64.b64decode(request['audio_b64']))

    audio_path = request['audio_path']
    if not str(audio_path).lower().endswith('.wav'):
        return audio_path

    try:
        with wave.open(audio_path, 'rb') as wav_file:
            channels = wav_file.getnchannels()
            sample_width = wav_file.getsampwidth()
            sample_rate = wav_file.getframerate()
            frames = wav_file.readframes(wav_file.getnframes())

        if sample_width != 2:
            return audio_path

        samples = np.frombuffer(frames, dtype='<i2').astype(np.float32)
        if channels > 1:
            samples = samples.reshape(-1, channels).mean(axis=1)
        samples /= 32768.0

        if sample_rate != whisper.audio.SAMPLE_RATE and samples.size:
            output_size = max(1, round(samples.size * whisper.audio.SAMPLE_RATE / sample_rate))
            source_positions = np.arange(samples.size, dtype=np.float64)
            target_positions = np.linspace(0, samples.size - 1, output_size)
            samples = np.interp(target_positions, source_positions, samples).astype(np.float32)

        return samples
    except (OSError, EOFError, wave.Error, ValueError):
        # Non-PCM or malformed WAV files can still use Whisper's normal
        # filename loader when ffmpeg is available on the host.
        return audio_path


def transcribe(request):
    loaded_model, device, backend = ensure_model(request)
    language = request.get('language') or None
    if language in ('auto', 'detect'):
        language = None
    initial_prompt = request.get('prompt') or None
    beam_size = max(1, int(request.get('beam_size') or 1))
    audio = load_audio_input(request)

    # Guards against Whisper's repetition loops ("a machine that is not a
    # machine that is not ..."), which noisy or unclear audio can trigger:
    # - temperature fallback: a decode whose text compresses too well
    #   (repetitive) or has low confidence is retried at a higher temperature.
    #   A single fixed temperature disables that retry entirely.
    # - a token cap sized to the clip: speech is ~3 words/s, so a short clip
    #   can never legitimately produce hundreds of tokens.
    temperatures = (0.0, 0.2, 0.4, 0.6)
    max_tokens = None
    if not isinstance(audio, str):
        seconds = len(audio) / whisper.audio.SAMPLE_RATE
        max_tokens = int(min(300, 24 + seconds * 8))

    if backend == 'faster':
        segments, info = loaded_model.transcribe(
            audio,
            language=language,
            task='transcribe',
            beam_size=beam_size,
            best_of=1,
            temperature=temperatures,
            compression_ratio_threshold=2.4,
            log_prob_threshold=-1.0,
            no_speech_threshold=0.6,
            condition_on_previous_text=False,
            without_timestamps=True,
            # Silero speech detection: strips music, chimes and hum before
            # decoding, so non-speech audio yields no text instead of an
            # invented sentence ("This is the end of the session.").
            vad_filter=True,
            vad_parameters={'min_silence_duration_ms': 500, 'speech_pad_ms': 300},
            initial_prompt=initial_prompt,
            max_new_tokens=max_tokens,
        )
        kept = []
        dropped = []
        for segment in segments:
            # Whisper's own verdict: likely non-speech AND unsure of the words.
            if segment.no_speech_prob > NO_SPEECH_DROP and segment.avg_logprob < LOGPROB_DROP:
                dropped.append(segment.text.strip())
                continue
            kept.append(segment.text)
        if dropped:
            sys.stderr.write(f'dropped low-confidence non-speech text: {dropped!r}\n')
            sys.stderr.flush()
        text = ''.join(kept)
        detected = info.language
    else:
        options = {'sample_len': max_tokens} if max_tokens else {}
        result = loaded_model.transcribe(
            audio,
            language=language,
            task='transcribe',
            fp16=device == 'cuda',
            verbose=None,
            temperature=temperatures,
            compression_ratio_threshold=2.4,
            logprob_threshold=-1.0,
            no_speech_threshold=0.6,
            beam_size=beam_size if beam_size > 1 else None,
            condition_on_previous_text=False,
            initial_prompt=initial_prompt,
            **options,
        )
        text = result.get('text') or ''
        detected = result.get('language')

    return {
        'text': text.strip(),
        'language': detected,
        'device': device,
        'backend': backend,
        'model': model_name,
        'cuda_available': torch.cuda.is_available(),
        'gpu': torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    }


send({
    'event': 'ready',
    'pid': os.getpid(),
    'torch': torch.__version__,
    'cuda_available': torch.cuda.is_available(),
    'torch_cuda': torch.version.cuda,
    'faster_whisper': FasterWhisperModel is not None,
    'gpu': torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
})

for raw_line in sys.stdin:
    raw_line = raw_line.strip()
    if not raw_line:
        continue

    request_id = None
    try:
        request = json.loads(raw_line)
        request_id = request.get('id')
        action = request.get('action')

        if action == 'transcribe':
            send({'id': request_id, 'ok': True, **transcribe(request)})
        elif action == 'warmup':
            _, device, backend = ensure_model(request)
            send({
                'id': request_id,
                'ok': True,
                'warmed': True,
                'device': device,
                'backend': backend,
                'model': model_name,
                'cuda_available': torch.cuda.is_available(),
                'gpu': torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            })
        elif action == 'unload':
            unload_model()
            send({'id': request_id, 'ok': True, 'unloaded': True})
        elif action == 'shutdown':
            unload_model()
            send({'id': request_id, 'ok': True, 'shutdown': True})
            break
        else:
            raise ValueError(f'Unsupported worker action: {action}')
    except Exception as error:
        send({
            'id': request_id,
            'ok': False,
            'error': str(error),
            'traceback': traceback.format_exc(),
        })
