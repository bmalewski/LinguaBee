import torch
from pyannote.audio import Pipeline
import noisereduce as nr
import librosa
import soundfile as sf
from pydub import AudioSegment
from pydub.effects import normalize as pydub_normalize
import os
from config import downloads_dir

# Keep a global cache for the pipeline to avoid reloading it.
diarization_pipeline_cache = {}

def diarize_audio(audio_path: str, config, status_signal, progress_signal):
    """
    Performs speaker diarization using pyannote.audio.
    """
    global diarization_pipeline_cache

    if not config.hf_token:
        status_signal.emit("Błąd: Brak tokenu Hugging Face do diaryzacji. Proszę go dodać w ustawieniach.", "error")
        return []

    pipeline_key = "diarization_pipeline"
    
    try:
        if pipeline_key not in diarization_pipeline_cache:
            status_signal.emit("Ładowanie modelu diaryzacji (pyannote/speaker-diarization-3.1)...", "info")
            pipeline = Pipeline.from_pretrained(
                "pyannote/speaker-diarization-3.1"
            )
            
            # Ustawienie niestandardowych hiperparametrów, jeśli są zdefiniowane w konfiguracji
            segmentation_batch_size = getattr(config, 'diar_segmentation_batch_size', 32)
            embedding_batch_size = getattr(config, 'diar_embedding_batch_size', 32)
            pipeline.segmentation_batch_size = segmentation_batch_size
            pipeline.embedding_batch_size = embedding_batch_size
            
            device = torch.device(config.whisper_device)
            if config.whisper_device == "cuda" and torch.cuda.is_available():
                device = torch.device(f"cuda:{config.whisper_device_index}")
            
            pipeline.to(device)
            diarization_pipeline_cache[pipeline_key] = pipeline
        
        pipeline = diarization_pipeline_cache[pipeline_key]
        
        status_signal.emit("Wykonywanie diaryzacji...", "info")
        num_speakers = getattr(config, 'num_speakers', 0)
        diarization = pipeline(audio_path, num_speakers=num_speakers if num_speakers > 0 else None)
        status_signal.emit("Diaryzacja zakończona.", "info")
        
        timeline = []
        for turn, _, speaker in diarization.itertracks(yield_label=True):
            timeline.append({
                'start': turn.start,
                'end': turn.end,
                'speaker': speaker
            })
            
        return timeline

    except Exception as e:
        status_signal.emit(f"Błąd podczas diaryzacji: {e}", "error")
        # Attempt to clear cache if loading failed, so it can be retried.
        if pipeline_key in diarization_pipeline_cache:
            del diarization_pipeline_cache[pipeline_key]
        return []

def assign_speakers_to_words(diarization_timeline: list, whisper_segments: list):
    """
    Assigns a speaker to each word in the Whisper segments based on the diarization timeline.
    """
    for segment in whisper_segments:
        if 'words' not in segment or not segment['words']:
            continue
        for word in segment['words']:
            word_center = (word.get('start', 0.0) + word.get('end', 0.0)) / 2
            best_speaker = 'UNKNOWN'
            for turn in diarization_timeline:
                if turn['start'] <= word_center < turn['end']:
                    best_speaker = turn['speaker']
                    break
            word['speaker'] = best_speaker
    return whisper_segments

def create_speaker_paragraphs(word_level_segments: list, max_chars: int = 0, break_on_speaker: bool = True):
    """
    Groups words into paragraphs based on speaker changes and other criteria.
    """
    if not word_level_segments or 'words' not in word_level_segments[0]:
        return []

    paragraphs = []
    current_paragraph = None

    for segment in word_level_segments:
        if 'words' not in segment or not segment['words']:
            continue
        
        for word in segment['words']:
            speaker = word.get('speaker', 'UNKNOWN')
            
            # Start of the very first paragraph
            if current_paragraph is None:
                current_paragraph = {
                    'text': word['word'],
                    'start': word['start'],
                    'end': word['end'],
                    'speakers': {speaker}
                }
                continue

            current_speaker = list(current_paragraph['speakers'])[0] if len(current_paragraph['speakers']) == 1 else 'MIXED'
            
            # Conditions to create a new paragraph:
            # 1. Speaker changes (if break_on_speaker is True)
            # 2. Maximum character limit is reached (if max_chars > 0)
            new_paragraph = False
            if break_on_speaker and speaker != current_speaker:
                new_paragraph = True
            if max_chars > 0 and len(current_paragraph['text']) + len(word['word']) + 1 > max_chars:
                new_paragraph = True
            
            if new_paragraph:
                # Finalize the current paragraph
                paragraphs.append(current_paragraph)
                # Start a new one
                current_paragraph = {
                    'text': word['word'],
                    'start': word['start'],
                    'end': word['end'],
                    'speakers': {speaker}
                }
            else:
                # Extend the current paragraph
                current_paragraph['text'] += f" {word['word']}"
                current_paragraph['end'] = word['end']
                current_paragraph['speakers'].add(speaker) # Add speaker to the set

    # Add the last paragraph
    if current_paragraph is not None:
        paragraphs.append(current_paragraph)

    # Clean up speaker sets if they contain UNKNOWN and other speakers
    for para in paragraphs:
        if len(para['speakers']) > 1 and 'UNKNOWN' in para['speakers']:
            para['speakers'].remove('UNKNOWN')

    return paragraphs

def process_audio(audio_path: str, config, status_signal) -> str:
    """
    Applies selected audio processing steps: denoising, normalization, and mono conversion.
    Returns the path to the processed audio file. If no processing is done, returns the original path.
    """
    do_denoise = getattr(config, 'enable_denoising', False)
    do_normalize = getattr(config, 'enable_normalization', False)
    do_mono = getattr(config, 'force_mono', False)

    if not any([do_denoise, do_normalize, do_mono, do_mono]):
        return audio_path # No processing needed

    try:
        status_signal.emit("Przetwarzanie audio (odszumianie/normalizacja/mono)...", "info")
        
        # 1. Load audio using pydub, which is great for format handling and normalization
        audio_segment = AudioSegment.from_file(audio_path)

        # 2. Convert to mono if requested
        if do_mono and audio_segment.channels > 1:
            status_signal.emit(" - Krok 1: Konwersja do mono...", "info")
            audio_segment = audio_segment.set_channels(1)

        # 3. Perform loudness normalization if enabled (more intelligent than peak normalization)
        if do_normalize:
            status_signal.emit(" - Krok 2: Normalizacja głośności (LUFS)...", "info")
            # Normalize to a standard -20.0 dBFS loudness. This is a good target for speech.
            audio_segment = pydub_normalize(audio_segment, headroom=0.1)

        # 4. Get audio data as numpy array for noise reduction
        sample_rate = audio_segment.frame_rate
        audio_data = audio_segment.get_array_of_samples()
        # Convert to float, which is required by noisereduce
        import numpy as np
        audio_data = np.array(audio_data).astype(np.float32) / (2**(audio_segment.sample_width * 8 - 1))
        
        # 5. Perform noise reduction if enabled, but with less aggression
        if do_denoise:
            status_signal.emit(" - Krok 3: Odszumianie (mniej agresywne)...", "info")
            # Use a less aggressive noise reduction setting.
            # prop_decrease=0.9 means it will reduce noise by 90% of the default, leaving more of the original signal.
            audio_data = nr.reduce_noise(y=audio_data, sr=sample_rate, prop_decrease=0.9)
        
        # 6. Create a path for the new, processed file
        base_name = os.path.splitext(os.path.basename(audio_path))[0]
        # Ensure the original base name doesn't already end with '_audio' from video extraction
        if base_name.endswith('_audio'):
            base_name = base_name[:-6]
        processed_filename = f"{base_name}_processed.wav"
        processed_path = os.path.join(downloads_dir, processed_filename)

        # 7. Save the processed audio to the new file
        # We use WAV as it's a lossless format suitable for further processing.
        sf.write(processed_path, audio_data, sample_rate)
        
        status_signal.emit(f"Przetwarzanie audio zakończone. Plik zapisano w: {processed_path}", "info")
        return processed_path
    except Exception as e:
        status_signal.emit(f"Błąd podczas przetwarzania audio: {e}. Kontynuowanie na oryginalnym pliku.", "warning")
        return audio_path # On failure, return the original path