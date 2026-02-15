import os
import sys
import re
import yt_dlp
from config import downloads_dir

class YtdlpLogger:
    """Logger for yt-dlp to redirect its output to the GUI status signal."""
    def __init__(self, status_signal):
        self.status_signal = status_signal

    def debug(self, msg):
        # Ignore debug messages from yt-dlp
        if msg.startswith('[debug] '):
            return
        # Clean up ANSI escape codes
        ansi_escape = re.compile(r'\x1b\[[0-9;]*m')
        msg = ansi_escape.sub('', msg).strip()
        if msg:
            self.status_signal.emit(msg, "info")

    def warning(self, msg):
        self.status_signal.emit(f"WARNING: {msg}", "warning")

    def error(self, msg):
        self.status_signal.emit(f"ERROR: {msg}", "error")

def download_audio(url, progress_signal, status_signal, finished_signal, is_stopped_check, cookies_file: str = None):
    """Downloads audio from a given URL, converts it to MP3, and returns the path and base name."""
    def hook(d):
        if is_stopped_check():
            raise StopIteration("Download stopped by user.")
        if d.get("status") == 'downloading':
            total = d.get('total_bytes') or d.get('total_bytes_estimate')
            if total:
                downloaded = d.get('downloaded_bytes', 0)
                percentage = (downloaded / total) * 100
                progress_signal.emit(int(percentage))
        elif d.get("status") == 'finished':
            progress_signal.emit(100)

    ytdlp_logger = YtdlpLogger(status_signal)
    output_template = os.path.join(downloads_dir, '%(title)s.%(ext)s')

    # Determine if ffmpeg/ffprobe are available on PATH. If not, we won't use postprocessing
    def _ffmpeg_available():
        from shutil import which
        return which('ffmpeg') is not None and which('ffprobe') is not None

    ffmpeg_ok = _ffmpeg_available()
    if not ffmpeg_ok:
        status_signal.emit("ffmpeg/ffprobe not found on PATH. Will download original audio without conversion (will keep original container).", "warning")

    base_ydl_opts = {
        'format': 'bestaudio/best',
        'outtmpl': output_template,
        'progress_hooks': [hook],
        'logger': ytdlp_logger,
        'quiet': True,
        'no_check_certificate': True,
        'retries': 5,
        'fragment_retries': 5,
        # Prefer modern clients and avoid requiring local JS runtime for basic extraction.
        'extractor_args': {
            'youtube': {
                'player_client': ['default', 'android', 'web_safari'],
            }
        },
    }

    if ffmpeg_ok:
        # Only add postprocessing if ffmpeg/ffprobe available
        base_ydl_opts['postprocessors'] = [dict(key='FFmpegExtractAudio', preferredcodec='mp3', preferredquality='192')]

    info = None
    original_filepath = None

    def attempt_download(ydl_opts):
        nonlocal info, original_filepath
        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(url, download=True)
                original_filepath = ydl.prepare_filename(info)
            return True
        except yt_dlp.utils.DownloadError as e:
            msg = str(e)
            lowered = msg.lower()
            # Browser-cookie errors are expected on some Windows setups (locked DB, missing profile).
            if (
                'cookies database' in lowered
                or 'could not copy chrome cookie database' in lowered
                or 'could not find' in lowered and 'cookies' in lowered
            ):
                status_signal.emit(f"Cookie attempt skipped: {msg}", "info")
            else:
                status_signal.emit(f"Download attempt failed: {msg}", "warning")
            return False
        except StopIteration:
            raise # Re-raise to be caught by the outer try block
        except Exception as e:
            status_signal.emit(f"An unexpected error occurred during download attempt: {e}", "error")
            return False

    try:
        download_successful = False

        # If a cookies file was provided explicitly (e.g. exported by the user), prefer that and skip browser extraction
        if cookies_file:
            status_signal.emit(f"Using provided cookies file: {cookies_file}", "info")
            opts = base_ydl_opts.copy()
            opts['cookiefile'] = cookies_file
            if attempt_download(opts):
                download_successful = True
        else:
            # 1. Try downloading with browser cookies
            if sys.platform == "darwin":
                browsers_to_try = ['safari', 'chrome']
                status_signal.emit("macOS detected. Trying cookies from: Safari, Chrome.", "info")
            else:
                browsers_to_try = ['edge', 'chrome', 'firefox', 'opera', 'vivaldi', 'brave']
            
            for browser in browsers_to_try:
                if is_stopped_check(): return None, None
                status_signal.emit(f"Attempting download with '{browser}' browser cookies...", "info")
                opts = base_ydl_opts.copy()
                opts['cookiesfrombrowser'] = (browser,)
                if attempt_download(opts):
                    download_successful = True
                    break

        # 2. Fallback: download without cookies
        if not download_successful:
            if is_stopped_check(): return None, None
            status_signal.emit("Downloading without cookies (fallback profiles)...", "info")

            # Try several yt-dlp profiles to mitigate YouTube 403/SABR issues.
            no_cookie_profiles = [
                {
                    'label': 'default clients',
                    'opts': {
                        'extractor_args': {
                            'youtube': {
                                'player_client': ['default', 'android', 'web_safari']
                            }
                        }
                    }
                },
                {
                    'label': 'android client',
                    'opts': {
                        'extractor_args': {
                            'youtube': {
                                'player_client': ['android']
                            }
                        }
                    }
                },
                {
                    'label': 'audio m4a preferred',
                    'opts': {
                        'format': 'bestaudio[ext=m4a]/bestaudio/best',
                        'extractor_args': {
                            'youtube': {
                                'player_client': ['android', 'default']
                            }
                        }
                    }
                },
            ]

            for profile in no_cookie_profiles:
                if is_stopped_check():
                    return None, None
                status_signal.emit(f"Trying no-cookie profile: {profile['label']}...", "info")
                opts = base_ydl_opts.copy()
                opts.update(profile['opts'])
                if attempt_download(opts):
                    download_successful = True
                    break

            if not download_successful:
                raise yt_dlp.utils.DownloadError("Failed to download after all attempts.")

        if info is None or original_filepath is None:
            raise ValueError("Failed to get video information.")

        # If postprocessing (ffmpeg) was used, yt-dlp will have converted to .mp3
        base_path_no_ext, orig_ext = os.path.splitext(original_filepath)
        if ffmpeg_ok:
            final_path = base_path_no_ext + '.mp3'
            base_name = os.path.basename(base_path_no_ext)
            if not os.path.exists(final_path):
                # Fallback: maybe yt-dlp kept original file
                if os.path.exists(original_filepath):
                    final_path = original_filepath
                else:
                    raise FileNotFoundError(f"Critical error: Downloaded audio file not found: {final_path}")
        else:
            # No ffmpeg: keep the downloaded file as-is (which may be .webm, .m4a, etc.)
            final_path = original_filepath
            base_name = os.path.basename(os.path.splitext(final_path)[0])

        return final_path, base_name

    except StopIteration:
        status_signal.emit("Download cancelled by user.", "info")
        return None, None
    except Exception as e:
        finished_signal.emit(f"Download failed: {e}", "error")
        return None, None
