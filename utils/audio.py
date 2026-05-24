import yt_dlp   
from pydub import AudioSegment
import os


DOWNLOAD_DIR = 'downloades'

os.makedirs(DOWNLOAD_DIR,exist_ok=True)


def download_audio(url: str) -> str:

    output_path = os.path.join(
        DOWNLOAD_DIR,
        "%(title)s.%(ext)s"
    )

    ydl_opts = {
        'format': 'bestaudio/best',
        'outtmpl': output_path,
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'mp3',
        }],
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:

        info = ydl.extract_info(url, download=True)

        filename = ydl.prepare_filename(info)

        filename = os.path.splitext(filename)[0] + ".mp3"

    return filename



def chunk_audio(audio_path: str, chunk_minutes: int = 5) -> list:

    audio = AudioSegment.from_file(audio_path)

    audio = audio.set_channels(1)
    audio = audio.set_frame_rate(16000)

    chunk_ms = chunk_minutes * 60 * 1000

    chunks = []

    for i, start in enumerate(range(0, len(audio), chunk_ms)):

        chunk = audio[start:start + chunk_ms]

        chunk_path = f"{audio_path}_chunk_{i}.mp3"

        chunk.export(
            chunk_path,
            format="mp3",
            bitrate="64k"
        )

        chunks.append(chunk_path)

    return chunks

def process_input(source: str) -> list:

    if source.startswith("http://") or source.startswith("https://"):

        print("Detected YouTube URL. Downloading audio...")

        audio_path = download_audio(source)

    else:

        print("Detected local file...")

        audio_path = source

    print("Chunking audio...")

    chunks = chunk_audio(audio_path)

    print(f"Audio ready — {len(chunks)} chunk(s) created.")

    return chunks
