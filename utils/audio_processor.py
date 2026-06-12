import yt_dlp
import os
import subprocess

DOWNLOAD_DIR = "downloads"
os.makedirs(DOWNLOAD_DIR, exist_ok=True)


def download_youtube_audio(url: str) -> str:
    """
    Download audio from YouTube.
    No conversion is done here.
    """

    output_path = os.path.join(
        DOWNLOAD_DIR,
        "%(title)s.%(ext)s"
    )

    ydl_opts = {
        "format": "bestaudio/best",
        "outtmpl": output_path,
        "quiet": True,
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=True)

        file_path = ydl.prepare_filename(info)

    return file_path


def convert_to_wav(input_path: str) -> str:
    """
    Convert any audio/video file to
    16kHz mono WAV using FFmpeg.
    """

    output_path = (
        os.path.splitext(input_path)[0]
        + ".wav"
    )

    subprocess.run(
        [
            "ffmpeg",
            "-i",
            input_path,
            "-ac", "1",      # mono
            "-ar", "16000",  # 16 kHz
            output_path,
            "-y",
        ],
        check=True,
    )

    return output_path


def chunk_audio(
    wav_path: str,
    chunk_minutes: int = 10
) -> list[str]:
    """
    Split WAV into smaller WAV chunks.
    """

    output_dir = (
        os.path.splitext(wav_path)[0]
        + "_chunks"
    )

    os.makedirs(output_dir, exist_ok=True)

    chunk_pattern = os.path.join(
        output_dir,
        "chunk_%03d.wav"
    )

    subprocess.run(
        [
            "ffmpeg",
            "-i",
            wav_path,
            "-f",
            "segment",
            "-segment_time",
            str(chunk_minutes * 60),
            "-c",
            "copy",
            chunk_pattern,
            "-y",
        ],
        check=True,
    )

    return [
        os.path.join(output_dir, file)
        for file in sorted(os.listdir(output_dir))
        if file.endswith(".wav")
    ]


def process_input(
    source: str,
    chunk_minutes: int = 10
) -> list[str]:
    """
    Accept either:
    - YouTube URL
    - Local file path
    """

    if source.startswith(("http://", "https://")):
        print("Detected YouTube URL. Downloading audio...")
        source = download_youtube_audio(source)

    print("Converting to WAV...")
    wav_path = convert_to_wav(source)

    print("Chunking audio...")
    chunks = chunk_audio(
        wav_path,
        chunk_minutes
    )

    print(
        f"Audio ready — {len(chunks)} chunk(s) created."
    )

    return chunks


if __name__ == "__main__":

    source = input(
        "Enter YouTube URL or local file path: "
    )

    chunks = process_input(
        source,
        chunk_minutes=1
    )

    print("\nGenerated Chunks:")

    for chunk in chunks:
        print(chunk)