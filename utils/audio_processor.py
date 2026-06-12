import yt_dlp
# from pydub import AudioSegment
import os 
import subprocess

DOWNLOAD_DIR = 'downloads'
os.makedirs(DOWNLOAD_DIR, exist_ok= True)

def downlaod_youtube_audio(url: str) -> str:
    output_path = os.path.join(DOWNLOAD_DIR, "%(title)s.%(ext)s")
    ydl_opts = {
        "format" : "bestaudio/best",
        "outtmpl" : output_path,
        "postprocessors" : [
            
            {
                "key" : "FFmpegExtractAudio",
                "preferredcodec" : "wav",
                "preferredquality" : "192",
            }
        ],
        "quiet" : True,
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=True)
        filename = ydl.prepare_filename(info).replace(".webm", ".wav").replace(".m4a", ".wav")
    return filename

data =downlaod_youtube_audio("https://youtu.be/T-D1OfcDW1M?si=aG13Cxd357EpkQTu")
# print(f1)

def convert_to_wav(input_path: str) -> str:
    """Convert any audio/video to WAV using FFmpeg."""
    
    output_path = os.path.splitext(input_path)[0] + "_converted.wav"

    subprocess.run(
        [
            "ffmpeg",
            "-i",
            input_path,
            output_path,
            "-y"
        ],
        check=True
    )

    return output_path
print(convert_to_wav(data))

