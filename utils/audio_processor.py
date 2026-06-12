import yt_dlp
from pydub import AudioSegment
import os 


DOWNLOAD_DIR = 'downloads'
os.makedirs(DOWNLOAD_DIR, exist_ok= True)

# def downlaod_youtube_audio(url: str) -> str:
