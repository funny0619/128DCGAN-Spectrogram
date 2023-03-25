import os
from scipy.io import wavfile
from tqdm import trange
from pathlib import Path
from PIL import Image
import glob



print("If running this with newly added songs, make sure to remove previously converted songs from directory")
song_file = input("Input Directory of Songs: ")
index = 0
if os.path.isdir("data"):
    index = len(os.listdir("data"))
else:
    os.mkdir("data")
# Split songs in directory into 2 second clips
for song in os.listdir(song_file):
    song_dir = os.path.join(song_file,song)
    sample_rate, data = wavfile.read(song_dir)
    len_data = len(data)  # holds length of the numpy array
    seconds = len_data / sample_rate  # returns duration but in floats
    print(f"Splitting {song}")
    for starting_second in trange(0,int(seconds),2):
        output_file = f"data/{index}.wav"
        os.system(f"ffmpeg -ss {starting_second} -t 2 -loglevel panic -i {song_dir} -bitexact -map_metadata -1 {output_file}")
        index += 1

index = 0
# Convert into BMP
wav_files = os.listdir("data")

if os.path.isdir("bmp"):
    bmp_files = os.listdir("bmp")
    index = len(bmp_files)
else:
    os.mkdir("bmp")
for file_number in range(index,len(os.listdir("data"))):
    os.system(f"arss -q data/{file_number}.wav bmp/{file_number}.bmp -min 27.5 -max 20000 -x 129 -y 128")

index = 0
if os.path.isdir("png"):
    index = len(os.listdir("png"))
else:
    os.mkdir("png")

for file_number in trange(index,len(bmp_files)):
    bmp_file_name = os.path.join("bmp",f"{file_number}.bmp")
    png_file_name = os.path.join("png", f"{file_number}.png")
    Image.open(bmp_file_name).convert('L').save(png_file_name)