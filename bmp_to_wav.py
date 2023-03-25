import os

if not os.path.isdir("wav_generated"):
        os.mkdir("wav_generated")

# Convert into WAV
bmp_files = os.listdir("bmp_generated")
for bmp in bmp_files:
    wav_name = str(bmp.split(".")[0]) + ".wav"
    os.system(f"arss bmp_generated/{bmp} wav_generated/{wav_name} -q -n -f 16 -r 44100 -min 27.5 -b 13.359 -p 64.5")