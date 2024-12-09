from pydub import AudioSegment

# Path to the MP4 file
mp4_file = "./data/test_Twist_And_Shout.mp4"
# Path to save the WAV file
wav_file = "./data/test_Twist_And_Shout.wav"

# Convert MP4 to WAV
audio = AudioSegment.from_file(mp4_file, format="mp4")
audio.export(wav_file, format="wav")

print(f"File has been converted and saved as {wav_file}")
