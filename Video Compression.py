import subprocess
import os

input_video = input("Enter full video path: ").strip().strip('"')

# Output file name
base, ext = os.path.splitext(input_video)
output_video = base + "_compressed.mp4"

# Compression command
command = [
    "ffmpeg",
    "-i", input_video,
    "-vcodec", "libx264",
    "-crf", "23",          # Lower = better quality (18–28 range)
    "-preset", "medium",   # slower = smaller file
    "-acodec", "aac",
    output_video
]

print("Compressing video...")
subprocess.run(command)

print("Done!")
print("Compressed file saved as:")
print(output_video)
