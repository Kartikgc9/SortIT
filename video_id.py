import re

video_url = "https://youtu.be/MfdXFbB-1oo?si=ltjYaJgNJ-2tFM8Z"

video_id_pattern = r"(?:v=|be/|embed/)([^&\n]+)"
match = re.search(video_id_pattern, video_url)

if match:
    video_id = match.group(1)
    print(video_id)
else:
    print("Video ID not found in the URL")