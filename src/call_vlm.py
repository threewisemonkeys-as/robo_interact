from openai import OpenAI

client = OpenAI()              # reads key from env

import base64, mimetypes, pathlib

def as_data_url(path: str) -> str:
    mime, _ = mimetypes.guess_type(path)
    if mime is None:
        mime = "application/octet-stream"
    b64 = base64.b64encode(pathlib.Path(path).read_bytes()).decode()
    return f"data:{mime};base64,{b64}"

image_data_url = as_data_url("/Users/daiyijia/robo_interact/data/masked/a1.png")

response = client.chat.completions.create(
    model="gpt-4o",
    messages=[
        {
            "role": "user",
            "content": [
                {"type": "text",
                 "text": "Which point should I grasp to pick up the string? I prefer the point on the body of the string. Only return the point, no other text."},
                {"type": "image_url",
                 "image_url": {
                     "url": image_data_url,
                     "detail": "high"
                 }}
            ]
        }
    ],
    max_tokens=250
)

print(response.choices[0].message)
