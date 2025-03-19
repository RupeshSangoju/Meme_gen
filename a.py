import os
from dotenv import load_dotenv
import requests
import pandas as pd
from PIL import Image, ImageDraw, ImageFont
from io import BytesIO
import matplotlib.pyplot as plt
import ast
import re
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict
import random
from random import shuffle

# Load environment variables from .env file
load_dotenv()
API_KEY = os.getenv("PERPLEXITY_API_KEY")

EMOTIONAL_CATEGORIES = [
    "Happy", "Sad", "Angry", "Relaxing", "Surprised", "Confused", "Excited", "Informative", "Sarcastic",
    "Wholesome", "Dark Humor", "Cringe", "Edgy", "Relatable", "Offensive", "Political", "Motivational",
    "Nostalgic", "Clever"
]

app = FastAPI()

class TextBox(BaseModel):
    x: str
    y: str
    width: str
    height: str
    text: str = None

class UserInput(BaseModel):
    text: str
    genre: str
    text_boxes: List[TextBox]

def classify_emotion_with_perplexity(text):
    url = "https://api.perplexity.ai/chat/completions"
    system_message = f"Identify the emotion conveyed in the user's text only from this list: {', '.join(EMOTIONAL_CATEGORIES)} without any explanation."

    payload = {
        "model": "sonar",
        "messages": [
            {"role": "system", "content": system_message},
            {"role": "user", "content": text}
        ],
        "temperature": 0.2,
        "top_p": 0.9,
        "max_tokens": 100,
        "presence_penalty": 0,
        "frequency_penalty": 1,
        "stream": False,
        "response_format": None
    }

    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }

    response = requests.post(url, json=payload, headers=headers)

    if response.status_code == 200:
        data = response.json()
        emotion = data.get("choices", [{}])[0].get("message", {}).get("content", "").strip()
        if "is" in emotion:
            emotion = emotion.split("is")[-1].strip()
        return emotion
    else:
        print("Error:", response.text)
        return None

def detect_emotion_categories(emotion_text):
    detected_categories = []
    emotion_text = emotion_text.lower()

    emotion_parts = [part.strip() for part in emotion_text.split("and")]

    for part in emotion_parts:
        part = part.strip('.')
        for category in EMOTIONAL_CATEGORIES:
            if category.lower() in part:
                detected_categories.append(category)

    if not detected_categories:
        detected_categories.append("Informative Meme")

    return detected_categories

def generate_meme_text(category, num_texts):
    url = "https://api.perplexity.ai/chat/completions"
    payload = {
        "model": "sonar",
        "messages": [
            {"role": "system", "content": "You are a meme expert. Only output the meme captions, nothing else."},
            {"role": "user", "content": f"""Generate exactly {num_texts} funny, witty, and engaging meme captions for a '{category}' meme.
The captions must be:
- Short, punchy, and interrelated.
- Form a logical and humorous conversation or reaction.
- Strictly numbered as '1.', '2.', '3.', etc. without any introduction or explanation.
ONLY output the numbered captions."""}
        ],
        "max_tokens": 200,
        "temperature": 0.8,
        "top_p": 0.9,
        "stream": False
    }

    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }

    response = requests.post(url, json=payload, headers=headers)

    if response.status_code == 200:
        raw_text = response.json().get("choices", [{}])[0].get("message", {}).get("content", "").strip()
        meme_texts = re.findall(r"^\d+\.\s*(.+)", raw_text, re.MULTILINE)
        return meme_texts[:num_texts]
    else:
        return [f"Error: {response.status_code} - {response.text}"]

def get_text_color(image, x, y, width, height):
    cropped_area = image.crop((x, y, x + width, y + height))
    avg_color = cropped_area.resize((1, 1)).getpixel((0, 0))
    brightness = (0.299 * avg_color[0] + 0.587 * avg_color[1] + 0.114 * avg_color[2])
    return "white" if brightness < 128 else "black"

def draw_text_with_outline(draw, position, text, font, text_color):
    x, y = position
    outline_color = "black" if text_color == "white" else "white"
    offsets = [-3, -2, -1, 0, 1, 2, 3]

    for dx in offsets:
        for dy in offsets:
            draw.text((x + dx, y + dy), text, font=font, fill=outline_color)

    draw.text(position, text, font=font, fill=text_color)

def wrap_text(draw, text, font, max_width):
    words = text.split()
    lines = []
    current_line = ""

    for word in words:
        test_line = current_line + " " + word if current_line else word
        bbox = draw.textbbox((0, 0), test_line, font=font)
        text_width = bbox[2] - bbox[0]

        if text_width <= max_width:
            current_line = test_line
        else:
            lines.append(current_line)
            current_line = word

    lines.append(current_line)
    return lines

def add_text_to_image(image_url, text_boxes):
    response = requests.get(image_url)
    image = Image.open(BytesIO(response.content)).convert("RGB")
    draw = ImageDraw.Draw(image)

    font_size = 50
    try:
        font = ImageFont.truetype("/content/arial.ttf", font_size)
    except IOError:
        font = ImageFont.load_default()

    for text_position in text_boxes:
        x = int(text_position.x.replace("px", "").strip())
        y = int(text_position.y.replace("px", "").strip())
        width = int(text_position.width.replace("px", "").strip())
        height = int(text_position.height.replace("px", "").strip())

        text_color = get_text_color(image, x, y, width, height)

        while True:
            wrapped_lines = wrap_text(draw, text_position.text, font, width)
            total_text_height = sum(draw.textbbox((0, 0), line, font=font)[3] - draw.textbbox((0, 0), line, font=font)[1] for line in wrapped_lines)

            if total_text_height <= height or font_size <= 20:
                break

            font_size -= 2
            try:
                font = ImageFont.truetype("/content/arial.ttf", font_size)
            except IOError:
                font = ImageFont.load_default()
                break

        text_y = y + (height - total_text_height) // 2
        for line in wrapped_lines:
            bbox = draw.textbbox((0, 0), line, font=font)
            text_width = bbox[2] - bbox[0]
            text_x = x + (width - text_width) // 2

            draw_text_with_outline(draw, (text_x, text_y), line, font, text_color)
            text_y += bbox[3] - bbox[1]

    return image

@app.post("/generate-meme/")
async def generate_meme(user_input: UserInput):
    detected_emotion = classify_emotion_with_perplexity(user_input.text)

    if detected_emotion:
        categories = detect_emotion_categories(detected_emotion)
        num_emotions = len(categories)
        
        # Generate all possible memes first
        all_meme_images = {}
        for category in categories:
            meme_texts = generate_meme_text(category, len(user_input.text_boxes))
            category_images = []
            
            for _ in range(10):
                for i, box in enumerate(user_input.text_boxes):
                    box.text = meme_texts[i] if i < len(meme_texts) else ""
                processed_image = add_text_to_image(user_input.text_boxes[0].x, user_input.text_boxes)
                category_images.append(processed_image)
            all_meme_images[category] = category_images

        # Determine final number of images
        total_images = len(categories) * 10
        final_num_images = total_images
        if total_images > 5:
            final_num_images = random.randint(5, 10)
        
        # Calculate images per emotion (x)
        images_per_emotion = final_num_images // num_emotions
        extra_images = final_num_images % num_emotions
        
        # Select images
        selected_images = []
        for category in categories:
            available_images = all_meme_images[category]
            shuffle(available_images)
            
            num_to_take = images_per_emotion + (1 if extra_images > 0 else 0)
            selected_images.extend(available_images[:num_to_take])
            extra_images -= 1 if extra_images > 0 else 0

        return {"message": "Meme generated successfully", "meme_images": selected_images}
    else:
        raise HTTPException(status_code=400, detail="Could not detect emotion")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)