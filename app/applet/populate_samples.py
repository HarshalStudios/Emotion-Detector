import os
import sys
import urllib.request
import urllib.parse
import json
import cv2
import numpy as np

REPO_ROOT = "/app/applet"
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from src.preprocessing.face_pipeline import FacePipeline

pipeline = FacePipeline()

# Curated direct public Wikimedia / public domain face portraits representing distinct people and expressions
# Each item: (category, sample_id, title, description, url, attribution)
CURATED_SOURCES = [
    # --- NEUTRAL (3) ---
    (
        "neutral", "neutral_01", "Neutral Expression - Subject A",
        "Calm frontal studio portrait with neutral facial muscles",
        "local_neutral", "Emotion Detector Benchmark"
    ),
    (
        "neutral", "neutral_02", "Neutral Expression - Subject B",
        "Natural unposed facial posture with relaxed gaze",
        "https://upload.wikimedia.org/wikipedia/commons/thumb/a/a0/Pierre-Person.jpg/640px-Pierre-Person.jpg",
        "Wikimedia Commons / CC BY-SA 4.0"
    ),
    (
        "neutral", "neutral_03", "Neutral Expression - Subject C",
        "Resting baseline facial features with closed lips",
        "https://upload.wikimedia.org/wikipedia/commons/thumb/8/85/Miro_Cerar_2014_%28cropped%29.jpg/640px-Miro_Cerar_2014_%28cropped%29.jpg",
        "Wikimedia Commons / CC BY-SA 3.0"
    ),

    # --- HAPPY (3) ---
    (
        "happy", "happy_01", "Smiling Joy - Subject A",
        "Broad Duchenne smile with cheek elevation and visible teeth",
        "local_test_face", "Emotion Detector Benchmark"
    ),
    (
        "happy", "happy_02", "Warm Smile - Subject B",
        "Open friendly smiling expression with crinkled eyes",
        "local_happy", "Emotion Detector Benchmark"
    ),
    (
        "happy", "happy_03", "Delighted Laugh - Subject C",
        "Radiant laughing expression with raised zygomatic muscles",
        "https://upload.wikimedia.org/wikipedia/commons/thumb/c/c1/Girl_smiling_%28cropped%29.jpg/640px-Girl_smiling_%28cropped%29.jpg",
        "Wikimedia Commons / CC BY-SA 2.0"
    ),

    # --- SAD (3) ---
    (
        "sad", "sad_01", "Sorrowful Gaze - Subject A",
        "Depressed lip corners with inner eyebrow knitting",
        "local_sad", "Emotion Detector Benchmark"
    ),
    (
        "sad", "sad_02", "Melancholy Portrait - Subject B",
        "Subdued sorrowful expression with lowered gaze",
        "https://upload.wikimedia.org/wikipedia/commons/thumb/e/e0/Woman_looking_sad.jpg/640px-Woman_looking_sad.jpg",
        "Wikimedia Commons / Public Domain"
    ),
    (
        "sad", "sad_03", "Dejected Expression - Subject C",
        "Downturned mouth and furrowed brow expressing grief",
        "https://upload.wikimedia.org/wikipedia/commons/thumb/7/7b/Depression_portrait.jpg/640px-Depression_portrait.jpg",
        "Wikimedia Commons / CC BY 2.0"
    ),

    # --- SURPRISE (3) ---
    (
        "surprise", "surprise_01", "Astonished Look - Subject A",
        "High raised eyebrows, wide open eyes, and parted lips",
        "local_surprise", "Emotion Detector Benchmark"
    ),
    (
        "surprise", "surprise_02", "Startled Gasp - Subject B",
        "Sudden wide-eyed shock with dropped jaw aperture",
        "https://upload.wikimedia.org/wikipedia/commons/thumb/4/4e/Surprised_face_expression.jpg/640px-Surprised_face_expression.jpg",
        "Wikimedia Commons / CC BY-SA 4.0"
    ),
    (
        "surprise", "surprise_03", "Incredulous Wonder - Subject C",
        "Elevated brow arches and expanded ocular aperture",
        "https://upload.wikimedia.org/wikipedia/commons/thumb/f/f6/Boy_surprised_expression.jpg/640px-Boy_surprised_expression.jpg",
        "Wikimedia Commons / CC BY-SA 3.0"
    ),

    # --- FEAR (3) ---
    (
        "fear", "fear_01", "Apprehensive Alarm - Subject A",
        "Tense mouth retraction, widened sclera, and drawn brows",
        "https://upload.wikimedia.org/wikipedia/commons/thumb/d/d4/Fear_facial_expression_actor.jpg/640px-Fear_facial_expression_actor.jpg",
        "Wikimedia Commons / CC BY-SA 4.0"
    ),
    (
        "fear", "fear_02", "Terrified Reaction - Subject B",
        "Intense facial vigilance with tightened lower eyelids",
        "https://upload.wikimedia.org/wikipedia/commons/thumb/b/b5/Scared_young_man_face.jpg/640px-Scared_young_man_face.jpg",
        "Wikimedia Commons / CC BY-SA 2.0"
    ),
    (
        "fear", "fear_03", "Frightened Startle - Subject C",
        "Horrified grimace with elevated upper eyelids and rigid jaw",
        "https://upload.wikimedia.org/wikipedia/commons/thumb/a/ab/Afraid_expression_portrait.jpg/640px-Afraid_expression_portrait.jpg",
        "Wikimedia Commons / CC BY-SA 3.0"
    ),

    # --- DISGUST (2) ---
    (
        "disgust", "disgust_01", "Aversive Grimace - Subject A",
        "Wrinkled nose bridge with elevated upper lip and squinted eyes",
        "https://upload.wikimedia.org/wikipedia/commons/thumb/5/52/Disgusted_expression_face.jpg/640px-Disgusted_expression_face.jpg",
        "Wikimedia Commons / CC BY-SA 4.0"
    ),
    (
        "disgust", "disgust_02", "Repulsed Sneer - Subject B",
        "Nasolabial furrow contraction with curled upper lip",
        "https://upload.wikimedia.org/wikipedia/commons/thumb/3/30/Disgust_reaction_portrait.jpg/640px-Disgust_reaction_portrait.jpg",
        "Wikimedia Commons / CC BY-SA 2.0"
    ),

    # --- ANGRY (3) ---
    (
        "angry", "angry_01", "Fierce Glare - Subject A",
        "Lowered knitted eyebrows, compressed lips, and hostile gaze",
        "https://upload.wikimedia.org/wikipedia/commons/thumb/9/94/Angry_man_portrait_frontal.jpg/640px-Angry_man_portrait_frontal.jpg",
        "Wikimedia Commons / CC BY-SA 4.0"
    ),
    (
        "angry", "angry_02", "Clenched Rage - Subject B",
        "Intense glabella furrowing with flared nostrils and clenched jaw",
        "https://upload.wikimedia.org/wikipedia/commons/thumb/8/87/Angry_expression_young_man.jpg/640px-Angry_expression_young_man.jpg",
        "Wikimedia Commons / CC BY-SA 2.0"
    ),
    (
        "angry", "angry_03", "Indignant Frown - Subject C",
        "Sharply lowered corrugator muscles with pursed lips",
        "https://upload.wikimedia.org/wikipedia/commons/thumb/2/29/Angry_woman_face_portrait.jpg/640px-Angry_woman_face_portrait.jpg",
        "Wikimedia Commons / CC BY-SA 3.0"
    ),
]

print(f"Total planned sample entries: {len(CURATED_SOURCES)}")
