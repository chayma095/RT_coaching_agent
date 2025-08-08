from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Any
import google.generativeai as genai
import os
from dotenv import load_dotenv

# ------------------ Setup ------------------
load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    raise RuntimeError("GEMINI_API_KEY not found in environment variables")
genai.configure(api_key=api_key)

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ------------------ Input models ------------------
class AudioModelOutput(BaseModel):
    timestamp: str
    predicted_emotion: str
    librosa_features: Dict[str, float]

class FacialModelOutput(BaseModel):
    timestamp: str
    predicted_emotion: str
    expression_scores: Dict[str, float]
    # Add any facial analysis raw fields your model outputs

class CombinedMetrics(BaseModel):
    timestamp: str
    emotional_metrics: Dict[str, float]
    sentiment_analysis: Dict[str, Any]
    behavioral_data: Dict[str, Any]
    voice_analysis: Dict[str, float]
    facial_analysis: Dict[str, Any]

# ------------------ Step 1: Merge through LLM ------------------
def build_merge_prompt(audio_data: dict, facial_data: dict) -> str:
    return f"""
You are given two JSON objects:
1. Audio emotion analysis result:
{audio_data}

2. Facial sentiment analysis result:
{facial_data}

Using both sources, produce a single JSON matching this structure exactly:

{{
  "timestamp": "<from input>",
  "user_id": "user_alice_123",
  "session_id": "session_001",
  "emotional_metrics": {{
    "confidence_level": <float>,
    "stress_level": <float>,
    "engagement_score": <float>,
    "emotional_stability": <float>,
    "voice_clarity": <float>,
    "facial_expression_score": <float>
  }},
  "sentiment_analysis": {{
    "overall_sentiment": "<positive|neutral|negative>",
    "sentiment_score": <float between 0 and 1>,
    "emotional_tone": "<string>",
    "detected_emotions": [<list of strings>]
  }},
  "behavioral_data": {{
    "filler_words_count": <int>,
    "speech_pace": <float>,
    "gesture_analysis": "<string>",
    "eye_contact_score": <float>
  }},
  "voice_analysis": {{
    "tone_consistency": <float>,
    "volume_variation": <float>,
    "pitch_stability": <float>,
    "articulation_clarity": <float>
  }},
  "facial_analysis": {{
    "smile_frequency": <float>,
    "eye_engagement": <float>,
    "micro_expressions": [<list of strings>],
    "overall_expressiveness": <float>
  }}
}}

Rules:
- Populate all numeric fields with reasonable values inferred from input.
- Keep timestamp from audio data.
- Ensure valid JSON, no comments, no extra text.
    """

@app.post("/merge-data")
async def merge_data(audio: AudioModelOutput, facial: FacialModelOutput):
    prompt = build_merge_prompt(audio.dict(), facial.dict())
    try:
        model = genai.GenerativeModel("gemini-2.5-flash-lite")
        response = model.generate_content(prompt)
        merged_json = response.text.strip()
        # Optionally, validate JSON here
        return {"merged_data": merged_json}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Merge LLM failed: {str(e)}")

# ------------------ Step 2: Coaching tip ------------------
@app.post("/generate-tip")
async def generate_tip(combined_data: CombinedMetrics):
    prompt = build_prompt(combined_data)  # reuse your existing prompt builder
    try:
        model = genai.GenerativeModel("gemini-2.5-flash-lite")
        response = model.generate_content(prompt)
        coaching_message = response.text.strip()
        return {"tip": coaching_message}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Tip LLM failed: {str(e)}")

# ------------------ Prompt builder for tips ------------------
def build_prompt(data: CombinedMetrics) -> str:
    return f"""
You are a real-time coaching assistant for public speaking. Given the following data about the speaker, generate a brief, supportive, and actionable pop-up coaching message.

Timestamp: {data.timestamp}

Emotional Metrics: {data.emotional_metrics}
Sentiment Analysis: {data.sentiment_analysis}
Behavioral Data: {data.behavioral_data}
Voice Analysis: {data.voice_analysis}
Facial Analysis: {data.facial_analysis}

Generate one short actionable sentence (max 15 words).
"""
