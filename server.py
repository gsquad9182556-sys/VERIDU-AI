from fastapi import FastAPI, APIRouter, File, UploadFile, HTTPException, Form
from fastapi.responses import JSONResponse
from dotenv import load_dotenv
from starlette.middleware.cors import CORSMiddleware
from motor.motor_asyncio import AsyncIOMotorClient
import os
import logging
from pathlib import Path
from pydantic import BaseModel, Field
from typing import List, Optional
import uuid
from datetime import datetime, timezone
import asyncio
import base64
import json
from emergentintegrations.llm.chat import LlmChat, UserMessage
import cv2
import numpy as np
from PIL import Image
import io
import re

ROOT_DIR = Path(__file__).parent
load_dotenv(ROOT_DIR / '.env')

mongo_url = os.environ['MONGO_URL']
client = AsyncIOMotorClient(mongo_url)
db = client[os.environ['DB_NAME']]

app = FastAPI()
api_router = APIRouter(prefix="/api")
llm_key = os.environ.get('EMERGENT_LLM_KEY')

class ContentAnalysisRequest(BaseModel):
    content_type: str
    text_content: Optional[str] = None
    file_data: Optional[str] = None
    filename: Optional[str] = None

class AnalysisResult(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    content_type: str
    authenticity_score: float
    confidence_level: float
    analysis_details: dict
    risk_factors: List[str]
    recommendations: str
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    filename: Optional[str] = None

class AnalysisHistory(BaseModel):
    id: str
    content_type: str
    authenticity_score: float
    confidence_level: float
    risk_factors: List[str]
    timestamp: datetime
    filename: Optional[str] = None

def prepare_for_mongo(data):
    if isinstance(data.get('timestamp'), datetime):
        data['timestamp'] = data['timestamp'].isoformat()
    return data

def parse_from_mongo(item):
    if isinstance(item.get('timestamp'), str):
        item['timestamp'] = datetime.fromisoformat(item['timestamp'])
    return item

async def analyze_text_content(text: str) -> dict:
    try:
        chat = LlmChat(
            api_key=llm_key,
            session_id=f"analysis_{uuid.uuid4()}",
            system_message="""You are an expert fake news detector and misinformation analyst. 
            Analyze the given text for authenticity, bias, misinformation patterns, and credibility indicators.
            Provide analysis in this exact JSON format:
            {
                "authenticity_score": 85.5,
                "confidence_level": 92.0,
                "risk_factors": ["emotional_language", "lack_of_sources", "sensational_claims"],
                "analysis_details": {
                    "credibility_indicators": ["specific_dates", "named_sources"],
                    "red_flags": ["unverified_claims", "biased_language"],
                    "language_analysis": "professional tone with some emotional elements",
                    "source_evaluation": "no verifiable sources mentioned"
                },
                "recommendations": "Cross-reference with multiple reliable news sources before sharing"
            }
            Score authenticity from 0-100 (0=definitely fake, 100=highly authentic).
            Consider: source credibility, fact-checkable claims, emotional manipulation, logical consistency, writing quality."""
        ).with_model("anthropic", "claude-3-7-sonnet-20250219")
        analysis_message = UserMessage(
            text=f"Analyze this content for fake news patterns and authenticity:\n\n{text}"
        )
        response = await chat.send_message(analysis_message)
        json_match = re.search(r'\{[\s\S]*\}', response)
        if json_match:
            result = json.loads(json_match.group())
        else:
            result = {
                "authenticity_score": 50.0,
                "confidence_level": 60.0,
                "risk_factors": ["analysis_error"],
                "analysis_details": {"error": "Could not parse AI response"},
                "recommendations": "Manual review recommended due to analysis error"
            }
        return result
    except Exception as e:
        logger.error(f"Text analysis error: {str(e)}")
        return {
            "authenticity_score": 50.0,
            "confidence_level": 30.0,
            "risk_factors": ["analysis_failed"],
            "analysis_details": {"error": str(e)},
            "recommendations": "Analysis failed - manual review required"
        }

async def analyze_image_content(image_data: bytes, filename: str) -> dict:
    try:
        image = Image.open(io.BytesIO(image_data))
        img_array = np.array(image)
        if len(img_array.shape) == 3:
            img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
        analysis_details = {}
        risk_factors = []
        height, width = img_array.shape[:2]
        analysis_details["dimensions"] = f"{width}x{height}"
        gray = cv2.cvtColor(img_array, cv2.COLOR_BGR2GRAY)
        blur_score = cv2.Laplacian(gray, cv2.CV_64F).var()
        analysis_details["sharpness_score"] = float(blur_score)
        if blur_score < 100:
            risk_factors.append("low_image_quality")
        edges = cv2.Canny(gray, 50, 150)
        edge_density = np.sum(edges > 0) / (width * height)
        analysis_details["edge_density"] = float(edge_density)
        if edge_density < 0.01 or edge_density > 0.3:
            risk_factors.append("unusual_edge_patterns")
        authenticity_score = 70.0
        if blur_score > 200:
            authenticity_score += 10
        elif blur_score < 50:
            authenticity_score -= 20
        if 0.02 <= edge_density <= 0.25:
            authenticity_score += 10
        else:
            authenticity_score -= 15
        if filename.lower().endswith(('.jpg', '.jpeg')):
            analysis_details["format"] = "JPEG"
        elif filename.lower().endswith('.png'):
            analysis_details["format"] = "PNG"
            authenticity_score += 5
        else:
            analysis_details["format"] = "Other"
        authenticity_score = max(0, min(100, authenticity_score))
        return {
            "authenticity_score": authenticity_score,
            "confidence_level": 65.0,
            "risk_factors": risk_factors,
            "analysis_details": analysis_details,
            "recommendations": "Consider reverse image search and metadata analysis for deeper verification"
        }
    except Exception as e:
        logger.error(f"Image analysis error: {str(e)}")
        return {
            "authenticity_score": 50.0,
            "confidence_level": 30.0,
            "risk_factors": ["analysis_failed"],
            "analysis_details": {"error": str(e)},
            "recommendations": "Image analysis failed - manual review required"
        }

async def analyze_video_content(video_data: bytes, filename: str) -> dict:
    try:
        temp_path = f"/tmp/{uuid.uuid4()}.mp4"
        with open(temp_path, "wb") as f:
            f.write(video_data)
        cap = cv2.VideoCapture(temp_path)
        if not cap.isOpened():
            return {
                "authenticity_score": 50.0,
                "confidence_level": 30.0,
                "risk_factors": ["video_read_error"],
                "analysis_details": {"error": "Could not read video file"},
                "recommendations": "Video format not supported or corrupted"
            }
        analysis_details = {}
        risk_factors = []
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = frame_count / fps if fps > 0 else 0
        analysis_details["fps"] = fps
        analysis_details["frame_count"] = frame_count
        analysis_details["duration_seconds"] = duration
        frames_analyzed = 0
        blur_scores = []
        while frames_analyzed < min(10, frame_count):
            ret, frame = cap.read()
            if not ret:
                break
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            blur_score = cv2.Laplacian(gray, cv2.CV_64F).var()
            blur_scores.append(blur_score)
            frames_analyzed += 1
            if frame_count > 10:
                cap.set(cv2.CAP_PROP_POS_FRAMES,
                       int((frames_analyzed * frame_count) / 10))
        cap.release()
        os.remove(temp_path)
        if blur_scores:
            avg_blur = np.mean(blur_scores)
            blur_variance = np.var(blur_scores)
            analysis_details["average_sharpness"] = float(avg_blur)
            analysis_details["sharpness_variance"] = float(blur_variance)
            if blur_variance > 1000:
                risk_factors.append("inconsistent_quality")
            if avg_blur < 50:
                risk_factors.append("low_video_quality")
        authenticity_score = 60.0
        if fps >= 24 and fps <= 60:
            authenticity_score += 10
        elif fps < 15 or fps > 120:
            authenticity_score -= 15
            risk_factors.append("unusual_framerate")
        if duration > 5:
            authenticity_score += 5
        authenticity_score = max(0, min(100, authenticity_score))
        return {
            "authenticity_score": authenticity_score,
            "confidence_level": 55.0,
            "risk_factors": risk_factors,
            "analysis_details": analysis_details,
            "recommendations": "Video authenticity requires specialized deepfake detection tools for higher accuracy"
        }
    except Exception as e:
        logger.error(f"Video analysis error: {str(e)}")
        return {
            "authenticity_score": 50.0,
            "confidence_level": 30.0,
            "risk_factors": ["analysis_failed"],
            "analysis_details": {"error": str(e)},
            "recommendations": "Video analysis failed - manual review required"
        }

@api_router.post("/analyze", response_model=AnalysisResult)
async def analyze_content(
    content_type: str = Form(...),
    text_content: Optional[str] = Form(None),
    file: Optional[UploadFile] = File(None)
):
    try:
        if content_type == "text":
            if not text_content:
                raise HTTPException(status_code=400, detail="Text content is required")
            analysis = await analyze_text_content(text_content)
            result = AnalysisResult(
                content_type="text",
                authenticity_score=analysis["authenticity_score"],
                confidence_level=analysis["confidence_level"],
                analysis_details=analysis["analysis_details"],
                risk_factors=analysis["risk_factors"],
                recommendations=analysis["recommendations"]
            )
        elif content_type in ["image", "video"]:
            if not file:
                raise HTTPException(status_code=400, detail="File is required for image/video analysis")
            file_data = await file.read()
            if content_type == "image":
                analysis = await analyze_image_content(file_data, file.filename)
            else:
                analysis = await analyze_video_content(file_data, file.filename)
            result = AnalysisResult(
                content_type=content_type,
                authenticity_score=analysis["authenticity_score"],
                confidence_level=analysis["confidence_level"],
                analysis_details=analysis["analysis_details"],
                risk_factors=analysis["risk_factors"],
                recommendations=analysis["recommendations"],
                filename=file.filename
            )
        else:
            raise HTTPException(status_code=400, detail="Invalid content_type. Use 'text', 'image', or 'video'")
        result_dict = prepare_for_mongo(result.dict())
        await db.analysis_results.insert_one(result_dict)
        return result
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Analysis error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")

@api_router.get("/history", response_model=List[AnalysisHistory])
async def get_analysis_history(limit: int = 50):
    try:
        results = await db.analysis_results.find().sort("timestamp", -1).limit(limit).to_list(length=None)
        history = []
        for result in results:
            parsed_result = parse_from_mongo(result)
            history_item = AnalysisHistory(
                id=parsed_result["id"],
                content_type=parsed_result["content_type"],
                authenticity_score=parsed_result["authenticity_score"],
                confidence_level=parsed_result["confidence_level"],
                risk_factors=parsed_result["risk_factors"],
                timestamp=parsed_result["timestamp"],
                filename=parsed_result.get("filename")
            )
            history.append(history_item)
        return history
    except Exception as e:
        logger.error(f"History retrieval error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to retrieve history: {str(e)}")

@api_router.get("/stats")
async def get_analysis_stats():
    try:
        total_count = await db.analysis_results.count_documents({})
        text_count = await db.analysis_results.count_documents({"content_type": "text"})
        image_count = await db.analysis_results.count_documents({"content_type": "image"})
        video_count = await db.analysis_results.count_documents({"content_type": "video"})
        pipeline = [
            {"$group": {
                "_id": None,
                "avg_authenticity": {"$avg": "$authenticity_score"},
                "avg_confidence": {"$avg": "$confidence_level"}
            }}
        ]
        avg_results = await db.analysis_results.aggregate(pipeline).to_list(1)
        avg_authenticity = avg_results["avg_authenticity"] if avg_results else 0
        avg_confidence = avg_results["avg_confidence"] if avg_results else 0
        return {
            "total_analyses": total_count,
            "content_types": {
                "text": text_count,
                "image": image_count,
                "video": video_count
            },
            "averages": {
                "authenticity_score": round(avg_authenticity, 2),
                "confidence_level": round(avg_confidence, 2)
            }
        }
    except Exception as e:
        logger.error(f"Stats error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get stats: {str(e)}")

@api_router.get("/")
async def root():
    return {"message": "Truth Detector API - Fake News Detection System"}

app.include_router(api_router)
app.add_middleware(
    CORSMiddleware,
    allow_credentials=True,
    allow_origins=os.environ.get('CORS_ORIGINS', '*').split(','),
    allow_methods=["*"],
    allow_headers=["*"],
)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@app.on_event("shutdown")
async def shutdown_db_client():
    client.close()
