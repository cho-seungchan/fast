from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import joblib
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from pydantic import BaseModel
from typing import List

app = FastAPI()

origins = [
    "http://13.209.68.9:10000",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 감정 분석
# ▶️ 요청 데이터 모델 정의
class FeelingCheckRequest(BaseModel):
    contents: str


# ▶️ 응답 데이터 모델 정의
class FeelingCheckResponse(BaseModel):
    score: int


# ▶️ POST 방식으로 변경된 감정 분석 API
@app.post("/api/feeling-check", response_model=FeelingCheckResponse)
async def check_spam(request: FeelingCheckRequest):

    # 감정 분류 모델 로드
    model = joblib.load("feeling_machine.pkl")

    # 예측 결과 반환
    prediction = model.predict([request.contents])[0]
    return {"score": prediction}


# 댓글 비속어 처리
# ▶️ 요청 데이터 모델 정의
class ReplyCheckRequest(BaseModel):
    content: str


# ▶️ 응답 데이터 모델 정의 (선택사항이지만 명확하게 하기 위해 추가)
class ReplyCheckResponse(BaseModel):
    isBadWord: bool


# ▶️ POST 방식으로 변경된 스팸 체크 API
@app.post("/api/reply-check", response_model=ReplyCheckResponse)
async def check_spam(request: ReplyCheckRequest):

    # 스팸 분류 모델 로드
    model = joblib.load("reply_check_model.pkl")

    # 예측 결과 반환
    prediction = model.predict([request.content])[0]
    return {"isBadWord": bool(prediction)}
