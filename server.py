import base64
import io
import os
import re
import traceback
from typing import List, Optional
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import uvicorn

# 제미나이 공식 SDK 임포트
from google import genai
from google.genai import types

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

if os.path.exists("."):
    app.mount("/static", StaticFiles(directory="."), name="static")

class ChatMessage(BaseModel):
    role: str
    text: str

class ChatRequest(BaseModel):
    message: str = ""
    image: Optional[str] = None
    history: List[ChatMessage] = []

# --- 💡 서버 메모리에 캐싱할 데이터프레임 전역 변수 ---
df_symptom_cache = pd.DataFrame()
df_action_cache = pd.DataFrame()

def load_local_csv_data():
    global df_symptom_cache, df_action_cache
    try:
        symptom_path = os.path.join("data", "증상_마스터.csv")
        action_path = os.path.join("data", "조치별_검색데이터.csv")
        
        if os.path.exists(symptom_path):
            df_symptom_cache = pd.read_csv(symptom_path)
            print(f"✅ 증상_마스터.csv 로딩 완료 ({len(df_symptom_cache)}행)")
        
        if os.path.exists(action_path):
            df_action_cache = pd.read_csv(action_path)
            print(f"✅ 조치별_검색데이터.csv 로딩 완료 ({len(df_action_cache)}행)")
            
    except Exception as e:
        print(f"CSV 로딩 경고: {e}")

load_local_csv_data()

@app.get("/")
async def get_frontend():
    return FileResponse("index.html")

@app.post("/api/chat")
def chat_endpoint(req: ChatRequest):
    try:
        # Render 환경 변수에 등록된 GEMINI_API_KEY로 클라이언트 초기화
        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            return {"reply": "오류: Render 환경 변수에 GEMINI_API_KEY가 설정되지 않았습니다."}
            
        client = genai.Client(api_key=api_key)
        contents = []
        
        # --- Pandas를 활용한 사전 검색 (Pre-filtering) 로직 ---
        pandas_search_context = ""
        if req.message and not df_symptom_cache.empty and not df_action_cache.empty:
            matched = df_symptom_cache[df_symptom_cache['문제 발생(증상/질문)'].str.contains(req.message, na=False)]
            if not matched.empty:
                symptom_row = matched.iloc[0]
                s_id = symptom_row['증상_ID']
                s_name = symptom_row['문제 발생(증상/질문)']
                
                actions = df_action_cache[df_action_cache['증상_ID'] == s_id].sort_values('조치 순서')
                
                pandas_search_context = f"\n\n[내부 매뉴얼 Pandas 사전 검색 결과]\n- 감지된 증상: {s_name} (ID: {s_id})\n- 권장 조치 순서:\n"
                for _, act_row in actions.iterrows():
                    pandas_search_context += f"  * {act_row['조치 순서']}순위 ({act_row['우선순위 판단 기준']}): {act_row['해결 방법(조치 내용/답변)']}\n"

        for msg in req.history:
            role = "user" if msg.role == "user" else "model"
            contents.append(types.Content(role=role, parts=[types.Part.from_text(text=msg.text)]))

        current_parts = []
        if req.image:
            mime_type = "image/jpeg"
            image_data = req.image
            if req.image.startswith("data:"):
                header, b64_data = req.image.split(",", 1)
                mime_type = header.split(";")[0].split(":")[1]
                image_data = b64_data
            
            current_parts.append(types.Part.from_bytes(data=base64.b64decode(image_data), mime_type=mime_type))
        
        final_user_message = req.message + pandas_search_context
        if final_user_message:
            current_parts.append(types.Part.from_text(text=final_user_message))

        if current_parts:
            contents.append(types.Content(role="user", parts=current_parts))

        consumables_str = "석션디스크벨트(일반, 스페셜, 1mm, DELMA용), 링브로워필터(테라칸필터), 미디움필터, 원통형필터(L), 원통형필터(S), 테프론쿠션(L), 테프론쿠션(S), 부직포필터"

        system_instruction_text = (
            "You are an expert AI technical support assistant for NFA (엔에프에이) and a friendly AI companion.\n"
            f"- 본사: 서울 금천구 가산디지털2로 43-14 가산한화비즈메트로2차 15층 | Tel. 02-854-9500\n"
            f"- 장비 모델: ELMA, SELMA, LIMA 시리즈\n"
            f"- 소모품: {consumables_str}\n\n"
            f"- 회사 사장: 신용관\n"
            "Guidelines:\n"
            "1. 정보 우선순위: '등급 조정(Grade SOP)' 등 주요 스펙 문의는 [내부 매뉴얼 Pandas 사전 검색 결과]를 최우선으로 참고하세요.\n"
            "2. 증상 확인 필수: 사용자가 문제를 말하면 즉시 해결책을 주지 말고, '1. 기계가 멈췄어요', '2. 이상한 소리가 나요' 등 알기 쉬운 일상 용어로 번호 선택지를 먼저 제시하세요.\n"
            "3. 추가 입력 유도: 선택지 안내 시, 목록 아래에 \"원하시는 내용이 없다면 증상을 더 자세히, 구체적으로 입력해 주세요\"라는 문구를 함께 출력하여 사용자가 직접 글을 쓸 수 있도록 유도하세요.\n"
            "4. 해결책 안내: 사용자가 번호를 선택하면 그때 단계별 조치법을 간결하게 안내하세요.\n"
            "5. 강조 서식: 강조할 때는 마크다운(**) 대신 따옴표(\"텍스트\")를 사용하세요.\n"
            "6. 담당자 연결: 사용자가 연락을 원할때만 전화(02-854-9500)와 카카오톡 채널(http://pf.kakao.com/_AxiSrX/chat)을 안내하세요.\n"
            "7. 일상/기계 외 질문: 기계와 무관한 일상적인 질문이나 잡담도 친근하고 알기 쉽게 답변해 주세요.\n"
            "8. 말투: 전문용어와 장황한 설명을 배제하고, 모든 연령이 알아듣기 쉽게 아주 쉽고 짧게 답변하세요.\n"
            "9. 언어: 질문 언어에 맞춰 답변하세요."
        )

        response = client.models.generate_content(
            model='gemini-3.8-flash',
            contents=contents,
            config=types.GenerateContentConfig(
                system_instruction=system_instruction_text
            )
        )
        
        reply_text = response.text.strip() if response and response.text else "답변을 생성하지 못했습니다."
        return {"reply": reply_text}
    except Exception as e:
        traceback.print_exc()
        return {"reply": f"오류가 발생했습니다: {str(e)}"}

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)