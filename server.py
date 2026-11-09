import base64
import re
import traceback
from typing import Dict, List, Optional, Tuple
import google.auth
from google.cloud import storage
from google.genai import Client, types
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import uvicorn
import io
import os
import pandas as pd

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

# --- 지식 파일 GCS 리소스 정의 ---
_KNOWLEDGE_RESOURCES = [
    {"fileName": '증상_마스터.csv', "gcsUri": 'gs://nfaaidatav261029/증상_마스터.csv', "mimeType": 'text/csv'},
    {"fileName": '조치별_검색데이터.csv', "gcsUri": 'gs://nfaaidatav261029/조치별_검색데이터.csv', "mimeType": 'text/csv'},
    {"fileName": '우선순위_규칙.csv', "gcsUri": 'gs://nfaaidatav261029/우선순위_규칙.csv', "mimeType": 'text/csv'},
    {"fileName": 'Lima_Engineering_Manual_KR_Rev1.1 (3).pdf', "gcsUri": 'gs://nfaaidatav261029/Lima_Engineering_Manual_KR_Rev1.1 (3).pdf', "mimeType": 'application/pdf'},
    {"fileName": 'NFA선별기_기계셋팅_메뉴얼.pdf', "gcsUri": 'gs://nfaaidatav261029/NFA선별기_기계셋팅_메뉴얼.pdf', "mimeType": 'application/pdf'},
    {"fileName": 'SELMA100T Grade SOP.pdf', "gcsUri": 'gs://nfaaidatav261029/SELMA100T Grade SOP.pdf', "mimeType": 'application/pdf'},
    {"fileName": 'SELMA150-H PDF 통합본.pdf', "gcsUri": 'gs://nfaaidatav261029/SELMA150-H PDF 통합본.pdf', "mimeType": 'application/pdf'},
    {"fileName": 'SELMA200-H PDF 통합본.pdf', "gcsUri": 'gs://nfaaidatav261029/SELMA200-H PDF 통합본.pdf', "mimeType": 'application/pdf'},
    {"fileName": 'sLaser Manual_kor_ver1.3.007.pdf', "gcsUri": 'gs://nfaaidatav261029/sLaser Manual_kor_ver1.3.007.pdf', "mimeType": 'application/pdf'},
    {"fileName": '등급 조정 SELMA200C SOP.pdf', "gcsUri": 'gs://nfaaidatav261029/등급 조정 SELMA200C SOP.pdf', "mimeType": 'application/pdf'},
    {"fileName": '20230116-제품 사양서 (SELMA200-H).pdf', "gcsUri": 'gs://nfaaidatav261029/20230116-제품 사양서 (SELMA200-H).pdf', "mimeType": 'application/pdf'},
    {"fileName": '260323 카메라 사양서.pdf', "gcsUri": 'gs://nfaaidatav261029/260323 카메라 사양서.pdf', "mimeType": 'application/pdf'},
    {"fileName": '등급 조정 SELMA200T SOP.pdf', "gcsUri": 'gs://nfaaidatav261029/등급 조정 SELMA200T SOP.pdf', "mimeType": 'application/pdf'},
    {"fileName": '제품 사양서 (LIMA200-H).pdf', "gcsUri": 'gs://nfaaidatav261029/제품 사양서 (LIMA200-H).pdf', "mimeType": 'application/pdf'},
    {"fileName": '제품 사양서 (SELMA150P-H).pdf', "gcsUri": 'gs://nfaaidatav261029/제품 사양서 (SELMA150P-H).pdf', "mimeType": 'application/pdf'},
    {"fileName": '20261023 하반기 LIMA_PT.pdf', "gcsUri": 'gs://nfaaidatav261029/20261023 하반기 LIMA_PT.pdf', "mimeType": 'application/pdf'},
]

_KNOWLEDGE_GCS_URI_RE = re.compile(r"^gs://([^/]+)/(.+)$")

def _parse_knowledge_gcs_uri(uri: str) -> Tuple[str, str]:
    match = _KNOWLEDGE_GCS_URI_RE.match(uri)
    if not match:
        raise ValueError(f"Invalid GCS URI: {uri}")
    return match.group(1), match.group(2)

_UPLOADED_FILE_PARTS = []

# 💡 Pandas로 검색하기 위해 서버 메모리에 캐싱할 데이터프레임 전역 변수
df_symptom_cache = pd.DataFrame()
df_action_cache = pd.DataFrame()

def preload_and_upload_files_to_gemini():
    global _UPLOADED_FILE_PARTS, df_symptom_cache, df_action_cache
    try:
        credentials, _ = google.auth.default(quota_project_id="305478931531")
        storage_client = storage.Client(project="305478931531", credentials=credentials)
        client = Client(vertexai=True, location="global")
        
        for kr in _KNOWLEDGE_RESOURCES:
            bucket_name, blob_name = _parse_knowledge_gcs_uri(kr["gcsUri"])
            blob = storage_client.bucket(bucket_name).blob(blob_name)
            data = blob.download_as_bytes()
            if data:
                # 1. 제미나이 업로드용 파일 파트 생성
                uploaded_file = client.files.upload(
                    file=io.BytesIO(data),
                    config=types.UploadFileConfig(
                        display_name=kr["fileName"],
                        mime_type=kr["mimeType"]
                    )
                )
                _UPLOADED_FILE_PARTS.append(uploaded_file)
                print(f"Uploaded knowledge file to Gemini: {kr['fileName']} (URI: {uploaded_file.uri})")
                
                # 2. CSV 파일인 경우 Pandas 캐싱을 위해 메모리에 로드
                if kr["fileName"] == '증상_마스터.csv':
                    df_symptom_cache = pd.read_csv(io.BytesIO(data))
                elif kr["fileName"] == '조치별_검색데이터.csv':
                    df_action_cache = pd.read_csv(io.BytesIO(data))
                    
        print("✅ Pandas 매뉴얼 데이터 캐싱 완료 (증상 마스터 & 조치 데이터)")
    except Exception as e:
        print(f"Gemini file upload warning: {e}")

preload_and_upload_files_to_gemini()

@app.get("/")
async def get_frontend():
    return FileResponse("index.html")

@app.post("/api/chat")
def chat_endpoint(req: ChatRequest):
    try:
        client = Client(vertexai=True, location="global")
        contents = []
        
        # --- 💡 Pandas를 활용한 사전 검색 (Pre-filtering) 로직 ---
        pandas_search_context = ""
        if req.message and not df_symptom_cache.empty and not df_action_cache.empty:
            # 사용자의 질문 키워드가 포함된 증상 검색
            matched = df_symptom_cache[df_symptom_cache['문제 발생(증상/질문)'].str.contains(req.message, na=False)]
            if not matched.empty:
                symptom_row = matched.iloc[0]
                s_id = symptom_row['증상_ID']
                s_name = symptom_row['문제 발생(증상/질문)']
                
                # 해당 증상 ID의 조치 방법들을 '조치 순서' 순으로 정렬
                actions = df_action_cache[df_action_cache['증상_ID'] == s_id].sort_values('조치 순서')
                
                pandas_search_context = f"\n\n[내부 매뉴얼 Pandas 사전 검색 결과]\n- 감지된 증상: {s_name} (ID: {s_id})\n- 권장 조치 순서:\n"
                for _, act_row in actions.iterrows():
                    pandas_search_context += f"  * {act_row['조치 순서']}순위 ({act_row['우선순위 판단 기준']}): {act_row['해결 방법(조치 내용/답변)']}\n"
        # ----------------------------------------------------

        for i, msg in enumerate(req.history):
            role = "user" if msg.role == "user" else "model"
            parts = []
            if i == 0 and role == "user" and _UPLOADED_FILE_PARTS:
                parts.extend(_UPLOADED_FILE_PARTS)
            parts.append(types.Part.from_text(text=msg.text))
            contents.append(types.Content(role=role, parts=parts))

        current_parts = []
        if not req.history and _UPLOADED_FILE_PARTS:
            current_parts.extend(_UPLOADED_FILE_PARTS)

        if req.image:
            mime_type = "image/jpeg"
            image_data = req.image
            if req.image.startswith("data:"):
                header, b64_data = req.image.split(",", 1)
                mime_type = header.split(";")[0].split(":")[1]
                image_data = b64_data
            
            current_parts.append(types.Part.from_bytes(data=base64.b64decode(image_data), mime_type=mime_type))
        
        # 사용자 메시지에 Pandas가 찾아낸 정밀 매뉴얼 결과를 덧붙여서 전달
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
            "1. 정보 우선순위: '등급 조정(Grade SOP)' 등 주요 스펙 문의는 첨부된 PDF 매뉴얼과 [내부 매뉴얼 Pandas 사전 검색 결과]를 최우선으로 참고하세요.\n"
            "2. 증상 확인 필수: 사용자가 문제를 말하면 즉시 해결책을 주지 말고, '1. 기계가 멈췄어요', '2. 이상한 소리가 나요' 등 알기 쉬운 일상 용어로 번호 선택지를 먼저 제시하세요.\n"
            "3. 추가 입력 유도: 선택지 안내 시, 목록 아래에 \"원하시는 내용이 없다면 증상을 더 자세히, 구체적으로 입력해 주세요\"라는 문구를 함께 출력하여 사용자가 직접 글을 쓸 수 있도록 유도하세요.\n"
            "4. 해결책 안내: 사용자가 번호를 선택하면 그때 단계별 조치법을 간결하게 안내하세요.\n"
            "5. 강조 서식: 강조할 때는 마크다운(**) 대신 따옴표(\"텍스트\")를 사용하세요.\n"
            "6. 담당자 연결: 사용자가 연락을 원할때만 전화(02-854-9500)와 카카오톡 채널(http://pf.kakao.com/_AxiSrX/chat)을 안내하세요.\n"
            "7. 일상/기계 외 질문: 기계와 무관한 일상적인 질문이나 잡담도 친근하고 알기 쉽게 답변해 주세요.\n"
            "8. 말투: 전문용어와 장황한 설명을 배제하고, 모든 연령이 알아듣기 쉽게 아주 쉽고 짧게 답변하세요."
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
    uvicorn.run(app, host="127.0.0.1", port=8000, timeout_keep_alive=120)