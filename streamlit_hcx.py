import streamlit as st
import fitz  # pymupdf
import numpy as np
import re
import http.client
import requests
import json

class Embedding:
    def __init__(self, host, api_key, request_id):
        self._host = host
        self._api_key = api_key
        self._request_id = request_id

    def _send_request(self, completion_request):
        headers = {
            'Content-Type': 'application/json; charset=utf-8',
            'Authorization': self._api_key,
            'X-NCP-CLOVASTUDIO-REQUEST-ID': self._request_id
        }

        conn = http.client.HTTPSConnection(self._host)
        conn.request('POST', '/v1/api-tools/embedding/v2', json.dumps(completion_request), headers)
        response = conn.getresponse()
        result = json.loads(response.read().decode(encoding='utf-8'))
        conn.close()
        
        return result

    def execute(self, completion_request):
        res = self._send_request(completion_request)
        # print(res)
        if res['status']['code'] == '20000':
            return res['result']
        else:
            return 'Error'

class CompletionExecutor:
    def __init__(self, host, api_key, request_id):
        self._host = host
        self._api_key = api_key
        self._request_id = request_id
        
    def execute(self, completion_request):
        # print("여긴 오나?")
        # print(completion_request)
        headers = {
            'Authorization': self._api_key,
            'X-NCP-CLOVASTUDIO-REQUEST-ID': self._request_id,
            'Content-Type': 'application/json; charset=utf-8',
            'Accept': 'text/event-stream'
        }
        full_content = ""
        with requests.post(self._host + '/v3/chat-completions/HCX-007',
                           headers=headers, json=completion_request, stream=True) as r:
            # print(f"HTTP 상태코드: {r.status_code}")  # 200
            # r.raise_for_status()
            for line in r.iter_lines():
                # print(line)
                if line:
                    decoded = line.decode("utf-8").strip()
                    # print(type(decoded), decoded)
                    if not decoded.startswith("data:"):
                        continue
                    json_part = decoded[6:].strip()
                    # print(json_part)
                    if '"data":"[DONE]"' in json_part:
                        # print("DONE")
                        break
                    try:
                        # print(json_part)
                        if not json_part.startswith("{"):
                            # "message":{...} 형태면 → {"message":{...}} 로 만들어야 함
                            if json_part.startswith('"message":'):
                                json_part = "{" + json_part
                        try:
                            data = json.loads(json_part)
                        except json.JSONDecodeError as e:
                            print("JSON 파싱 실패")
                            print(f"오류 메시지: {e.msg}")
                            print(f"문제 되는 부분: →{json_part[max(0, e.pos-20):e.pos+20]}←")
                            print(f"전체 문자열: {json_part}")
                        # print(data)
                        message = data.get("message", {})
                        
                        # 핵심: content만 추출 (thinkingContent는 무시!)
                        content = message.get("content", "")
                        if content:                     # 빈 문자열은 무시
                            full_content += content
                            
                    except json.JSONDecodeError:
                        continue
                    
        # print(full_content)
        return {"result": {"message": {"content": full_content.strip()}}}

# 텍스트 추출 함수 : PDF의 모든 페이지 텍스트 추출 
def extract_text_from_pdf(pdf_data):
    document = fitz.open(stream=pdf_data, filetype="pdf")
    text = ""
    for page in document:
        text += page.get_text() + "\n\n"
    document.close()
    return text

# 텍스트 청킹 함수 (문단 단위, ~500자)
def chunk_text(text, chunk_size=500):
    paragraphs = re.split(r'\n\s*\n', text) # 문단 단위 분할 
    chunks = []
    current_chunk = ""
    for para in paragraphs:
        if len(current_chunk) + len(para) < chunk_size:
            current_chunk += para + "\n\n" # 청크 크기 제한
        else:
            if current_chunk:
                chunks.append(current_chunk.strip())
            current_chunk = para + "\n\n"
    if current_chunk:
        chunks.append(current_chunk.strip())
    return chunks

# 임베딩 생성 함수 : 각 청크를 1536 차원 벡터로 변환 
def get_hcx_embeddings(key, texts):
    completion_executor = Embedding(
        host='clovastudio.stream.ntruss.com',
        api_key=f'Bearer {key}',
        request_id='543d766ecc044afb9b3d3835e188f00b'
    )
    # print(texts)
    # request_data = {"text": texts[0]}
    all_embeddings = []
    
    for i, text in enumerate(texts):
        # 청크가 너무 길면 자르기 (HCX 토큰 제한 대비, 3000자 안전선)
        if len(text) > 3000:
            text = text[:3000] + "..."  # 끝에 마커 추가
        
        request_data = {"text": text}  # string으로만!
        
        for retry in range(3):  # API 불안정 대비 재시도
            try:
                raw_result = completion_executor.execute(request_data)
                if raw_result != 'Error' and 'embedding' in raw_result:
                    all_embeddings.append(raw_result["embedding"])
                    # print(f"청크 {i+1}/{len(texts)} 임베딩 성공 (토큰: {raw_result.get('inputTokens', 'N/A')})")
                    break
                else:
                    print(f"청크 {i+1} 응답 오류: {raw_result}")
            except Exception as e:
                print(f"청크 {i+1} 예외: {e}")
            
            if retry < 2:
                time.sleep(1)  # 1초 대기 후 재시도 (API rate limit 대비)
        
        if len(all_embeddings) <= i:  # 실패 시
            raise ValueError(f"청크 {i+1} 임베딩 완전 실패")
    
    return np.array(all_embeddings, dtype=np.float32)  # 이제 (n_chunks, 1536)

# 코사인 유사도 검색 (FAISS 대체)
def search_chunks(query_embedding, chunk_embeddings, chunks, top_k=0):
    # 코사인 유사도 계산 
    # top k : 다양한 후보 중에서 확률 값이 가장 높은 K개 중 하나를 선택하게 하는 기준값
    similarities = np.dot(chunk_embeddings, query_embedding) / (
        np.linalg.norm(chunk_embeddings, axis=1) * np.linalg.norm(query_embedding)
    )
    # 상위 k개의 인덱스 반환 
    top_indices = np.argsort(similarities)[-top_k:][::-1]
    return [chunks[idx] for idx in top_indices]

def main():
    st.set_page_config(layout="wide")
    with st.sidebar:
        st.title("PDF 기반 RAG 시스템")
        hcx_api_key = st.text_input("HCX API Key 설정", type="password")
        pdf_file = st.file_uploader("PDF 파일을 업로드하세요.", type=["pdf"])
        
        # 세션 상태 초기화 (메모리 유지)
        if "chunks" not in st.session_state:
            st.session_state.chunks = []
        if "chunk_embeddings" not in st.session_state:
            st.session_state.chunk_embeddings = None
        if "messages" not in st.session_state:
            st.session_state.messages = []
        if "pdf_processed" not in st.session_state:
            st.session_state.pdf_processed = False

    if pdf_file and not st.session_state.pdf_processed:
        pdf_data = pdf_file.read()
        if st.sidebar.button("PDF 처리 및 임베딩"):
            with st.spinner("PDF 텍스트 추출 및 임베딩 중..."):
                try:
					# 텍스트 추출 
                    full_text = extract_text_from_pdf(pdf_data) 
                    # 추출한 텍스트를 청킹 
                    st.session_state.chunks = chunk_text(full_text)
                    # print(st.session_state.chunks)
                    # 각 청크를 1536 차원 벡터로 변환 
                    embeddings = get_hcx_embeddings(hcx_api_key, st.session_state.chunks)
                    st.session_state.chunk_embeddings = embeddings
                    st.session_state.pdf_processed = True # PDF 텍스트 추출 완료 플래그 
                    st.success(f"PDF가 성공적으로 처리되었습니다! ({len(st.session_state.chunks)}개 청크)")
                except Exception as e:
                    st.error(f"임베딩 오류: {e}")

    # PDF 처리 완료 확인 후 질의 응답 
    if st.session_state.pdf_processed:
        st.subheader("PDF 내용에 대해 질문하세요")
        
        # 채팅 히스토리 표시
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        # 사용자 입력
        if prompt := st.chat_input("질문을 입력하세요"): # 실시간 입력 감지 
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            with st.chat_message("assistant"):
                with st.spinner("답변 생성 중..."):
                    try:
                        chunk_embeddings = st.session_state.chunk_embeddings
                        chunks = st.session_state.chunks

                        # 쿼리 임베딩
                        # 사용자가 입력한 질문에 대해 1536 벡터로 변환 
                        query_embedding = get_hcx_embeddings(hcx_api_key, [prompt])[0]
                        # print(query_embedding)

                        # top-k 청크 검색f
                        relevant_chunks = search_chunks(query_embedding, chunk_embeddings, chunks, top_k=0)

                        # 프롬프트 구성
                        context = "\n\n".join(relevant_chunks)
                        system_prompt = "당신은 PDF 문서 내용을 기반으로 질문에 답하는 AI입니다. 제공된 컨텍스트를 사용해 정확하고 간결하게 답변하세요. 컨텍스트에 없는 정보는 추측하지 마세요. 이전 대화 내용을 고려해 일관된 답변을 유지하세요."

                        # 3. 대화 히스토리 + 컨텍스트 + 현재 질문 구성
                        messages_for_hcx = [{"role": "system", "content": system_prompt}]
                        
                        # 이전 대화 기록 추가 (역순이 아닌 순서대로)
                        for msg in st.session_state.messages[:-1]:  # 현재 질문 제외
                            messages_for_hcx.append({
                                "role": "user" if msg["role"] == "user" else "assistant",
                                "content": msg["content"]
                            })
                        
                        # 현재 질문 + 컨텍스트 추가
                        messages_for_hcx.append({
                            "role": "user",
                            "content": f"컨텍스트:\n{context}\n\n질문: {prompt}"
                        })

                        # 4. CompletionExecutor 설정 (API 키는 Bearer 토큰 형식)
                        completion_executor = CompletionExecutor(
                            host='https://clovastudio.stream.ntruss.com',
                            api_key=f'Bearer {hcx_api_key}',
                            request_id='e15290bd91554d4a96a15416d5b50c84'
                        )

                        # 5. 요청 데이터 구성
                        request_data = {
                            "messages": messages_for_hcx,
                            "maxCompletionTokens": 1024,
                            "temperature": 0.3,
                            "topP": 0.9,
                            "repetitionPenalty": 1.0,
                            "includeAiFilters": True,
                            "seed": 42
                        }

                        # 6. 실행 및 결과 수신
                        response = completion_executor.execute(request_data)
                        # print(response)
                        
                        # 응답에서 content 추출
                        answer = response["result"]["message"]["content"]
                        
                        st.markdown(answer)
                        st.session_state.messages.append({"role": "assistant", "content": answer})
                    except Exception as e:
                        st.error(f"답변 오류: {e}")

if __name__ == "__main__":
    main()