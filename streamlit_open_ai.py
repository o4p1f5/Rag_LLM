import streamlit as st
import fitz  # pymupdf
from openai import OpenAI
import numpy as np
import re
from dotenv import load_dotenv
import os

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
def get_embeddings(client, texts, model="text-embedding-ada-002"):
    response = client.embeddings.create(input=texts, model=model)
    return np.array([item.embedding for item in response.data])

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
        openai_api_key = st.text_input("OpenAI API Key 설정", type="password", value=os.getenv("OPENAI_API_KEY", ""))
        st.write("[OpenAI API Key 받기](https://platform.openai.com/account/api-keys)")
        pdf_file = st.file_uploader("PDF 파일을 업로드하세요.", type=["pdf"])
        
        # 세션 상태 초기화 (메모리 유지)
        if "chunks" not in st.session_state:
            st.session_state.chunks = []
        if "chunk_embeddings" not in st.session_state:
            st.session_state.chunk_embeddings = None
        if "client" not in st.session_state:
            st.session_state.client = None
        if "messages" not in st.session_state:
            st.session_state.messages = []
        if "pdf_processed" not in st.session_state:
            st.session_state.pdf_processed = False

    if openai_api_key and not st.session_state.client:
        st.session_state.client = OpenAI(api_key=openai_api_key)

    if pdf_file and not st.session_state.pdf_processed:
        pdf_data = pdf_file.read()
        if st.sidebar.button("PDF 처리 및 임베딩"):
            with st.spinner("PDF 텍스트 추출 및 임베딩 중..."):
                try:
					# 텍스트 추출 
                    full_text = extract_text_from_pdf(pdf_data) 
                    # 추출한 텍스트를 청킹 
                    st.session_state.chunks = chunk_text(full_text)
                    # 각 청크를 1536 차원 벡터로 변환 
                    embeddings = get_embeddings(st.session_state.client, st.session_state.chunks)
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
                        client = st.session_state.client
                        chunk_embeddings = st.session_state.chunk_embeddings
                        chunks = st.session_state.chunks

                        # 쿼리 임베딩
                        # 사용자가 입력한 질문에 대해 1536 벡터로 변환 
                        query_embedding = get_embeddings(client, [prompt])[0]

                        # top-k 청크 검색
                        relevant_chunks = search_chunks(query_embedding, chunk_embeddings, chunks, top_k=0)

                        # 프롬프트 구성
                        context = "\n\n".join(relevant_chunks)
                        system_prompt = "당신은 PDF 문서 내용을 기반으로 질문에 답하는 AI입니다. 제공된 컨텍스트를 사용해 정확하고 간결하게 답변하세요. 컨텍스트에 없는 정보는 추측하지 마세요. 이전 대화 내용을 고려해 일관된 답변을 유지하세요."

                        # 이전 메시지 포함
                        messages = [
                            {"role": "system", "content": system_prompt},
                        ] + st.session_state.messages[:-1] + [  # 이전 히스토리 + 현재 쿼리
                            {"role": "user", "content": f"컨텍스트:\n{context}\n\n질문: {prompt}"}
                        ]

                        response = client.chat.completions.create(
                            model="gpt-4o-mini",
                            messages=messages,
                        )
                        answer = response.choices[0].message.content
                        st.markdown(answer)
                        st.session_state.messages.append({"role": "assistant", "content": answer})
                    except Exception as e:
                        st.error(f"답변 오류: {e}")

if __name__ == "__main__":
    main()