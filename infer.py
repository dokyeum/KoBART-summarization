import streamlit as st
import torch
from transformers import PreTrainedTokenizerFast
from transformers.models.bart import BartForConditionalGeneration

# 모델과 토크나이저 로드
model_path = r'C:\Users\kdoky\KoBART-summarization\kobart_summary'  # 저장된 KoBART 모델 경로
tokenizer = PreTrainedTokenizerFast.from_pretrained('gogamza/kobart-base-v1', clean_up_tokenization_spaces=False)
model = BartForConditionalGeneration.from_pretrained(model_path)

# 텍스트 정제 함수
def preprocess_text(text):
    """
    입력 텍스트를 정제하여 불필요한 기호와 중복 공백을 제거
    Args:
        text (str): 원본 텍스트
    Returns:
        str: 정제된 텍스트
    """
    text = " ".join(text.split())  # 중복 공백 제거
    for symbol in ["△", "●", "■", "…", "-", "※"]:
        text = text.replace(symbol, "")  # 불필요한 기호 제거
    return text

# 요약 후처리 함수
def postprocess_summary(summary):
    """
    생성된 요약문에서 중복 문장을 제거
    Args:
        summary (str): KoBART가 생성한 요약문
    Returns:
        str: 후처리된 요약문
    """
    sentences = list(dict.fromkeys(summary.split('.')))  # 중복 제거
    return ". ".join(sentences).strip()

# Streamlit 웹 앱 제목
st.title("KoBART 요약 생성기")

# 사용자로부터 입력 받기
text = st.text_area("요약할 텍스트를 입력하세요:", height=200)

# 버튼 클릭 시 추론 수행
if st.button("요약 생성하기"):
    if text:
        # 입력 텍스트를 정제
        preprocessed_text = preprocess_text(text)

        # 입력 텍스트를 토크나이즈
        input_ids = tokenizer.encode(preprocessed_text, return_tensors='pt')

        # 모델을 통해 요약 생성
        with torch.no_grad():
            output = model.generate(
                input_ids,
                max_length=120,      # 요약문 최대 길이
                min_length=50,       # 요약문 최소 길이
                num_beams=7,         # 빔 서치 수
                repetition_penalty=2.0,  # 반복 방지 가중치
                no_repeat_ngram_size=3,  # 반복 방지 (n-그램)
                length_penalty=1.5,  # 길이 제한 중요도
                do_sample=False,     # 빔 서치에 집중
                early_stopping=True  # 조기 종료
            )

        # 요약 결과 디코딩 및 후처리
        summary = tokenizer.decode(output[0], skip_special_tokens=True)
        refined_summary = postprocess_summary(summary)

        # 생성된 요약 출력
        st.subheader("생성된 요약:")
        st.write(refined_summary)
    else:
        st.warning("텍스트를 입력해 주세요.")

