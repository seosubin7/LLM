import streamlit as st
from transformers import AutoTokenizer, AutoModelForCausalLM

st.set_page_config(page_title="🦜🔗 뭐든지 질문하세요~ ")
st.title('🦜🔗 뭐든지 질문하세요~ ')

def generate_response(input_text):
    BASE_MODEL = "google/gemma-2b-it"

    model = AutoModelForCausalLM.from_pretrained(BASE_MODEL, device_map="auto")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, add_special_tokens=True)


    input_ids = tokenizer(input_text, return_tensors="pt")["input_ids"]

    outputs = model.generate(input_ids.to("cuda"), max_length=100, num_return_sequences=1, temperature=0.1)

    decoded_output = tokenizer.decode(outputs[0], skip_special_tokens=True)
    st.info(decoded_output)

with st.form('Question'):
    text = st.text_area('질문 입력:', 'What types of text models does Google\'s Gemma provide?')
    submitted = st.form_submit_button('보내기')
    if submitted:
        generate_response(text)
