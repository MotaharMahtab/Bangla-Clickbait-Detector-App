import streamlit as st
from config import max_seq_length,device,model_name
import predictor
import os,wget
from preprocess import clean_title
from transformers import AutoTokenizer
import torch

@st.cache
def load_model():
  if not os.path.exists('clickbait_model.pt'):  
    site_url = 'https://clickbait-model.s3.amazonaws.com/clickbait_model.pt'
    file_name = wget.download(site_url)
  return predictor.PythonPredictor()


st.title('Bangla Clickbait Detector')
st.text('\n')
text = st.text_area('Input Article Title')
clicked = st.button('Detect')
st.text('\n')

if clicked:
  predict_clickbait = load_model()
  text = clean_title(text)
  tokenizer = AutoTokenizer.from_pretrained(model_name)
  input_ids = tokenizer.encode(text, add_special_tokens=True, max_length=max_seq_length, padding="max_length", truncation=True)
  input_mask_array = [int(token_id > 0) for token_id in input_ids]                          
  # Convertion to Tensor
  input_ids = torch.unsqueeze(torch.tensor(input_ids),0) 
  input_mask_array = torch.unsqueeze(torch.tensor(input_mask_array),0)
  label,probability = predict_clickbait.predict(input_ids,input_mask_array)
  st.subheader(f'Classified as {label}')
  st.text(f'Predicted with {probability*100:.2f}% confidence.')