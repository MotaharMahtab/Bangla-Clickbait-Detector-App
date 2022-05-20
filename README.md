# Bangla-Clickbait-Detector-App
Bangla Clickbait Detection app made with Streamlit using GAN-Transformer model. This app is not yet runnable as the main model.py file will
be uploaded when its corresponding thesis paper is published.

## Demo
  [![Alt text](https://github.com/MotaharMahtab/Bangla-Clickbait-Detector-App/blob/main/clickbait_detection_demo.gif)

[Streamlit website](https://www.streamlit.io/)

## Installation
It is recommended to use a virtual environment before installing the dependencies
```console
pip install -r requirements.txt
```
## Usage
Pretrained model is served as S3 bucket which will be made public when the corresponding thesis paperr is published
```console
import wget
wget.download(s3_bucket_link)
```
Run
```console
streamlit run main.py
```
