FROM python:3.10.11-slim-buster

RUN apt update -y&&apt upgrade -y

RUN apt-get install libgl1-mesa-glx -y

RUN apt install libleptonica-dev libtesseract-dev tesseract-ocr-eng -y

COPY requirements.txt /

RUN python3 -m pip install -r /requirements.txt

RUN python3 -c "import cv2;import pytesseract"

COPY ./flexdaytestdata.png /

COPY ExtractTable.py /

WORKDIR /

ENTRYPOINT [ "python3", "ExtractTable.py"]

