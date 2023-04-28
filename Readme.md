# Table Data Extraction Module based on Python and Pytesseract

## Build:

For windows:

Install tesseract following instructions from [here](https://github.com/UB-Mannheim/tesseract/wiki).


For Linux 

```
sudo apt install libleptonica-dev libtesseract-dev tesseract-ocr-eng
```

## Python Dependencies
```
python3 -m pip install -r requirements.txt
```


## Usage:

```
python ExtractTable.py --help
```

## Docker

### Build Docker image
```
docker build -t tabledataextraction:latest .
```
### Run 

```
docker run -it tabledataextraction:latest --help
```

