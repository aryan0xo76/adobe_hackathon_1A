# Challenge 1A – PDF Outline Extraction (Adobe India Hackathon 2025)

**Important Note for Running Dockerfile:**

The `model` folder is too large for GitHub.  
Please download it from: https://drive.google.com/drive/folders/1RdchzRG1YmGgSt3jPGpHlz1RzuE95s0m?usp=drive_link  
Place it in the same directory as the `Dockerfile` before building/running Docker.

## Dependencies

All Python dependencies are listed in `requirements.txt`. Key packages include:

- pdfplumber
- numpy
- pandas
- gensim
- xgboost
- tqdm
- rapidfuzz
- scikit-learn
- fuzzywuzzy==0.18.0
- python-Levenshtein==0.20.9

No external downloads or internet connection are required at runtime.

## How to Run

### 1. Build the Docker Image

From the project root directory:

docker build -t adobe-challenge1a:latest .


### 2. Prepare Input and Output Folders

- Place your input PDFs into `dataset/input/`
- Ensure `dataset/output/` exists and is empty (outputs will be saved here)

### 3. Run the Solution

**On Linux/macOS/Git Bash:**
docker run --rm
-v "$(pwd)/dataset/input:/app/dataset/input"
-v "$(pwd)/dataset/output:/app/dataset/output"
adobe-challenge1a:latest


**On Windows PowerShell:**
docker run --rm -v "${PWD}\dataset\input:/app/dataset/input"
-v "${PWD}\dataset\output:/app/dataset/output" `
adobe-challenge1a:latest


### 4. Output

A `.json` outline file will be created in `dataset/output/` for each PDF in `dataset/input/`.

