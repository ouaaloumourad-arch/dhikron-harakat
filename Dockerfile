FROM runpod/base:0.4.0-cuda11.8.0

SHELL ["/bin/bash", "-c"]
WORKDIR /

RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

# numpy<2 obligatoire pour PyTorch 2.1
RUN python3 -m pip install --no-cache-dir --upgrade pip && \
    python3 -m pip install --no-cache-dir "numpy<2" && \
    python3 -m pip install --no-cache-dir \
    torch==2.1.2 torchaudio==2.1.2 --index-url https://download.pytorch.org/whl/cu118 && \
    python3 -m pip install --no-cache-dir \
    transformers==4.38.2 \
    accelerate==0.27.2 \
    soundfile==0.12.1 \
    runpod==1.6.0

# Pré-téléchargement du modèle avec Harakât
RUN python3 -c "\
from transformers import AutoProcessor, AutoModelForCTC; \
model_name = 'rabah2026/wav2vec2-large-xlsr-53-arabic-quran-v_final'; \
AutoProcessor.from_pretrained(model_name); \
AutoModelForCTC.from_pretrained(model_name); \
print('Model cached successfully')\
"

ADD handler.py /handler.py

CMD ["python3", "-u", "/handler.py"]
