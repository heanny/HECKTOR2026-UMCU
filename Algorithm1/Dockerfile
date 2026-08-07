FROM --platform=linux/amd64 pytorch/pytorch:2.9.1-cuda12.6-cudnn9-runtime AS hecktor2026-task

ENV PYTHONUNBUFFERED=1

RUN groupadd -r user && useradd -m --no-log-init -r -g user user

# Install gcc and build tools needed for pyradiomics C extension
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

USER user
WORKDIR /opt/app

COPY --chown=user:user requirements.txt /opt/app/
RUN python -m pip install \
    --user \
    --no-cache-dir \
    --no-color \
    --requirement /opt/app/requirements.txt

COPY --chown=user:user STUNetTrainer.py    /opt/app/
COPY --chown=user:user constants.py        /opt/app/
COPY --chown=user:user clinical_utils.py   /opt/app/
COPY --chown=user:user image_preprocess.py /opt/app/
COPY --chown=user:user models.py           /opt/app/
COPY --chown=user:user inference.py        /opt/app/

ENTRYPOINT ["python", "inference.py"]
