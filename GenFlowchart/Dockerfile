# GPU + cuDNN base (runtime variant is lighter; switch to *devel* if you need nvcc)
FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04

# ---- Basics & OS deps ----
ENV DEBIAN_FRONTEND=noninteractive \
    TZ=UTC
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates curl wget git nano vim tini \
    build-essential pkg-config \
    ffmpeg libsm6 libxext6 libgl1 \
    openssh-client \
    && rm -rf /var/lib/apt/lists/*

# ---- Install Tesseract
RUN apt-get update && apt install -y tesseract-ocr libtesseract-dev

# ---- Create a non-root user (uid/gid 1000) ----
ARG USERNAME=developer
ARG USER_UID=1000
ARG USER_GID=1000
RUN groupadd --gid ${USER_GID} ${USERNAME} \
 && useradd -m -s /bin/bash --uid ${USER_UID} --gid ${USER_GID} ${USERNAME}

# ---- Micromamba (fast, minimal conda) ----
ENV MAMBA_ROOT_PREFIX=/opt/conda
RUN mkdir -p $MAMBA_ROOT_PREFIX && chown -R ${USERNAME}:${USERNAME} $MAMBA_ROOT_PREFIX

# Install micromamba (single static binary)
RUN wget -qO /tmp/micromamba.tar.bz2 \
      https://micro.mamba.pm/api/micromamba/linux-64/latest \
 && tar -xjf /tmp/micromamba.tar.bz2 -C /tmp \
 && install -m 0755 /tmp/bin/micromamba /usr/local/bin/micromamba \
 && rm -rf /tmp/micromamba.tar.bz2 /tmp/bin

USER ${USERNAME}
SHELL ["/bin/bash", "-lc"]

# ---- Create a base conda env with Jupyter ----
# Pin Python if you like: "python=3.10" or "python=3.11"
ENV MAMBA_DOCKERFILE_ACTIVATE=1
RUN micromamba create -y -n dl \
    python=3.10 \
    jupyterlab notebook ipykernel ipywidgets \
    numpy scipy pandas matplotlib \
    && micromamba clean -a -y
ENV PATH=$MAMBA_ROOT_PREFIX/envs/dl/bin:$PATH

# Register the kernel so it appears in Jupyter
RUN python -m ipykernel install --user --name "dl" --display-name "Python (dl)"

# ---- Workspace & permissions ----
USER root
RUN mkdir -p /workspace && chown -R ${USERNAME}:${USERNAME} /workspace
WORKDIR /workspace
USER ${USERNAME}

WORKDIR /home/zhengxin.chai/GenFlowchart

# Copy files
COPY GenFlowchart/requirements.txt .

# Install pypi packages
RUN pip install -r requirements.txt

# Install SAM
RUN mkdir weights
RUN cd weights && wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth

# Download some test data
RUN mkdir data
RUN cd data && wget https://media.roboflow.com/notebooks/examples/dog.jpeg && \
     wget https://media.roboflow.com/notebooks/examples/dog-2.jpeg && \
     wget https://media.roboflow.com/notebooks/examples/dog-3.jpeg && \
     wget https://media.roboflow.com/notebooks/examples/dog-4.jpeg

# Jupyter config: bind on all interfaces; token comes from env
EXPOSE 8888

# tini for proper signal handling; start JupyterLab
ENTRYPOINT ["/usr/bin/tini", "--"]
CMD ["bash", "-lc", "jupyter lab --ip=0.0.0.0 --port=8888 --no-browser"]

# pre-download BertScore default model
RUN python -c 'from bert_score import score; score(["hello world"], ["hello world"], lang="en")'

# pre-download word2vec pretrained model
RUN python -c 'import gensim.downloader as api; api.load("word2vec-google-news-300")'

# pre-download sentence-transformer models
RUN python -c 'from sentence_transformers import SentenceTransformer; SentenceTransformer("paraphrase-MiniLM-L6-v2"); SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")'
