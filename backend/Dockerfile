# GPU-capable base (works on CPU too)
FROM pytorch/pytorch:2.2.2-cuda12.1-cudnn8-runtime

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app

WORKDIR /app

# Install deps first (better caching)
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Etc/UTC

RUN apt-get update && apt-get install -y --no-install-recommends \
    tzdata \
    r-base \
    r-base-dev \
    littler \
    libcurl4-openssl-dev \
    libssl-dev \
    libxml2-dev \
 && ln -snf /usr/share/zoneinfo/$TZ /etc/localtime \
 && echo $TZ > /etc/timezone \
 && dpkg-reconfigure -f noninteractive tzdata \
 && rm -rf /var/lib/apt/lists/*

# --- Install R packages needed by your script ---
RUN R -q -e "install.packages(c('BiocManager'), repos='https://cloud.r-project.org')" \
 && R -q -e "BiocManager::install(c('edgeR','limma','biomaRt'), ask=FALSE, update=FALSE)"

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY cnmf_package /opt/mypkg
RUN echo "=== /opt/mypkg ===" && ls -la /opt/mypkg && \
    echo "=== /opt/mypkg/src ===" && ls -la /opt/mypkg/src || true && \
    echo "=== /opt/mypkg/src/cnmf ===" && ls -la /opt/mypkg/src/cnmf || true && \
    echo "=== find version.py ===" && find /opt/mypkg -maxdepth 5 -name version.py -print || true
RUN pip install --no-cache-dir -e /opt/mypkg

# Copy the rest of your backend code (fast_api.py, utils/, etc.)
COPY . .

EXPOSE 8000

# Start FastAPI
CMD ["uvicorn", "fast_api:app", "--host", "0.0.0.0", "--port", "8000"]
