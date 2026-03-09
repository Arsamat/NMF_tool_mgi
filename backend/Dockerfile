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
    dirmngr \
    gnupg \
    ca-certificates \
    wget \
    curl \
    software-properties-common \
    build-essential \
    gfortran \
    pkg-config \
    libcairo2-dev \
    libpng-dev \
    libcurl4-openssl-dev \
    libssl-dev \
    libxml2-dev \
    libfontconfig1-dev \
    libharfbuzz-dev \
    libfribidi-dev \
    libfreetype6-dev \
    libtiff5-dev \
    libjpeg-dev \
    libicu-dev \
    zlib1g-dev \
    libbz2-dev \
    liblzma-dev \
 && ln -snf /usr/share/zoneinfo/$TZ /etc/localtime \
 && echo $TZ > /etc/timezone \
 && dpkg-reconfigure -f noninteractive tzdata \
 && wget -qO- https://cloud.r-project.org/bin/linux/ubuntu/marutter_pubkey.asc \
    | tee /etc/apt/trusted.gpg.d/cran_ubuntu_key.asc > /dev/null \
 && . /etc/os-release \
 && echo "deb https://cloud.r-project.org/bin/linux/ubuntu ${VERSION_CODENAME}-cran40/" \
    > /etc/apt/sources.list.d/cran-r.list \
 && apt-get update \
 && apt-get install -y --no-install-recommends \
    r-base \
    r-base-dev \
    littler \
 && rm -rf /var/lib/apt/lists/*

# --- Install R packages needed by your script ---
RUN R -q -e "install.packages(c('BiocManager','msigdbr'), repos='https://cloud.r-project.org')" \
&& R -q -e "BiocManager::install(version='3.22', ask=FALSE)" \
&& R -q -e "options(repos = BiocManager::repositories()); BiocManager::install(c('edgeR','limma','biomaRt','AnnotationDbi','org.Hs.eg.db','org.Mm.eg.db','ggtree','enrichplot','clusterProfiler'), ask=FALSE, update=FALSE)" \
&& R -q -e "stopifnot(requireNamespace('edgeR', quietly=TRUE)); stopifnot(requireNamespace('limma', quietly=TRUE)); stopifnot(requireNamespace('biomaRt', quietly=TRUE)); stopifnot(requireNamespace('AnnotationDbi', quietly=TRUE)); stopifnot(requireNamespace('org.Hs.eg.db', quietly=TRUE)); stopifnot(requireNamespace('org.Mm.eg.db', quietly=TRUE)); stopifnot(requireNamespace('msigdbr', quietly=TRUE)); stopifnot(requireNamespace('ggtree', quietly=TRUE)); stopifnot(requireNamespace('enrichplot', quietly=TRUE)); stopifnot(requireNamespace('clusterProfiler', quietly=TRUE)); cat(R.version.string, '\n'); cat('Bioc ', as.character(BiocManager::version()), '\n', sep='')"

COPY requirements.txt .
RUN pip install --default-timeout=1000 --retries 5 -r requirements.txt

COPY packages/cnmf /opt/mypkg
RUN echo "=== /opt/mypkg ===" && ls -la /opt/mypkg && \
    echo "=== /opt/mypkg/src ===" && ls -la /opt/mypkg/src || true && \
    echo "=== /opt/mypkg/src/cnmf ===" && ls -la /opt/mypkg/src/cnmf || true && \
    echo "=== find version.py ===" && find /opt/mypkg -maxdepth 5 -name version.py -print || true
RUN pip install --no-cache-dir -e /opt/mypkg

# Copy the rest of your backend code (fast_api.py, utils/, etc.)
COPY . .

# Make R script executable
RUN chmod +x deg_analysis.R

EXPOSE 8000

# Start FastAPI
CMD ["uvicorn", "fast_api:app", "--host", "0.0.0.0", "--port", "8000"]
