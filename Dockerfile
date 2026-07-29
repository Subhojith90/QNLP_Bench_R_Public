FROM python:3.12.7-slim-bookworm@sha256:60d9996b6a8a3689d36db740b49f4327be3be09a21122bd02fb8895abb38b50d

LABEL org.opencontainers.image.title="QNLPBench-R Stage 8A portable verification environment" \
      org.opencontainers.image.description="Digest-pinned Linux CPU environment for the independent seed-11 replay" \
      org.opencontainers.image.base.name="python:3.12.7-slim-bookworm" \
      org.opencontainers.image.base.digest="sha256:60d9996b6a8a3689d36db740b49f4327be3be09a21122bd02fb8895abb38b50d"

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONHASHSEED=0 \
    MPLBACKEND=Agg \
    MPLCONFIGDIR=/tmp/matplotlib \
    OMP_NUM_THREADS=1 \
    OPENBLAS_NUM_THREADS=1 \
    MKL_NUM_THREADS=1 \
    NUMEXPR_NUM_THREADS=1 \
    VECLIB_MAXIMUM_THREADS=1

WORKDIR /work
COPY requirements-lock.txt pyproject.toml ./
RUN python -m pip install --no-cache-dir --disable-pip-version-check \
        -r requirements-lock.txt
COPY . .
RUN python -m pip install --no-cache-dir --disable-pip-version-check \
        --no-deps --no-build-isolation -e .

CMD ["bash", "scripts/run_stage8a_smoke.sh"]
