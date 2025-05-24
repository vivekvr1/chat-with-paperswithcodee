FROM python:3.11-slim as builder

ARG user=app_user
ARG group=${user}
ARG uid=1010
ARG gid=1010

ARG APP_DIR=/app



RUN apt-get -y -q update && \
    apt-get -y -q install --no-install-recommends curl &&  | python - --version 1.3.2



WORKDIR $APP_DIR
COPY requirements.txt ./
RUN find . | grep -E "(__pycache__|\.pyc$)" | xargs rm -rf
RUN pip install -r requiremnts.txt

FROM python:3.11-slim as runtime

ENV VIRTUAL_ENV=/app/.venv \
    PATH="/app/.venv/bin:$PATH" \
    PROJECT_ID="playground-351113"

COPY --from=builder ${VIRTUAL_ENV} ${VIRTUAL_ENV}

COPY src src

ENV PORT 80

ENTRYPOINT [ "python", "-m", "streamlit", "run", "src/app.py", "--server.port=80", "--server.address=0.0.0.0", "--theme.primaryColor=#135aaf"]