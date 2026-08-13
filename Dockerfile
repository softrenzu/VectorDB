FROM python:3.12-slim
WORKDIR /app
COPY pyproject.toml README.md LICENSE ./
COPY src ./src
RUN pip install --no-cache-dir .
ENV ROOOMTECH_VECTOR_DB_PATH=/data/rooomtech_vector.db
EXPOSE 8080
VOLUME ["/data"]
CMD ["rooomtech-vector"]
