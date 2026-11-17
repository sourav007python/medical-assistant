# medical-assistant
Collects user-reported symptoms to offer preliminary health insights or direct users to appropriate care levels.

## Run with Docker

Build the image:

```powershell
docker build -t medical-assistant .
```

Create a local `.env` file containing `OPENAI_API_KEY` and `PINECONE_API_KEY`, then start the container:

```powershell
docker run --rm -p 8080:8080 --env-file .env medical-assistant
```

Open `http://localhost:8080`. The `.env` file and local PDF data are excluded from the Docker build context; provide API keys as environment variables or through a secrets manager when deploying.
