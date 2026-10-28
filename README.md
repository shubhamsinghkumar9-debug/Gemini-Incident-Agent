# Gemini Incident Agent (GA5 v2)

FastAPI incident agent designed for Render and the `ga5-incident-agent/v2` API contract.

## Deploy
1. Create a GitHub repository and upload `app.py`, `requirements.txt`, `render.yaml`, `.gitignore`, and this README.
2. In Render, choose **New → Blueprint** and select the repository.
3. Set `GEMINI_API_KEY` in the Render service environment. Keep it secret; never commit it.
4. Deploy. Verify `https://YOUR-SERVICE.onrender.com/health` returns JSON.
5. Submit the base URL without a trailing path to the grader.

`DATABASE_URL` is wired to a Render PostgreSQL database by `render.yaml`. If creating the database separately, set `DATABASE_URL` to its internal connection string.

## API
- `POST /v2/incidents`
- `POST /v2/incidents/{runId}/receipts`
- `GET /v2/incidents/{runId}`
- `GET /health`

## Environment
- `GEMINI_API_KEY` required
- `GEMINI_MODEL` optional (default `gemini-2.5-flash`)
- `DATABASE_URL` supplied by Render Blueprint

## Important
This is a starter implementation, not a guarantee of a passing hidden grader. Run the full grader after deployment. In particular, validate every incident-specific effect argument, retry/approval edge case, and OTLP correlation against the latest grader feedback before relying on a score. The service does not execute infrastructure tools itself; it returns dispatches for the grader's simulated tool system.
