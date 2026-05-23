# Dependencies

Install all dependencies with:

```bash
pip install -r requirements.txt
```

| Package | Purpose |
|---|---|
| `fastapi` | Web framework |
| `uvicorn[standard]` | ASGI server to run the app |
| `httpx` | Async HTTP client for LLM API calls |
| `python-dotenv` | Loads environment variables from `.env` |
| `sqlalchemy` | ORM and SQLite database integration |
| `bcrypt` / `passlib[bcrypt]` | Password hashing |
| `python-jose[cryptography]` | JWT token generation and validation |
| `pytest` / `pytest-asyncio` | Test framework |
| `pdfplumber` | PDF text extraction |
| `python-multipart` | File upload support |
| `yfinance` | Real-time and historical stock data |
| `numpy` | Numerical computations |
| `scikit-learn` | Data preprocessing for the Cox model |
| `vaderSentiment` | News headline sentiment scoring |
| `lifelines` | Cox proportional hazard survival model |

## Environment Variables

Copy `.env.example` to `.env` and fill in:

```
OPENROUTER_API_KEY=your_api_key_here   # from openrouter.ai (free tier available)
SECRET_KEY=your_secret_key_here        # any long random string for JWT signing
```

## Running the App

```bash
uvicorn main:app --reload
```

Then open `http://localhost:8000`.

## Training the Cox Model

Run once before starting the server:

```bash
python train_cox.py
```
