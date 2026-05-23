from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
from sqlalchemy.orm import Session
from sqlalchemy.sql import func
from pydantic import BaseModel
from typing import Optional
from database import get_db
from models import User, Chat, Message
from auth import get_current_user
import httpx
import json
import os
import re
import asyncio
from datetime import date

router = APIRouter(tags=["chat"])

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
OLLAMA_URL = "http://localhost:11434/v1/chat/completions"
DEFAULT_MODEL = "qwen/qwen3-next-80b-a3b-instruct:free"

_FALLBACK_MODELS = [
    "qwen/qwen3-next-80b-a3b-instruct:free",
    "openai/gpt-oss-120b:free",
    "qwen/qwen3-coder:free",
    "meta-llama/llama-3.3-70b-instruct:free",
    "nousresearch/hermes-3-llama-3.1-405b:free",
    "google/gemma-3-27b-it:free",
    "mistralai/mistral-small-3.1-24b-instruct:free",
    "google/gemma-3-12b-it:free",
]


def _is_ollama_model(model: str) -> bool:
    return "/" not in model


# words to skip when scanning a message for stock tickers
_SKIP_WORDS = {
    "A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M",
    "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z",
    "AN", "AS", "AT", "BE", "BY", "DO", "GO", "IF", "IN", "IS", "IT",
    "ME", "MY", "NO", "OF", "OK", "ON", "OR", "SO", "TO", "UP", "US",
    "WE", "AI", "ML", "UI", "UX", "ID", "IP",
    "AND", "ARE", "BUT", "CAN", "FOR", "GET", "GOT", "HAD", "HAS", "HER",
    "HIM", "HIS", "HOW", "ITS", "LET", "MAY", "NOT", "NOW", "OUR", "OUT",
    "SAY", "SEE", "SET", "THE", "TOO", "TWO", "USE", "WAS", "WAY", "WHO",
    "WHY", "YOU", "YET", "NEW", "OLD", "ALL", "ANY", "OFF", "ONE", "OWN",
    "PER", "PUT", "RUN", "TRY", "OWE", "HIT", "BIG", "LOW", "HIGH", "TOP",
    # finance terms that look like tickers but aren't
    "CEO", "CFO", "COO", "CTO", "IPO", "ETF", "SEC", "GDP", "CPI", "FED",
    "EPS", "RSI", "ATH", "ATL", "MA", "SMA", "EMA", "MACD", "ADX", "ROI",
    "APR", "MAR", "JAN", "FEB", "JUN", "JUL", "AUG", "SEP", "OCT", "NOV",
    "DEC", "YTD", "MTD", "QTD", "TTM", "YOY", "MOM", "QOQ",
    "USD", "EUR", "GBP", "JPY", "CAD", "AUD", "BTC", "ETH",
    "PDF", "CSV", "URL", "API", "LLM", "NLP", "RAG",
    "INC", "LLC", "LTD", "CORP", "PLC", "NYSE", "NASDAQ",
    "BUY", "SELL", "HOLD", "LONG", "SHORT", "BULL", "BEAR",
    "DONT", "CANT", "WONT", "ISNT", "ARENT", "WASNT",
}


async def _stock_context_async(message: str) -> str:
    from routers.demo import (
        _fetch_yfinance, _rsi, _bollinger, _market_regime,
        _linear_forecast, _sentiment, COMPANY_NAMES,
    )

    # map common company names (case-insensitive) to tickers
    _NAME_MAP = {
        "google": "GOOGL", "alphabet": "GOOGL",
        "apple": "AAPL", "tesla": "TSLA", "nvidia": "NVDA",
        "microsoft": "MSFT", "meta": "META", "amazon": "AMZN",
        "netflix": "NFLX", "amd": "AMD", "palantir": "PLTR",
        "salesforce": "CRM", "uber": "UBER", "airbnb": "ABNB",
        "shopify": "SHOP", "snowflake": "SNOW", "coinbase": "COIN",
        "spy": "SPY", "sp500": "SPY",
    }
    candidates = set(re.findall(r'\b([A-Z]{2,5})\b', message))
    candidates -= _SKIP_WORDS
    msg_lower = message.lower()
    for name, ticker in _NAME_MAP.items():
        if name in msg_lower:
            candidates.add(ticker)
    if not candidates:
        return ""

    tickers = list(candidates)[:3]

    blocks = []
    for ticker in tickers:
        try:
            hist, news, _lp, _pc, _cn = await asyncio.to_thread(_fetch_yfinance, ticker)
        except Exception:
            continue

        closes = hist["Close"]
        current = float(closes.iloc[-1])
        prev = float(closes.iloc[-2])
        daily_pct = (current - prev) / prev * 100
        sign = "+" if daily_pct >= 0 else ""

        rsi_val = float(_rsi(closes).iloc[-1])
        ma7 = float(closes.tail(7).mean())
        ma30 = float(closes.tail(30).mean())
        _, upper_bb, lower_bb = _bollinger(closes)
        upper_val = float(upper_bb.iloc[-1])
        lower_val = float(lower_bb.iloc[-1])
        regime = _market_regime(closes)

        try:
            sentiment = await asyncio.to_thread(_sentiment, news)
            mood = sentiment["label"]
        except Exception:
            mood = "neutral"

        try:
            forecast = await asyncio.to_thread(_linear_forecast, closes)
            fc_str = f"{forecast['pct_change']:+.2f}%"
            fc_price = f"${closes.iloc[-1] * (1 + forecast['pct_change'] / 100):.2f}"
        except Exception:
            fc_str = "n/a"
            fc_price = "n/a"

        score = 0
        if rsi_val < 30: score += 2
        elif rsi_val > 70: score -= 2
        if current > ma7 > ma30: score += 1
        elif current < ma7 < ma30: score -= 1
        try:
            if forecast["pct_change"] > 2: score += 1
            elif forecast["pct_change"] < -2: score -= 1
        except Exception:
            pass
        signal = "Buy" if score >= 2 else "Sell" if score <= -2 else "Hold"

        company = COMPANY_NAMES.get(ticker, ticker)
        lines = [
            f"[LIVE MARKET DATA — {ticker} ({company}) — fetched from yfinance right now. Use these exact numbers; do NOT use training-data prices.]",
            f"  Price: ${current:.2f} ({sign}{daily_pct:.2f}% today)",
            f"  RSI(14): {rsi_val:.1f} · MA7: ${ma7:.2f} · MA30: ${ma30:.2f}",
            f"  Bollinger: upper ${upper_val:.2f} · lower ${lower_val:.2f}",
            f"  Regime: {regime} · News mood: {mood}",
            f"  Signal: {signal} · 7-day forecast: {fc_str} → {fc_price}",
        ]
        blocks.append("\n".join(lines))

    if not blocks:
        return ""

    header = (
        f"IMPORTANT: The following {len(blocks)} ticker(s) have live data injected below — "
        "cite these exact prices and metrics. Do NOT guess or use training knowledge for any price, RSI, or signal shown here."
    )
    return header + "\n\n" + "\n\n".join(blocks)


def _get_headers() -> dict:
    return {
        "Authorization": f"Bearer {os.getenv('OPENROUTER_API_KEY', '')}",
        "Content-Type": "application/json",
        "HTTP-Referer": "http://localhost:8000",
        "X-Title": "Chatbot",
    }

class ChatRequest(BaseModel):
    chat_id: int
    message: str
    image_data_url: Optional[str] = None
    context_ticker: Optional[str] = None
    panel_context: Optional[str] = None  # pre-formatted stock panel data from frontend


def _build_user_content(message: str, image_data_url: Optional[str]):
    if image_data_url:
        return [
            {"type": "text", "text": message},
            {"type": "image_url", "image_url": {"url": image_data_url}},
        ]
    return message


async def _stream_from_openrouter(model: str, messages: list, timeout: float = 60.0):
    chain = [model] + [m for m in _FALLBACK_MODELS if m != model]

    for try_model in chain:
        try:
            async with httpx.AsyncClient(timeout=timeout) as client:
                async with client.stream(
                    "POST", OPENROUTER_URL,
                    headers=_get_headers(),
                    json={
                        "model": try_model,
                        "stream": True,
                        "messages": messages,
                        "max_tokens": 800,
                        "temperature": 0.7,
                        "chat_template_kwargs": {"thinking": False},
                    },
                ) as response:
                    if response.status_code == 401:
                        raise RuntimeError("Invalid API key. Check your OPENROUTER_API_KEY in .env")
                    if response.status_code in (400, 404, 429):
                        continue  # rate limited or model unavailable, try next
                    if response.status_code != 200:
                        raise RuntimeError(f"OpenRouter error ({response.status_code}). Please try again.")

                    async for line in response.aiter_lines():
                        if not line or not line.startswith("data: "):
                            continue
                        payload = line[len("data: "):]
                        if payload.strip() == "[DONE]":
                            return
                        try:
                            data = json.loads(payload)
                            token = data["choices"][0]["delta"].get("content", "")
                            if token:
                                yield token
                        except (json.JSONDecodeError, KeyError, IndexError):
                            continue
                    return

        except RuntimeError:
            raise
        except httpx.ConnectError:
            raise RuntimeError("Could not reach OpenRouter. Check your internet connection.")
        except Exception as e:
            raise RuntimeError(f"Something went wrong: {str(e)}")

    raise RuntimeError(
        "All available models are currently rate limited. "
        "Please wait a minute and try again."
    )


async def _stream_from_ollama(model: str, messages: list, timeout: float = 120.0):
    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            async with client.stream(
                "POST",
                OLLAMA_URL,
                headers={"Content-Type": "application/json"},
                json={"model": model, "stream": True, "messages": messages},
            ) as response:
                if response.status_code != 200:
                    raise RuntimeError(
                        f"Ollama error ({response.status_code}). "
                        "Make sure Ollama is running: `ollama serve`"
                    )
                async for line in response.aiter_lines():
                    if not line or not line.startswith("data: "):
                        continue
                    payload = line[len("data: "):]
                    if payload.strip() == "[DONE]":
                        return
                    try:
                        data = json.loads(payload)
                        token = data["choices"][0]["delta"].get("content", "")
                        if token:
                            yield token
                    except (json.JSONDecodeError, KeyError, IndexError):
                        continue
    except RuntimeError:
        raise
    except httpx.ConnectError:
        raise RuntimeError(
            "Could not reach Ollama at localhost:11434. "
            "Start it with: `ollama serve`"
        )
    except Exception as e:
        raise RuntimeError(f"Something went wrong: {str(e)}")


@router.post("/chat")
async def chat(
    request: ChatRequest,
    db: Session = Depends(get_db),
    username: str = Depends(get_current_user),
):
    if not request.message.strip():
        raise HTTPException(status_code=400, detail="Message cannot be empty")
    if len(request.message) > 8000:
        raise HTTPException(status_code=400, detail="Message too long (max 8000 characters)")
    if request.image_data_url and len(request.image_data_url) > 6_000_000:
        raise HTTPException(status_code=400, detail="Image too large (max 4 MB)")
    if request.image_data_url and not request.image_data_url.startswith("data:image/"):
        raise HTTPException(status_code=400, detail="image_data_url must be a base64 data URL (data:image/...)")

    user = db.query(User).filter(User.username == username).first()
    if not user:
        raise HTTPException(status_code=401, detail="User not found")

    chat_obj = db.query(Chat).filter(Chat.id == request.chat_id, Chat.user_id == user.id).first()
    if not chat_obj:
        raise HTTPException(status_code=404, detail="Chat not found")

    model = chat_obj.model or DEFAULT_MODEL

    if not _is_ollama_model(model) and not os.getenv("OPENROUTER_API_KEY"):
        raise HTTPException(status_code=500, detail="OPENROUTER_API_KEY not set in .env")

    is_first = db.query(Message).filter(Message.chat_id == chat_obj.id, Message.role == "user").count() == 0
    if is_first and chat_obj.title == "New Chat" and request.message != "(image)":
        chat_obj.title = request.message[:40]
        db.commit()

    history = (
        db.query(Message)
        .filter(Message.chat_id == chat_obj.id)
        .order_by(Message.created_at)
        .all()
    )
    today = date.today().isoformat()
    date_note = f"Today's date is {today}."
    messages = []
    system_base = f"{date_note}\n{chat_obj.system_prompt}" if chat_obj.system_prompt else date_note
    messages.append({"role": "system", "content": system_base})
    messages += [{"role": m.role, "content": m.content} for m in history]

    if request.panel_context:
        live_ctx = (
            "LIVE STOCK PANEL DATA IS NOW LOADED — use these exact numbers in your reply. "
            "Do NOT ask the user to provide anything. Answer directly and concisely.\n\n"
            + request.panel_context
        )
        messages.append({"role": "system", "content": live_ctx})
    else:
        stock_ctx = await _stock_context_async(request.message)
        if not stock_ctx and request.context_ticker:
            stock_ctx = await _stock_context_async(request.context_ticker)
        if stock_ctx:
            messages.append({"role": "system", "content": stock_ctx})

    messages.append({"role": "user", "content": _build_user_content(request.message, request.image_data_url)})

    async def stream_response():
        full_response = ""
        user_msg_saved = False
        streamer = _stream_from_ollama if _is_ollama_model(model) else _stream_from_openrouter
        try:
            async for token in streamer(model, messages):
                if not user_msg_saved:
                    db.add(Message(chat_id=chat_obj.id, role="user", content=request.message))
                    db.commit()
                    user_msg_saved = True
                full_response += token
                yield token
        except RuntimeError as e:
            yield str(e)
            return

        if full_response:
            db.add(Message(chat_id=chat_obj.id, role="assistant", content=full_response))
            db.query(Chat).filter(Chat.id == chat_obj.id).update({"updated_at": func.now()})
            db.commit()

    return StreamingResponse(stream_response(), media_type="text/plain")


@router.post("/chats/{chat_id}/greeting")
async def greeting(
    chat_id: int,
    db: Session = Depends(get_db),
    username: str = Depends(get_current_user),
):
    user = db.query(User).filter(User.username == username).first()
    if not user:
        raise HTTPException(status_code=401, detail="User not found")

    chat_obj = db.query(Chat).filter(Chat.id == chat_id, Chat.user_id == user.id).first()
    if not chat_obj:
        raise HTTPException(status_code=404, detail="Chat not found")

    msg_count = db.query(Message).filter(Message.chat_id == chat_id).count()
    if msg_count > 0:
        raise HTTPException(status_code=400, detail="Chat already has messages")

    model = chat_obj.model or DEFAULT_MODEL
    messages = [
        {"role": "user", "content": "Greet the user briefly. One or two sentences max, warm but concise. No lists."}
    ]

    async def stream_greeting():
        full_response = ""
        streamer = _stream_from_ollama if _is_ollama_model(model) else _stream_from_openrouter
        try:
            async for token in streamer(model, messages, timeout=30.0):
                full_response += token
                yield token
        except RuntimeError:
            return  # greeting failing silently is fine

        if full_response:
            db.add(Message(chat_id=chat_obj.id, role="assistant", content=full_response))
            db.query(Chat).filter(Chat.id == chat_obj.id).update({"updated_at": func.now()})
            db.commit()

    return StreamingResponse(stream_greeting(), media_type="text/plain")
