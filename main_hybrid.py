import os
import uuid
import json
import random
import logging
import asyncio
import re
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from contextlib import asynccontextmanager
from datetime import datetime, timedelta
from typing import List, Optional, Tuple, Dict

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Depends, status, Body, Request, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse, JSONResponse, Response
from pydantic import BaseModel, EmailStr
from bson import ObjectId

import google.generativeai as genai
import motor.motor_asyncio
from passlib.context import CryptContext
from jose import JWTError, jwt
import redis.asyncio as aioredis
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
from authlib.integrations.starlette_client import OAuth
from starlette.config import Config
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

auth_scheme = HTTPBearer(auto_error=False)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
load_dotenv()

FRONTEND_ORIGIN = os.getenv("FRONTEND_ORIGIN", "http://localhost:5174")
API_BASE_URL    = os.getenv("API_BASE_URL",    "http://localhost:8000")
MONGO_URI       = os.getenv("MONGODB_URI")
REDIS_HOST      = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT      = int(os.getenv("REDIS_PORT", "6379"))

SMTP_HOST = os.getenv("SMTP_HOST", "smtp.gmail.com")
SMTP_PORT = int(os.getenv("SMTP_PORT", "587"))
SMTP_USER = os.getenv("SMTP_USER", "")
SMTP_PASS = os.getenv("SMTP_PASS", "")
SMTP_FROM = os.getenv("SMTP_FROM", SMTP_USER)

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
SECRET_KEY                  = os.getenv("JWT_SECRET_KEY", "your-secret-key-for-jwt")
ALGORITHM                   = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", "60"))
REFRESH_TOKEN_EXPIRE_DAYS   = int(os.getenv("REFRESH_TOKEN_EXPIRE_DAYS",   "14"))
OTP_EXPIRE_MINUTES          = int(os.getenv("OTP_EXPIRE_MINUTES",          "10"))

# ── SLA config (hours) ────────────────────────────────────────────────────────
SLA_HOURS = {"critical": 4, "high": 20, "medium": 48, "low": 72}

# ── Priority keyword map ──────────────────────────────────────────────────────
PRIORITY_KEYWORDS = {
    "critical": [
        "hacked", "fraud", "stolen", "breach", "ransomware", "identity theft",
        "data leak", "unauthorized access", "account compromised", "emergency",
        "medical emergency", "police", "stranded", "illegal",
    ],
    "high": [
        "urgent", "asap", "immediately", "cannot access", "locked out",
        "system down", "outage", "not working", "broken", "crash",
        "billing error", "unauthorized charge", "refund", "visa rejected",
        "missed flight", "stuck at airport",
    ],
    "medium": [
        "slow", "issue", "problem", "incorrect", "wrong", "help",
        "confused", "not sure", "question", "inquiry", "change",
    ],
    "low": [],          # default fallback
}

def detect_priority(text: str) -> str:
    t = text.lower()
    for priority in ("critical", "high", "medium"):
        if any(kw in t for kw in PRIORITY_KEYWORDS[priority]):
            return priority
    return "low"

def sla_deadline(priority: str) -> datetime:
    hours = SLA_HOURS.get(priority, 72)
    return datetime.utcnow() + timedelta(hours=hours)

# ─────────────────────────────────────────────────────────────────────────────
# Password / JWT
# ─────────────────────────────────────────────────────────────────────────────
def verify_password(plain: str, hashed: Optional[str]) -> bool:
    return bool(hashed) and pwd_context.verify(plain, hashed)

def get_password_hash(pw: str) -> str:
    return pwd_context.hash(pw)

def create_access_token(data: dict, expires_delta: Optional[timedelta] = None) -> str:
    to_encode = data.copy()
    expire = datetime.utcnow() + (expires_delta or timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES))
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)

def create_refresh_token(data: dict, expires_delta: Optional[timedelta] = None) -> str:
    to_encode = data.copy()
    expire = datetime.utcnow() + (expires_delta or timedelta(days=REFRESH_TOKEN_EXPIRE_DAYS))
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)

# ─────────────────────────────────────────────────────────────────────────────
# Email helpers
# ─────────────────────────────────────────────────────────────────────────────
def _smtp_send(to: str, msg: MIMEMultipart):
    with smtplib.SMTP(SMTP_HOST, SMTP_PORT) as s:
        s.starttls()
        s.login(SMTP_USER, SMTP_PASS)
        s.sendmail(SMTP_FROM, to, msg.as_string())

async def _send_email_async(to: str, msg: MIMEMultipart):
    if not SMTP_USER or not SMTP_PASS:
        logging.warning("SMTP not configured — email skipped")
        return
    try:
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, _smtp_send, to, msg)
        logging.info(f"Email sent → {to}")
    except Exception as e:
        logging.error(f"Email error: {e}")

async def send_otp_email(to_email: str, name: str, otp: str):
    msg = MIMEMultipart("alternative")
    msg["Subject"] = "🔐 EchoMind — Verify your email"
    msg["From"]    = SMTP_FROM
    msg["To"]      = to_email
    html = f"""
    <html><body style="font-family:Arial,sans-serif;background:#0f0f17;color:#e0e0e0;padding:32px">
    <div style="max-width:520px;margin:auto;background:#1a1a2e;border-radius:16px;padding:32px;
                border:1px solid rgba(139,92,246,0.35)">
      <h2 style="color:#8B5CF6;margin-bottom:4px">Verify your email</h2>
      <p>Hi <b>{name}</b>, use the OTP below to complete registration.
         It expires in <b>{OTP_EXPIRE_MINUTES} minutes</b>.</p>
      <div style="text-align:center;margin:28px 0">
        <span style="font-size:42px;font-weight:700;letter-spacing:10px;color:#10B981">{otp}</span>
      </div>
      <p style="color:#888;font-size:13px">If you did not request this, ignore the email.</p>
    </div></body></html>"""
    msg.attach(MIMEText(html, "html"))
    await _send_email_async(to_email, msg)

async def send_approval_email(to_email: str, agent_name: str, approved: bool):
    status_word = "approved ✅" if approved else "rejected ❌"
    msg = MIMEMultipart("alternative")
    msg["Subject"] = f"EchoMind — Your agent account has been {status_word}"
    msg["From"]    = SMTP_FROM
    msg["To"]      = to_email
    html = f"""
    <html><body style="font-family:Arial,sans-serif;background:#0f0f17;color:#e0e0e0;padding:32px">
    <div style="max-width:520px;margin:auto;background:#1a1a2e;border-radius:16px;padding:32px;
                border:1px solid rgba(139,92,246,0.35)">
      <h2 style="color:{'#10B981' if approved else '#ef4444'}">Account {status_word}</h2>
      <p>Hi <b>{agent_name}</b>,</p>
      {'<p>Your agent account has been <b>approved</b>. You can now log in and start handling tickets.</p>'
       if approved else
       '<p>Unfortunately your agent account request was <b>rejected</b> by the admin. '
       'Please contact support if you believe this is a mistake.</p>'}
    </div></body></html>"""
    msg.attach(MIMEText(html, "html"))
    await _send_email_async(to_email, msg)

async def send_resolution_email(to_email: str, user_name: str, ticket_subject: str,
                                ticket_id: str, resolution_note: str = ""):
    msg = MIMEMultipart("alternative")
    msg["Subject"] = f"✅ Ticket Resolved — #{ticket_id[:8]}"
    msg["From"]    = SMTP_FROM
    msg["To"]      = to_email
    html = f"""
    <html><body style="font-family:Arial,sans-serif;background:#0f0f17;color:#e0e0e0;padding:32px">
    <div style="max-width:520px;margin:auto;background:#1a1a2e;border-radius:16px;padding:32px;
                border:1px solid rgba(139,92,246,0.35)">
      <h2 style="color:#10B981">Your ticket has been resolved!</h2>
      <p>Hi <b>{user_name}</b>,</p>
      <p>Ticket: <b>{ticket_subject}</b> (#{ticket_id[:8]}) is now resolved.</p>
      {f'<p><b>Note:</b> {resolution_note}</p>' if resolution_note else ''}
      <p style="color:#888;font-size:13px">Start a new chat if you need further help.</p>
    </div></body></html>"""
    msg.attach(MIMEText(html, "html"))
    await _send_email_async(to_email, msg)

async def send_ticket_assigned_email(to_email: str, agent_name: str, ticket_subject: str,
                                     ticket_id: str, priority: str, deadline: datetime):
    msg = MIMEMultipart("alternative")
    msg["Subject"] = f"🎫 New Ticket Assigned — #{ticket_id[:8]}"
    msg["From"]    = SMTP_FROM
    msg["To"]      = to_email
    html = f"""
    <html><body style="font-family:Arial,sans-serif;background:#0f0f17;color:#e0e0e0;padding:32px">
    <div style="max-width:520px;margin:auto;background:#1a1a2e;border-radius:16px;padding:32px;
                border:1px solid rgba(139,92,246,0.35)">
      <h2 style="color:#8B5CF6">New Ticket Assigned</h2>
      <p>Hi <b>{agent_name}</b>, a new ticket has been assigned to you.</p>
      <p><b>Subject:</b> {ticket_subject}</p>
      <p><b>Priority:</b> <span style="color:{'#ef4444' if priority=='critical' else '#f59e0b' if priority=='high' else '#10B981'}">{priority.upper()}</span></p>
      <p><b>SLA Deadline:</b> {deadline.strftime('%Y-%m-%d %H:%M UTC')}</p>
      <p style="color:#888;font-size:13px">Please log in to the agent dashboard to handle this ticket.</p>
    </div></body></html>"""
    msg.attach(MIMEText(html, "html"))
    await _send_email_async(to_email, msg)

# ─────────────────────────────────────────────────────────────────────────────
# Gemini
# ─────────────────────────────────────────────────────────────────────────────
gemini_api_key = os.getenv("GEMINI_API_KEY")
if gemini_api_key:
    try:
        genai.configure(api_key=gemini_api_key)
        logging.info("✅ Gemini configured")
    except Exception as e:
        logging.warning(f"Gemini configure failed: {e}")
else:
    logging.warning("GEMINI_API_KEY missing — Gemini disabled.")

GEMINI_MODEL_NAME = os.getenv("GEMINI_MODEL_NAME", "gemini-2.0-flash")

async def _call_gemini(prompt_text: str, *, temperature: float = 0.7, max_output_tokens: int = 512) -> str:
    if not gemini_api_key:
        return "Sorry, AI is temporarily unavailable."
    try:
        loop = asyncio.get_running_loop()
        def sync_call():
            model = genai.GenerativeModel(GEMINI_MODEL_NAME)
            return model.generate_content(prompt_text)
        resp = await loop.run_in_executor(None, sync_call)
        if hasattr(resp, "text") and resp.text:
            return resp.text
        return ""
    except Exception as e:
        logging.error(f"Gemini error: {e}")
        return "Sorry, I am unable to process your request now."

# ─────────────────────────────────────────────────────────────────────────────
# Transformers
# ─────────────────────────────────────────────────────────────────────────────
sentiment_analyzer    = None
zero_shot_classifier  = None
INTENT_LABELS         = ["general", "technical", "finance", "travel"]
ZERO_SHOT_THRESHOLD   = float(os.getenv("ZERO_SHOT_THRESHOLD", "0.55"))

try:
    s_tok     = AutoTokenizer.from_pretrained("distilbert-base-uncased-finetuned-sst-2-english")
    s_model   = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased-finetuned-sst-2-english")
    sentiment_analyzer = pipeline("sentiment-analysis", model=s_model, tokenizer=s_tok)
    logging.info("✅ Sentiment model ready.")
except Exception as e:
    logging.warning(f"Sentiment model unavailable: {e}")

try:
    zero_shot_classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")
    logging.info("✅ Zero-shot classifier ready.")
except Exception as e:
    logging.warning(f"Zero-shot classifier unavailable: {e}")

def analyze_sentiment(text: str) -> str:
    if sentiment_analyzer is not None:
        try:
            return sentiment_analyzer(text[:512])[0]["label"]
        except Exception:
            pass
    neg = ["angry","frustrated","terrible","horrible","worst","awful","hate","broken","failed"]
    return "NEGATIVE" if any(w in text.lower() for w in neg) else "POSITIVE"

def _keyword_intent(text: str) -> Tuple[str, float, str]:
    t = text.lower()
    if any(w in t for w in ["payment","billing","invoice","refund","charge","money","bank","transaction"]):
        return "finance", 0.8, "keyword"
    if any(w in t for w in ["bug","error","crash","install","technical","code","api","server","network"]):
        return "technical", 0.8, "keyword"
    if any(w in t for w in ["flight","hotel","booking","travel","trip","visa","passport","ticket"]):
        return "travel", 0.8, "keyword"
    return "general", 0.6, "keyword"

async def classify_intent(text: str) -> Tuple[str, float, str]:
    if zero_shot_classifier is not None:
        try:
            loop   = asyncio.get_running_loop()
            result = await loop.run_in_executor(None, lambda: zero_shot_classifier(text[:512], INTENT_LABELS))
            label, score = result["labels"][0], result["scores"][0]
            if score >= ZERO_SHOT_THRESHOLD:
                return label, score, "zero_shot"
        except Exception as e:
            logging.warning(f"Zero-shot failed: {e}")
    return _keyword_intent(text)

# ─────────────────────────────────────────────────────────────────────────────
# KB / history helpers
# ─────────────────────────────────────────────────────────────────────────────
async def get_history_answer(user_query: str) -> Optional[str]:
    r = getattr(app.state, "redis", None)
    if r is None:
        return None
    try:
        return await r.get(f"qa:{user_query.lower().strip()}")
    except Exception:
        return None

async def get_kb_answer(user_query: str, domain: str) -> Optional[str]:
    col = getattr(app.state, "faq_kb_collection", None)
    if col is None:
        return None
    try:
        words = [w for w in user_query.lower().split() if len(w) > 3]
        if not words:
            return None
        docs = await col.find({"domain": domain, "keywords": {"$in": words}}).limit(5).to_list(5)
        return docs[0].get("answer") if docs else None
    except Exception as e:
        logging.error(f"KB error: {e}")
        return None

async def get_case_resolution_context(customer_id: str, domain: str, user_query: str) -> str:
    col = getattr(app.state, "cases_collection", None)
    if col is None:
        return ""
    try:
        recent = await col.find({"customer_id": customer_id, "status": "resolved"}).sort("last_updated",-1).limit(3).to_list(3)
        summaries = [c.get("summary","") for c in recent if c.get("summary")]
        return " | ".join(summaries[:2])
    except Exception:
        return ""

async def generate_bot_response(user_query: str, conversation_history, domain: str, customer_id: str) -> str:
    ctx          = await get_case_resolution_context(customer_id, domain, user_query)
    history_text = "\n".join([f"{m.role}: {m.content}" for m in (conversation_history or [])[-6:]])
    prompt = (
        f"You are EchoMind, an AI customer support assistant for {domain}.\n"
        f"Past resolution context: {ctx}\n"
        f"Recent conversation:\n{history_text}\n"
        f"Customer: {user_query}\n"
        "Reply helpfully and concisely. Analyze the user's request carefully against these domain-specific escalation rules:\n"
        "- General Support: Escalate if the request involves permanent data deletion, account cancellation, legal threats, account bans/appeals, harassment/abuse reports, or physical safety issues (e.g., 'lawsuit', 'attorney', 'GDPR', 'formal complaint', 'threat', 'police', 'abuse', 'ban appeal').\n"
        "- Technical Support: Escalate if the user provides a specific error code that you cannot resolve, reports a security breach ('hacked', 'unauthorized access'), catastrophic data loss ('lost all my data'), hardware replacement/RMA, or a critical enterprise production outage.\n"
        "- Finance & Billing: Escalate any mention of 'Fraud' immediately as a P1 ticket. Escalate financial issues needing human verification (e.g., double charged, dispute, chargeback, refunds), as well as tax document requests, collections, or wire transfers.\n"
        "- Travel: Escalate if the request is highly time-sensitive (traveling 'today' or 'within 24 hours') AND mentions disruptions ('missed flight', 'cancelled', 'stranded'). Also escalate for medical emergencies abroad, natural disasters ('hurricane', 'earthquake'), bereavement, or special disability accommodations.\n"
        "If the user's request matches ANY of these escalation triggers, or if you genuinely cannot resolve the issue, you MUST include the exact text '[ESCALATE_TO_HUMAN]' anywhere in your response. Otherwise, try to answer the question normally without creating a ticket."
    )
    return await _call_gemini(prompt)

def should_escalate(bot_response: str) -> bool:
    return "[ESCALATE_TO_HUMAN]" in bot_response

# ─────────────────────────────────────────────────────────────────────────────
# Agent auto-assignment (load-balance + priority preemption)
# ─────────────────────────────────────────────────────────────────────────────
PRIORITY_ORDER = {"critical": 4, "high": 3, "medium": 2, "low": 1}

async def find_free_agent() -> Optional[Dict]:
    """
    Returns the approved agent with the fewest active (non-resolved) tickets.
    Returns None if no approved agents exist.
    """
    users_col   = getattr(app.state, "users_collection",   None)
    tickets_col = getattr(app.state, "tickets_collection", None)
    if users_col is None:
        return None

    agents = await users_col.find({"role": "agent", "approved": True}).to_list(200)
    if not agents:
        return None

    best_agent, best_load = None, float("inf")
    for agent in agents:
        agent_id = str(agent.get("id", ""))
        load = 0
        if tickets_col is not None:
            load = await tickets_col.count_documents({
                "assigned_agent_id": agent_id,
                "status": {"$in": ["open", "in_progress"]}
            })
        if load < best_load:
            best_load  = load
            best_agent = agent
    return best_agent

async def preempt_if_needed(new_ticket: Dict) -> Optional[str]:
    """
    If new ticket priority > the priority of the ticket currently being worked
    by the assigned agent, reassign the new ticket to that agent immediately
    (the old ticket goes back to 'open' / unassigned).
    Returns the agent_id if preemption happened, else None.
    """
    tickets_col = getattr(app.state, "tickets_collection", None)
    users_col   = getattr(app.state, "users_collection",   None)
    if tickets_col is None or users_col is None:
        return None

    new_prio = PRIORITY_ORDER.get(new_ticket.get("priority", "low"), 1)
    if new_prio <= PRIORITY_ORDER["medium"]:       # only preempt for high / critical
        return None

    # Find agents currently working on a lower-priority ticket
    agents = await users_col.find({"role": "agent", "approved": True}).to_list(200)
    for agent in agents:
        agent_id = str(agent.get("id", ""))
        active = await tickets_col.find_one({
            "assigned_agent_id": agent_id,
            "status": "in_progress"
        }, sort=[("priority_order", 1)])           # lowest priority first

        if active is None:
            continue
        active_prio = PRIORITY_ORDER.get(active.get("priority", "low"), 1)
        if new_prio > active_prio:
            # Bump the lower-priority ticket back to open / unassigned
            await tickets_col.update_one(
                {"_id": active["_id"]},
                {"$set": {
                    "assigned_agent_id": None,
                    "status": "open",
                    "updated_at": datetime.utcnow(),
                    "preempted_by": str(new_ticket.get("id", "")),
                }}
            )
            logging.info(
                f"⚡ Preempted ticket {active.get('id')} "
                f"({active.get('priority')}) for new {new_ticket.get('priority')} ticket"
            )
            return agent_id
    return None

async def assign_ticket_to_agent(ticket_id: str, priority: str, subject: str,
                                  deadline: datetime) -> Optional[str]:
    """
    1. Try preemption (high/critical only).
    2. Otherwise assign to the freest approved agent.
    Returns assigned agent_id or None.
    """
    tickets_col = getattr(app.state, "tickets_collection", None)
    users_col   = getattr(app.state, "users_collection",   None)
    if tickets_col is None:
        return None

    ticket_doc = await tickets_col.find_one({"id": ticket_id})
    if ticket_doc is None:
        return None

    # 1. Try preemption
    agent_id = await preempt_if_needed(ticket_doc)

    # 2. Fall back to free-agent assignment
    if agent_id is None:
        agent = await find_free_agent()
        if agent is None:
            return None
        agent_id = str(agent.get("id", ""))

    # Assign the new ticket
    await tickets_col.update_one(
        {"id": ticket_id},
        {"$set": {
            "assigned_agent_id": agent_id,
            "status": "in_progress",
            "updated_at": datetime.utcnow(),
        }}
    )
    logging.info(f"✅ Ticket {ticket_id} assigned to agent {agent_id}")

    # Email notification to agent (fire and forget)
    if users_col is not None:
        agent_doc = await users_col.find_one({"id": agent_id})
        if agent_doc and agent_doc.get("email"):
            asyncio.create_task(send_ticket_assigned_email(
                to_email=agent_doc["email"],
                agent_name=agent_doc.get("name", "Agent"),
                ticket_subject=subject,
                ticket_id=ticket_id,
                priority=priority,
                deadline=deadline,
            ))
    return agent_id

async def create_mongo_ticket(customer_id: str, subject: str, description: str,
                               domain: str, failure_reason: str,
                               priority: Optional[str] = None) -> Optional[str]:
    col = getattr(app.state, "tickets_collection", None)
    if col is None:
        return None
    try:
        if priority is None:
            priority = detect_priority(description or subject)
        deadline    = sla_deadline(priority)
        ticket_id   = str(uuid.uuid4())
        doc = {
            "id":               ticket_id,
            "customer_id":      customer_id,
            "subject":          subject,
            "description":      description,
            "domain":           domain,
            "failure_reason":   failure_reason,
            "priority":         priority,
            "priority_order":   PRIORITY_ORDER.get(priority, 1),
            "sla_deadline":     deadline,
            "status":           "open",
            "assigned_agent_id": None,
            "messages":         [],
            "created_at":       datetime.utcnow(),
            "updated_at":       datetime.utcnow(),
        }
        await col.insert_one(doc)
        logging.info(f"🎫 Ticket {ticket_id} created | priority={priority} | deadline={deadline}")

        # Auto-assign in the background so chat response is not delayed
        asyncio.create_task(assign_ticket_to_agent(ticket_id, priority, subject, deadline))

        return ticket_id
    except Exception as e:
        logging.error(f"Ticket creation error: {e}")
        return None

# ─────────────────────────────────────────────────────────────────────────────
# FAQ helpers
# ─────────────────────────────────────────────────────────────────────────────
async def insert_faq_document(domain: str, keywords: list, answer: str):
    col = getattr(app.state, "faq_kb_collection", None)
    if col is None:
        return None
    try:
        result = await col.insert_one({"domain": domain, "keywords": keywords,
                                       "answer": answer, "created_at": datetime.utcnow()})
        return str(result.inserted_id)
    except Exception as e:
        logging.error(f"FAQ insert error: {e}")
        return None

async def create_dynamic_faq(domain: str, user_query: str, bot_response: str):
    words = [w for w in user_query.lower().split() if len(w) > 3][:8]
    if words and bot_response:
        await insert_faq_document(domain, words, bot_response)

async def save_chat_history_message(session_id: str, role: str, content: str,
                                     meta: Optional[Dict] = None):
    col = getattr(app.state, "chat_history_collection", None)
    if col is None:
        return
    try:
        await col.insert_one({"session_id": session_id, "role": role,
                               "content": content, "timestamp": datetime.utcnow(),
                               "meta": meta or {}})
    except Exception as e:
        logging.error(f"Chat history error: {e}")

# ─────────────────────────────────────────────────────────────────────────────
# Lifespan
# ─────────────────────────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Redis
    try:
        app.state.redis = await aioredis.from_url(
            f"redis://{REDIS_HOST}:{REDIS_PORT}", decode_responses=True)
        logging.info("✅ Redis connected.")
    except Exception as e:
        app.state.redis = None
        logging.warning(f"Redis unavailable (non-fatal): {e}")

    # MongoDB
    try:
        if MONGO_URI:
            app.state.mongo_client = motor.motor_asyncio.AsyncIOMotorClient(MONGO_URI)
            default_db = app.state.mongo_client.get_default_database()
            db = default_db if default_db is not None else app.state.mongo_client["chatbot"]
            app.state.chatbot_db = db

            app.state.users_collection               = db["users"]
            app.state.tickets_collection             = db["tickets"]
            app.state.orders_collection              = db["orders"]
            app.state.cases_collection               = db["cases"]
            app.state.customers_collection           = db["customers"]
            app.state.chat_history_collection        = db["chat_history"]
            app.state.faq_kb_collection              = db["faq_knowledge_base"]
            app.state.chat_sessions_collection       = db["chat_sessions"]
            app.state.contact_submissions_collection = db["contact_submissions"]
            app.state.otp_collection                 = db["otp_store"]          # NEW

            # Indexes
            await app.state.users_collection.create_index("email", unique=True)
            await app.state.tickets_collection.create_index("customer_id")
            await app.state.tickets_collection.create_index("assigned_agent_id")
            await app.state.tickets_collection.create_index("status")
            await app.state.tickets_collection.create_index("priority_order")
            await app.state.chat_history_collection.create_index("session_id")
            await app.state.chat_sessions_collection.create_index("user_id")
            await app.state.cases_collection.create_index("customer_id")
            # TTL index: OTPs auto-expire after 10 minutes
            await app.state.otp_collection.create_index(
                "created_at", expireAfterSeconds=OTP_EXPIRE_MINUTES * 60)

            await app.state.mongo_client.admin.command("ping")
            logging.info("✅ MongoDB connected.")
        else:
            raise RuntimeError("MONGODB_URI not set")
    except Exception as e:
        logging.error(f"MongoDB startup failed: {e}")
        for attr in ["mongo_client","chatbot_db","users_collection","tickets_collection",
                     "orders_collection","cases_collection","customers_collection",
                     "chat_history_collection","faq_kb_collection","chat_sessions_collection",
                     "contact_submissions_collection","otp_collection"]:
            setattr(app.state, attr, None)

    yield

    if getattr(app.state, "redis", None) is not None:
        await app.state.redis.close()
    if getattr(app.state, "mongo_client", None) is not None:
        app.state.mongo_client.close()
    logging.info("App shutdown complete.")

# ─────────────────────────────────────────────────────────────────────────────
# App
# ─────────────────────────────────────────────────────────────────────────────
app = FastAPI(title="EchoMind Support Backend", version="3.0.0", lifespan=lifespan)

app.add_middleware(CORSMiddleware,
    allow_origins=[FRONTEND_ORIGIN], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"])

starlette_config = Config(environ={
    "GOOGLE_CLIENT_ID":     os.getenv("GOOGLE_CLIENT_ID",     ""),
    "GOOGLE_CLIENT_SECRET": os.getenv("GOOGLE_CLIENT_SECRET", ""),
})
oauth = OAuth(starlette_config)
oauth.register(name="google",
    server_metadata_url="https://accounts.google.com/.well-known/openid-configuration",
    client_kwargs={"scope": "openid email profile"})

# ─────────────────────────────────────────────────────────────────────────────
# Pydantic models
# ─────────────────────────────────────────────────────────────────────────────
class UserRegister(BaseModel):
    name:     str
    email:    EmailStr
    password: str
    role:     str = "user"

class OtpRequest(BaseModel):
    email: EmailStr
    name:  str
    role:  str = "user"

class OtpVerify(BaseModel):
    email:    EmailStr
    otp:      str
    name:     str
    password: str
    role:     str = "user"

class LoginRequest(BaseModel):
    email:    EmailStr
    password: str

class Token(BaseModel):
    access_token:  str
    token_type:    str
    refresh_token: Optional[str] = None

class ChatMessage(BaseModel):
    role:      str
    content:   str
    timestamp: str

class CustomerProfile(BaseModel):
    customer_id:           str
    previous_interactions: List[str] = []
    purchase_history:      List[str] = []
    preference_settings:   dict      = {}
    sentiment_history:     List[str] = []
    active_case_id:        Optional[str] = None

class ChatRequest(BaseModel):
    user_query:           str
    session_id:           str
    customer_profile:     CustomerProfile
    conversation_history: List[ChatMessage] = []
    domain:               str

class ChatResponse(BaseModel):
    bot_response:     str
    case_status:      str           = "open"
    case_id:          Optional[str] = None
    faq_suggestion:   Optional[str] = None
    sentiment_detected:  Optional[str]   = None
    predicted_domain:    Optional[str]   = None
    intent_confidence:   Optional[float] = None
    intent_source:       Optional[str]   = None
    priority:            Optional[str]   = None
    sla_deadline:        Optional[str]   = None

class TicketCreate(BaseModel):
    customer_id: str
    subject:     str
    description: Optional[str] = None
    priority:    Optional[str] = None

class TicketStatusUpdate(BaseModel):
    status:          str
    resolution_note: Optional[str] = None

class NewFaqEntry(BaseModel):
    domain:    str
    keywords:  List[str]
    answer:    str

class ChatSessionMessage(BaseModel):
    id:        str
    text:      str
    sender:    str
    timestamp: str

class SaveChatSession(BaseModel):
    session_id: str
    domain:     str
    title:      str
    messages:   List[ChatSessionMessage]

class ContactSubmission(BaseModel):
    name:    str
    email:   EmailStr
    subject: str
    message: str
    user_id: Optional[str] = None

class AgentApproval(BaseModel):
    approved: bool
    reason:   Optional[str] = None

# ─────────────────────────────────────────────────────────────────────────────
# Auth helpers
# ─────────────────────────────────────────────────────────────────────────────
async def get_user_by_email_mongo(email: str) -> Optional[Dict]:
    col = getattr(app.state, "users_collection", None)
    if col is None:
        return None
    return await col.find_one({"email": email})

async def get_current_user(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(auth_scheme),
    request: Request = None,
) -> Optional[Dict]:
    token = None
    if credentials:
        token = credentials.credentials
    elif request:
        token = request.cookies.get("access_token")
    if not token:
        return None
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        email   = payload.get("sub")
        return await get_user_by_email_mongo(email) if email else None
    except JWTError:
        return None

def delete_cookies_dependency(response: Response):
    response.delete_cookie("access_token")
    response.delete_cookie("refresh_token")

async def get_ws_user(websocket: WebSocket) -> Optional[Dict]:
    token = websocket.query_params.get("token")
    if not token:
        return None
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        email   = payload.get("sub")
        return await get_user_by_email_mongo(email) if email else None
    except JWTError:
        return None

def _set_access_cookie(response: Response, token: str):
    response.set_cookie(key="access_token", value=token, httponly=True,
                        samesite="lax", max_age=ACCESS_TOKEN_EXPIRE_MINUTES * 60)

def _frontend_redirect_with_token_cookie(token: str) -> RedirectResponse:
    resp = RedirectResponse(url=f"{FRONTEND_ORIGIN}/auth/callback?token={token}")
    resp.set_cookie(key="access_token", value=token, httponly=True, samesite="lax",
                    max_age=ACCESS_TOKEN_EXPIRE_MINUTES * 60)
    return resp

# ─────────────────────────────────────────────────────────────────────────────
# OAuth
# ─────────────────────────────────────────────────────────────────────────────
@app.get("/auth/{provider}/login")
async def oauth_login(provider: str, request: Request):
    if provider != "google":
        raise HTTPException(400, "Unsupported provider")
    return await oauth.google.authorize_redirect(request, f"{API_BASE_URL}/auth/{provider}/callback")

@app.get("/auth/{provider}/callback")
async def oauth_callback(provider: str, request: Request):
    if provider != "google":
        raise HTTPException(400, "Unsupported provider")
    try:
        token_data = await oauth.google.authorize_access_token(request)
        user_info  = token_data.get("userinfo") or {}
        email      = user_info.get("email")
        name       = user_info.get("name", email)
        if not email:
            raise HTTPException(400, "No email from OAuth")
        col = getattr(app.state, "users_collection", None)
        if col is None:
            raise HTTPException(500, "DB unavailable")
        existing = await col.find_one({"email": email})
        if not existing:
            new_user = {"id": str(uuid.uuid4()), "name": name, "email": email,
                        "hashed_password": None, "role": "user",
                        "approved": True, "provider": "google",
                        "created_at": datetime.utcnow()}
            await col.insert_one(new_user)
        token = create_access_token({"sub": email})
        return _frontend_redirect_with_token_cookie(token)
    except Exception as e:
        logging.error(f"OAuth error: {e}")
        raise HTTPException(500, f"OAuth error: {e}")

# ─────────────────────────────────────────────────────────────────────────────
# ── REGISTRATION FLOW ────────────────────────────────────────────────────────
#   Step 1: POST /v1/register/send-otp   → validate email not taken, send OTP
#   Step 2: POST /v1/register/verify-otp → verify OTP, create account
# ─────────────────────────────────────────────────────────────────────────────
@app.post("/v1/register/send-otp")
async def send_otp(data: OtpRequest):
    """Send a 6-digit OTP to the given email. Validates email is not taken."""
    users_col = getattr(app.state, "users_collection", None)
    otp_col   = getattr(app.state, "otp_collection",   None)

    if users_col is not None:
        existing = await users_col.find_one({"email": data.email})
        if existing:
            raise HTTPException(status_code=409, detail="Email already registered")

    otp_code = str(random.randint(100000, 999999))

    # Store (upsert) OTP — TTL index auto-expires it after OTP_EXPIRE_MINUTES
    if otp_col is not None:
        await otp_col.update_one(
            {"email": data.email},
            {"$set": {
                "email":      data.email,
                "otp":        otp_code,
                "name":       data.name,
                "role":       data.role,
                "created_at": datetime.utcnow(),
            }},
            upsert=True,
        )

    await send_otp_email(data.email, data.name, otp_code)
    logging.info(f"OTP sent to {data.email}")
    return {"ok": True, "message": f"OTP sent to {data.email}. Valid for {OTP_EXPIRE_MINUTES} minutes."}

@app.post("/v1/register/verify-otp")
async def verify_otp_and_register(data: OtpVerify):
    """Verify OTP and create the user account."""
    otp_col   = getattr(app.state, "otp_collection",   None)
    users_col = getattr(app.state, "users_collection", None)

    # Verify OTP
    if otp_col is not None:
        record = await otp_col.find_one({"email": data.email})
        if not record:
            raise HTTPException(status_code=400, detail="OTP expired or not found. Please request a new one.")
        if record.get("otp") != data.otp:
            raise HTTPException(status_code=400, detail="Invalid OTP. Please try again.")
        # Consume OTP
        await otp_col.delete_one({"email": data.email})
    else:
        logging.warning("OTP collection unavailable — skipping OTP check (dev mode)")

    # Check email not taken (race-condition guard)
    if users_col is not None:
        existing = await users_col.find_one({"email": data.email})
        if existing:
            raise HTTPException(status_code=409, detail="Email already registered")

    allowed_roles = ["user", "agent", "admin"]
    role          = data.role if data.role in allowed_roles else "user"

    # Agents start as pending until admin approves
    approved = (role != "agent")

    customer_id = str(uuid.uuid4())
    new_user = {
        "id":              customer_id,
        "name":            data.name,
        "email":           data.email,
        "hashed_password": get_password_hash(data.password),
        "role":            role,
        "approved":        approved,
        "provider":        "local",
        "created_at":      datetime.utcnow(),
    }
    if users_col is not None:
        await users_col.insert_one(new_user)

    access_token  = create_access_token({"sub": data.email})
    refresh_token = create_refresh_token({"sub": data.email})

    return {
        "ok":          True,
        "customer_id": customer_id,
        "name":        data.name,
        "email":       data.email,
        "role":        role,
        "approved":    approved,
        "access_token":  access_token,
        "refresh_token": refresh_token,
        "message": (
            "Registration successful! Your account is pending admin approval."
            if not approved else
            "Registration successful!"
        ),
    }

# Legacy single-step register (kept for compatibility / admin creation)
@app.post("/v1/register")
async def register_legacy(data: UserRegister):
    users_col = getattr(app.state, "users_collection", None)
    if users_col is not None:
        existing = await users_col.find_one({"email": data.email})
        if existing:
            raise HTTPException(status_code=409, detail="Email already registered")

    allowed_roles = ["user", "agent", "admin"]
    role          = data.role if data.role in allowed_roles else "user"
    approved      = (role != "agent")
    customer_id   = str(uuid.uuid4())

    new_user = {
        "id":              customer_id,
        "name":            data.name,
        "email":           data.email,
        "hashed_password": get_password_hash(data.password),
        "role":            role,
        "approved":        approved,
        "provider":        "local",
        "created_at":      datetime.utcnow(),
    }
    if users_col is not None:
        await users_col.insert_one(new_user)

    access_token  = create_access_token({"sub": data.email})
    refresh_token = create_refresh_token({"sub": data.email})
    return {"customer_id": customer_id, "name": data.name, "email": data.email,
            "role": role, "approved": approved,
            "access_token": access_token, "refresh_token": refresh_token}

# ─────────────────────────────────────────────────────────────────────────────
# Login / logout / me
# ─────────────────────────────────────────────────────────────────────────────
@app.post("/login", response_model=Token)
async def login(response: Response, request: LoginRequest):
    user = await get_user_by_email_mongo(request.email)
    if not user or not verify_password(request.password, user.get("hashed_password")):
        raise HTTPException(status_code=401, detail="Invalid email or password")

    # Block unapproved agents
    if user.get("role") == "agent" and not user.get("approved", False):
        raise HTTPException(status_code=403,
            detail="Your agent account is pending admin approval. You'll receive an email once approved.")

    access_token  = create_access_token({"sub": request.email})
    refresh_token = create_refresh_token({"sub": request.email})
    _set_access_cookie(response, access_token)
    return {"access_token": access_token, "token_type": "bearer", "refresh_token": refresh_token}

@app.post("/logout")
async def logout(response: Response, _=Depends(delete_cookies_dependency)):
    return {"ok": True}

@app.post("/refresh", response_model=Token)
async def refresh_token_endpoint(refresh_token: str = Body(..., embed=True)):
    try:
        payload = jwt.decode(refresh_token, SECRET_KEY, algorithms=[ALGORITHM])
        email   = payload.get("sub")
        if not email:
            raise HTTPException(401, "Invalid token")
        return {"access_token": create_access_token({"sub": email}), "token_type": "bearer"}
    except JWTError:
        raise HTTPException(401, "Invalid or expired refresh token")

@app.get("/me")
async def me(current_user: Optional[Dict] = Depends(get_current_user)):
    if not current_user:
        raise HTTPException(401, "Not authenticated")
    return {
        "id":       str(current_user.get("id", "")),
        "name":     current_user.get("name",  ""),
        "email":    current_user.get("email", ""),
        "role":     current_user.get("role",  "user"),
        "approved": current_user.get("approved", True),
    }

# ─────────────────────────────────────────────────────────────────────────────
# WebSocket
# ─────────────────────────────────────────────────────────────────────────────
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    user = await get_ws_user(websocket)
    if not user:
        await websocket.close(code=4001)
        return
    await websocket.accept()
    try:
        while True:
            data = await websocket.receive_text()
            await websocket.send_text(json.dumps({"type": "pong", "data": data}))
    except WebSocketDisconnect:
        pass

# ─────────────────────────────────────────────────────────────────────────────
# ADMIN — agent approval
# ─────────────────────────────────────────────────────────────────────────────
@app.get("/admin/agents/pending")
async def list_pending_agents(current_user: Optional[Dict] = Depends(get_current_user)):
    """Return agents awaiting approval."""
    if not current_user or current_user.get("role") != "admin":
        raise HTTPException(403, "Admin only")
    col = getattr(app.state, "users_collection", None)
    if col is None:
        raise HTTPException(500, "Database not available")
    agents = await col.find({"role": "agent", "approved": False}).to_list(200)
    for a in agents:
        a["_id"] = str(a["_id"])
        a.pop("hashed_password", None)
    return {"pending_agents": agents}

@app.post("/admin/agents/{agent_id}/approve")
async def approve_agent(agent_id: str, body: AgentApproval,
                        current_user: Optional[Dict] = Depends(get_current_user)):
    """Admin approves or rejects a pending agent."""
    if not current_user or current_user.get("role") != "admin":
        raise HTTPException(403, "Admin only")
    col = getattr(app.state, "users_collection", None)
    if col is None:
        raise HTTPException(500, "Database not available")

    agent = await col.find_one({"id": agent_id})
    if not agent:
        raise HTTPException(404, "Agent not found")
    if agent.get("role") != "agent":
        raise HTTPException(400, "User is not an agent")

    if body.approved:
        await col.update_one({"id": agent_id},
                             {"$set": {"approved": True, "approval_note": body.reason}})
    else:
        # Rejected → delete the account
        await col.delete_one({"id": agent_id})

    asyncio.create_task(send_approval_email(
        to_email=agent["email"],
        agent_name=agent.get("name", "Agent"),
        approved=body.approved,
    ))
    return {"ok": True, "agent_id": agent_id, "approved": body.approved}

@app.get("/admin/agents")
async def admin_list_agents(current_user: Optional[Dict] = Depends(get_current_user)):
    if not current_user or current_user.get("role") != "admin":
        raise HTTPException(403, "Admin only")
    users_col   = getattr(app.state, "users_collection",   None)
    tickets_col = getattr(app.state, "tickets_collection", None)
    if users_col is None:
        raise HTTPException(500, "Database not available")

    agents = await users_col.find({"role": "agent"}).to_list(200)
    result = []
    for a in agents:
        a["_id"] = str(a["_id"])
        a.pop("hashed_password", None)
        agent_id = str(a.get("id",""))
        if tickets_col is not None:
            total    = await tickets_col.count_documents({"assigned_agent_id": agent_id})
            resolved = await tickets_col.count_documents({"assigned_agent_id": agent_id, "status": "resolved"})
            active   = await tickets_col.count_documents({"assigned_agent_id": agent_id, "status": {"$in": ["open","in_progress"]}})
        else:
            total = resolved = active = 0
        a["total_tickets"]    = total
        a["resolved_tickets"] = resolved
        a["active_tickets"]   = active
        a["satisfaction"]     = 0.0
        a["is_online"]        = False
        result.append(a)
    return result

@app.get("/admin/users")
async def admin_list_users(current_user: Optional[Dict] = Depends(get_current_user)):
    if not current_user or current_user.get("role") != "admin":
        raise HTTPException(403, "Admin only")
    col = getattr(app.state, "users_collection", None)
    if col is None:
        raise HTTPException(500, "Database not available")
    users = await col.find({}).to_list(500)
    for u in users:
        u["_id"] = str(u["_id"])
        u.pop("hashed_password", None)
    return {"users": users}

@app.delete("/admin/users/{user_id}")
async def admin_delete_user(user_id: str, current_user: Optional[Dict] = Depends(get_current_user)):
    if not current_user or current_user.get("role") != "admin":
        raise HTTPException(403, "Admin only")
    col = getattr(app.state, "users_collection", None)
    if col is None:
        raise HTTPException(500, "Database not available")
    result = await col.delete_one({"id": user_id})
    if result.deleted_count == 0:
        raise HTTPException(404, "User not found")
    return {"ok": True}

# ─────────────────────────────────────────────────────────────────────────────
# ADMIN — tickets
# ─────────────────────────────────────────────────────────────────────────────
@app.get("/admin/tickets")
async def admin_list_tickets(u: Optional[Dict] = Depends(get_current_user)):
    if not u or u.get("role") != "admin":
        raise HTTPException(403, "Admin only")
    col = getattr(app.state, "tickets_collection", None)
    if col is None:
        raise HTTPException(500, "DB unavailable")
    
    tickets = await col.find({}).sort([("priority_order", -1), ("created_at", -1)]).to_list(200)
    users_col = getattr(app.state, "users_collection", None)
    
    for t in tickets:
        t["_id"] = str(t["_id"])
        if isinstance(t.get("sla_deadline"), datetime):
            t["sla_deadline"] = t["sla_deadline"].isoformat()
        
        # Enrich customer name
        if users_col is not None and t.get("customer_id") and not t.get("customer_name"):
            try:
                ud = await users_col.find_one({"id": t["customer_id"]})
                if ud:
                    t["customer_name"] = ud.get("name", "")
                    t["customer_email"] = ud.get("email", "")
            except Exception:
                pass
        
        # Enrich assigned agent name
        if users_col is not None and t.get("assigned_agent_id") and not t.get("assigned_agent_name"):
            try:
                ad = await users_col.find_one({"id": t["assigned_agent_id"]})
                if ad:
                    t["assigned_agent_name"] = ad.get("name", "")
            except Exception:
                pass
    
    return {"tickets": tickets}

@app.put("/admin/tickets/{ticket_id}/assign")
async def admin_assign_ticket(ticket_id: str, body: dict = Body(...),
                               u: Optional[Dict] = Depends(get_current_user)):
    if not u or u.get("role") != "admin":
        raise HTTPException(403, "Admin only")
    col = getattr(app.state, "tickets_collection", None)
    if col is None:
        raise HTTPException(500, "DB unavailable")
    
    agent_id = body.get("agent_id")
    if not agent_id:
        raise HTTPException(400, "agent_id is required")
    
    # Try both UUID id field and MongoDB _id
    update_data = {
        "assigned_agent_id": agent_id,
        "status": "in_progress",
        "updated_at": datetime.utcnow()
    }
    
    # First try UUID id field
    result = await col.update_one({"id": ticket_id}, {"$set": update_data})
    
    # If not found, try ObjectId _id
    if result.matched_count == 0:
        try:
            result = await col.update_one(
                {"_id": ObjectId(ticket_id)}, {"$set": update_data}
            )
        except Exception:
            pass
    
    if result.matched_count == 0:
        raise HTTPException(404, "Ticket not found")
    
    # Also notify the assigned agent by email
    users_col = getattr(app.state, "users_collection", None)
    if users_col is not None and agent_id:
        agent_doc = await users_col.find_one({"id": agent_id})
        if not agent_doc:
            agent_doc = await users_col.find_one({"_id": ObjectId(agent_id)}) if len(agent_id) == 24 else None
        if agent_doc and agent_doc.get("email"):
            ticket_doc = await col.find_one({"id": ticket_id})
            if ticket_doc:
                asyncio.create_task(send_ticket_assigned_email(
                    to=agent_doc["email"],
                    name=agent_doc.get("name", "Agent"),
                    subject=ticket_doc.get("subject", "Support Ticket"),
                    ticket_id=ticket_id,
                    priority=ticket_doc.get("priority", "medium"),
                    deadline=ticket_doc.get("sla_deadline", datetime.utcnow()),
                    domain=ticket_doc.get("domain", "general"),
                ))
    
    return {"ok": True, "ticket_id": ticket_id, "assigned_agent_id": agent_id}

@app.get("/admin/tickets/{ticket_id}/messages")
async def admin_get_messages(ticket_id: str, current_user: Optional[Dict] = Depends(get_current_user)):
    if not current_user or current_user.get("role") not in ["admin","agent"]:
        raise HTTPException(403, "Admin/Agent only")
    col = getattr(app.state, "chat_history_collection", None)
    if col is None:
        return []
    msgs = await col.find({"session_id": ticket_id}).sort("timestamp",1).to_list(200)
    for m in msgs:
        m["_id"] = str(m["_id"])
    return msgs

@app.post("/admin/tickets/{ticket_id}/reply")
async def admin_reply(ticket_id: str, body: dict = Body(...),
                      current_user: Optional[Dict] = Depends(get_current_user)):
    if not current_user or current_user.get("role") not in ["admin","agent"]:
        raise HTTPException(403, "Admin/Agent only")
    await save_chat_history_message(ticket_id, body.get("sender_role","admin"),
                                    body.get("message",""), {"from_admin": True})
    return {"ok": True}

@app.post("/admin/tickets/{ticket_id}/resolve")
async def admin_resolve_ticket(ticket_id: str, body: dict = Body(default={}),
                                current_user: Optional[Dict] = Depends(get_current_user)):
    if not current_user or current_user.get("role") != "admin":
        raise HTTPException(403, "Admin only")
    col = getattr(app.state, "tickets_collection", None)
    if col is None:
        raise HTTPException(500, "Database not available")
    result = await col.update_one({"id": ticket_id}, {"$set": {
        "status": "resolved",
        "resolution_note": body.get("resolution_note",""),
        "updated_at": datetime.utcnow(),
    }})
    if result.matched_count == 0:
        raise HTTPException(404, "Ticket not found")
    return {"ok": True}

# ─────────────────────────────────────────────────────────────────────────────
# AGENT — tickets
# ─────────────────────────────────────────────────────────────────────────────
@app.get("/agent/tickets")
async def agent_list_tickets(current_user: Optional[Dict] = Depends(get_current_user)):
    if not current_user or current_user.get("role") not in ["agent","admin"]:
        raise HTTPException(403, "Agent only")
    col = getattr(app.state, "tickets_collection", None)
    if col is None:
        raise HTTPException(500, "Database not available")
    agent_id = str(current_user.get("id",""))
    tickets  = await col.find({"assigned_agent_id": agent_id}).sort(
        [("priority_order",-1),("updated_at",-1)]).to_list(100)
    for t in tickets:
        t["_id"] = str(t["_id"])
        if isinstance(t.get("sla_deadline"), datetime):
            t["sla_deadline"] = t["sla_deadline"].isoformat()
    return {"tickets": tickets}

@app.get("/agent/tickets/{ticket_id}/messages")
async def agent_get_messages(ticket_id: str, current_user: Optional[Dict] = Depends(get_current_user)):
    if not current_user or current_user.get("role") not in ["agent","admin"]:
        raise HTTPException(403, "Agent only")
    col = getattr(app.state, "chat_history_collection", None)
    if col is None:
        return []
    msgs = await col.find({"session_id": ticket_id}).sort("timestamp",1).to_list(200)
    for m in msgs:
        m["_id"] = str(m["_id"])
    return msgs

@app.post("/agent/tickets/{ticket_id}/reply")
async def agent_reply(ticket_id: str, body: dict = Body(...),
                      current_user: Optional[Dict] = Depends(get_current_user)):
    if not current_user or current_user.get("role") not in ["agent","admin"]:
        raise HTTPException(403, "Agent only")
    await save_chat_history_message(ticket_id, body.get("sender_role","agent"),
                                    body.get("message",""), {"from_agent": True})
    return {"ok": True}

@app.post("/agent/tickets/{ticket_id}/resolve")
async def agent_resolve_ticket(ticket_id: str, body: dict = Body(default={}),
                                current_user: Optional[Dict] = Depends(get_current_user)):
    if not current_user or current_user.get("role") not in ["agent","admin"]:
        raise HTTPException(403, "Agent only")
    col       = getattr(app.state, "tickets_collection", None)
    users_col = getattr(app.state, "users_collection",   None)
    if col is None:
        raise HTTPException(500, "Database not available")

    resolution_note = body.get("resolution_note","")
    result = await col.update_one({"id": ticket_id}, {"$set": {
        "status":          "resolved",
        "resolution_note": resolution_note,
        "resolved_by":     str(current_user.get("id","")),
        "updated_at":      datetime.utcnow(),
    }})
    if result.matched_count == 0:
        raise HTTPException(404, "Ticket not found")

    # Email customer
    if users_col is not None:
        ticket_doc = await col.find_one({"id": ticket_id})
        if ticket_doc:
            user_doc = await users_col.find_one({"id": ticket_doc.get("customer_id","")})
            if user_doc and user_doc.get("email"):
                asyncio.create_task(send_resolution_email(
                    to_email=user_doc["email"],
                    user_name=user_doc.get("name","Customer"),
                    ticket_subject=ticket_doc.get("subject","Support Ticket"),
                    ticket_id=ticket_doc.get("id", ticket_id),
                    resolution_note=resolution_note,
                ))

    return {"ok": True, "ticket_id": ticket_id, "status": "resolved"}

@app.post("/agent/tickets/{ticket_id}/accept")
async def agent_accept_ticket(ticket_id: str, current_user: Optional[Dict] = Depends(get_current_user)):
    if not current_user or current_user.get("role") not in ["agent","admin"]:
        raise HTTPException(403, "Agent only")
    col = getattr(app.state, "tickets_collection", None)
    if col is None:
        raise HTTPException(500, "Database not available")
    result = await col.update_one({"id": ticket_id}, {"$set": {
        "status":            "in_progress",
        "assigned_agent_id": str(current_user.get("id","")),
        "updated_at":        datetime.utcnow(),
    }})
    if result.matched_count == 0:
        raise HTTPException(404, "Ticket not found")
    return {"ok": True, "status": "in_progress"}

# ─────────────────────────────────────────────────────────────────────────────
# Tickets (user-facing)
# ─────────────────────────────────────────────────────────────────────────────
@app.post("/tickets", status_code=status.HTTP_201_CREATED)
async def create_ticket_endpoint(data: TicketCreate):
    ticket_id = await create_mongo_ticket(
        customer_id=data.customer_id, subject=data.subject,
        description=data.description or "Manual creation.",
        domain="manual", failure_reason="Manual API creation.",
        priority=data.priority,
    )
    if not ticket_id:
        raise HTTPException(500, "Failed to create ticket.")
    return {"ticket_id": ticket_id, "status": "open"}

@app.get("/tickets/customer/{customer_id}")
async def list_tickets_by_customer(customer_id: str):
    col = getattr(app.state, "tickets_collection", None)
    if col is None:
        raise HTTPException(500, "Ticket database not available.")
    cursor  = col.find({"customer_id": customer_id}).sort("created_at",-1)
    tickets = await cursor.to_list(50)
    for t in tickets:
        t["_id"] = str(t["_id"])
        if isinstance(t.get("sla_deadline"), datetime):
            t["sla_deadline"] = t["sla_deadline"].isoformat()
    return {"tickets": tickets}

@app.get("/tickets/{ticket_id}")
async def get_ticket(ticket_id: str):
    col = getattr(app.state, "tickets_collection", None)
    if col is None:
        raise HTTPException(500, "Ticket database not available.")
    ticket_doc = await col.find_one({"id": ticket_id})
    if not ticket_doc:
        raise HTTPException(404, "Ticket not found.")
    ticket_doc["_id"] = str(ticket_doc["_id"])
    if isinstance(ticket_doc.get("sla_deadline"), datetime):
        ticket_doc["sla_deadline"] = ticket_doc["sla_deadline"].isoformat()
    return ticket_doc

@app.put("/tickets/{ticket_id}/status")
async def update_ticket_status(ticket_id: str, update: TicketStatusUpdate):
    col       = getattr(app.state, "tickets_collection", None)
    users_col = getattr(app.state, "users_collection",   None)
    if col is None:
        raise HTTPException(500, "Ticket database not available.")

    update_data = {"status": update.status.lower(), "updated_at": datetime.utcnow()}
    if update.resolution_note:
        update_data["resolution_note"] = update.resolution_note

    result = await col.update_one({"id": ticket_id}, {"$set": update_data})
    if result.matched_count == 0:
        raise HTTPException(404, "Ticket not found.")

    if update.status.lower() == "resolved" and users_col is not None:
        ticket_doc = await col.find_one({"id": ticket_id})
        if ticket_doc:
            user_doc = await users_col.find_one({"id": ticket_doc.get("customer_id","")})
            if user_doc and user_doc.get("email"):
                asyncio.create_task(send_resolution_email(
                    to_email=user_doc["email"],
                    user_name=user_doc.get("name","Customer"),
                    ticket_subject=ticket_doc.get("subject","Support Ticket"),
                    ticket_id=ticket_id,
                    resolution_note=update.resolution_note or "",
                ))
    return {"ticket_id": ticket_id, "new_status": update.status.lower(), "ok": True}

# ─────────────────────────────────────────────────────────────────────────────
# Cases
# ─────────────────────────────────────────────────────────────────────────────
@app.get("/cases/customer/{customer_id}")
async def list_cases_by_customer(customer_id: str):
    col = getattr(app.state, "cases_collection", None)
    if col is None:
        raise HTTPException(500, "Cases database not available.")
    cases = await col.find({"customer_id": customer_id}).sort("last_updated",-1).to_list(50)
    for c in cases:
        c["_id"] = str(c["_id"])
    return {"cases": cases}

@app.post("/cases/{case_id}/resolve")
async def resolve_case(case_id: str, body: dict = Body(default={})):
    col = getattr(app.state, "cases_collection", None)
    if col is None:
        raise HTTPException(500, "Cases database not available.")
    query_id   = ObjectId(case_id) if len(case_id) == 24 else case_id
    update_data = {"status": "resolved", "last_updated": datetime.utcnow()}
    if body.get("summary"):
        update_data["summary"] = body["summary"]
    result = await col.update_one({"_id": query_id}, {"$set": update_data})
    if result.matched_count == 0:
        raise HTTPException(404, "Case not found.")
    return {"ok": True, "case_id": case_id, "status": "resolved"}

# ─────────────────────────────────────────────────────────────────────────────
# Chat sessions
# ─────────────────────────────────────────────────────────────────────────────
@app.get("/chat-sessions/{user_id}")
async def get_chat_sessions(user_id: str, current_user: Optional[Dict] = Depends(get_current_user)):
    if not current_user:
        raise HTTPException(401, "Authentication required")
    col = getattr(app.state, "chat_sessions_collection", None)
    if col is None:
        return {"sessions": []}
    sessions = await col.find({"user_id": user_id}).sort("updated_at",-1).to_list(100)
    for s in sessions:
        s["_id"] = str(s["_id"])
    return {"sessions": sessions}

@app.post("/chat-sessions")
async def save_chat_session(data: SaveChatSession, current_user: Optional[Dict] = Depends(get_current_user)):
    if not current_user:
        raise HTTPException(401, "Authentication required")
    col = getattr(app.state, "chat_sessions_collection", None)
    if col is None:
        raise HTTPException(500, "Database not available")
    user_id  = str(current_user.get("id",""))
    existing = await col.find_one({"session_id": data.session_id, "user_id": user_id})
    msgs     = [m.dict() for m in data.messages]
    if existing:
        await col.update_one({"session_id": data.session_id, "user_id": user_id},
                             {"$set": {"messages": msgs, "title": data.title, "updated_at": datetime.utcnow()}})
    else:
        await col.insert_one({"session_id": data.session_id, "user_id": user_id,
                              "domain": data.domain, "title": data.title,
                              "messages": msgs,
                              "created_at": datetime.utcnow(), "updated_at": datetime.utcnow()})
    return {"ok": True}

@app.delete("/chat-sessions/{session_id}")
async def delete_chat_session(session_id: str, current_user: Optional[Dict] = Depends(get_current_user)):
    if not current_user:
        raise HTTPException(401, "Authentication required")
    col = getattr(app.state, "chat_sessions_collection", None)
    if col is None:
        raise HTTPException(500, "Database not available")
    await col.delete_one({"session_id": session_id, "user_id": str(current_user.get("id",""))})
    return {"ok": True}

# ─────────────────────────────────────────────────────────────────────────────
# Contact form
# ─────────────────────────────────────────────────────────────────────────────
@app.post("/contact", status_code=status.HTTP_201_CREATED)
async def submit_contact_form(submission: ContactSubmission):
    col = getattr(app.state, "contact_submissions_collection", None)
    doc = {"name": submission.name, "email": submission.email,
           "subject": submission.subject, "message": submission.message,
           "user_id": submission.user_id, "created_at": datetime.utcnow(), "status": "pending"}
    if col is not None:
        await col.insert_one(doc)
    return {"ok": True, "message": "Your message has been received. We'll get back to you soon!"}

# ─────────────────────────────────────────────────────────────────────────────
# FAQ
# ─────────────────────────────────────────────────────────────────────────────
@app.post("/admin/faq", status_code=status.HTTP_201_CREATED)
async def add_faq_entry(entry: NewFaqEntry, current_user: Optional[Dict] = Depends(get_current_user)):
    if not current_user:
        raise HTTPException(401, "Authentication required.")
    doc_id = await insert_faq_document(entry.domain, entry.keywords, entry.answer)
    return {"ok": True, "id": doc_id}

# ─────────────────────────────────────────────────────────────────────────────
# Health
# ─────────────────────────────────────────────────────────────────────────────
@app.get("/health")
async def health():
    return {"status": "ok",
            "mongo": getattr(app.state, "mongo_client", None) is not None,
            "redis": getattr(app.state, "redis",        None) is not None}

# ─────────────────────────────────────────────────────────────────────────────
# CHAT endpoint
# ─────────────────────────────────────────────────────────────────────────────
SENSITIVE_FINANCE_PATTERN = re.compile(
    r"\b(\d{4}[- ]?\d{4}[- ]?\d{4}[- ]?\d{4}|\d{3}-\d{2}-\d{4})\b")

@app.post("/chat", response_model=ChatResponse)
async def chat_interaction(chat_req: ChatRequest):
    user_query  = chat_req.user_query
    session_id  = chat_req.session_id
    customer_id = chat_req.customer_profile.customer_id

    # Sensitive PII guard
    if SENSITIVE_FINANCE_PATTERN.search(user_query) and chat_req.domain == "finance":
        msg = "I cannot process sensitive financial details through chat. Please contact our secure phone line."
        await save_chat_history_message(session_id, "user", user_query, {"is_sensitive": True})
        await save_chat_history_message(session_id, "bot",  msg,        {"source": "Security Filter"})
        return ChatResponse(bot_response=msg, case_status="blocked")

    sentiment = analyze_sentiment(user_query)

    # Cache
    cached = await get_history_answer(user_query)
    if cached:
        await save_chat_history_message(session_id, "user", user_query)
        await save_chat_history_message(session_id, "bot",  cached, {"source": "Cache"})
        return ChatResponse(bot_response=cached, case_status="resolved",
                            sentiment_detected=sentiment, predicted_domain="cache",
                            intent_confidence=1.0, intent_source="cache")

    predicted_domain, confidence, intent_src = await classify_intent(user_query)
    priority    = detect_priority(user_query)
    deadline    = sla_deadline(priority)

    # KB answer
    kb_answer   = await get_kb_answer(user_query, predicted_domain)
    bot_response, source_type = None, None

    if kb_answer:
        bot_response, source_type = kb_answer, "KB"
    elif gemini_api_key:
        try:
            bot_response = await generate_bot_response(
                user_query, chat_req.conversation_history, predicted_domain, customer_id)
            source_type  = "Gemini"
        except Exception as e:
            logging.error(f"Gemini error: {e}")
            bot_response = "I couldn't generate a response. A human agent will step in."
            source_type  = "Fallback"
    else:
        bot_response = "A support agent will assist you shortly."
        source_type  = "Fallback"

    # Escalation?
    if should_escalate(bot_response) or source_type == "Fallback":
        clean_bot_response = bot_response.replace("[ESCALATE_TO_HUMAN]", "").strip() if bot_response else ""
        if not clean_bot_response:
            clean_bot_response = "Your issue has been escalated to a human agent."

        ticket_id = await create_mongo_ticket(
            customer_id=customer_id,
            subject=f"{priority.upper()}: {predicted_domain} — {user_query[:60]}",
            description=f"Query: {user_query}\nBot: {clean_bot_response}",
            domain=predicted_domain,
            failure_reason="BOT_ESCALATION",
            priority=priority,
        )
        final = (
            f"{clean_bot_response}\n\n"
            f"🎫 Ticket ID: **{ticket_id}**\n"
            f"⏱ Priority: **{priority.upper()}** | SLA: {SLA_HOURS[priority]}h\n"
            f"📅 Deadline: {deadline.strftime('%Y-%m-%d %H:%M UTC')}"
        ) if ticket_id else f"{clean_bot_response}\n\nEscalation failed — please contact support directly."

        await save_chat_history_message(session_id, "user", user_query)
        await save_chat_history_message(session_id, "bot",  final,
                                        {"source": source_type, "ticket_id": ticket_id})
        return ChatResponse(bot_response=final, case_status="escalated", case_id=ticket_id,
                            sentiment_detected=sentiment, predicted_domain=predicted_domain,
                            intent_confidence=confidence, intent_source=intent_src,
                            priority=priority, sla_deadline=deadline.isoformat())

    # Success
    if source_type == "Gemini" and len(user_query) > 10 and len(bot_response) > 20:
        asyncio.create_task(create_dynamic_faq(predicted_domain, user_query, bot_response))

    await save_chat_history_message(session_id, "user", user_query)
    await save_chat_history_message(session_id, "bot",  bot_response, {"source": source_type})

    return ChatResponse(bot_response=bot_response, case_status="resolved",
                        sentiment_detected=sentiment, predicted_domain=predicted_domain,
                        intent_confidence=confidence, intent_source=intent_src,
                        priority=priority, sla_deadline=deadline.isoformat())
