import base64
import os
import secrets
import sqlite3
from datetime import datetime, timezone, date
from functools import wraps
from pathlib import Path

from flask import Flask, jsonify, request, g, abort
from werkzeug.security import check_password_hash, generate_password_hash
import stripe

DATABASE = Path(os.getenv("BILLING_DB", "data/billing.db"))
DAILY_FREE_LIMIT = 10
stripe.api_key = os.getenv("STRIPE_SECRET_KEY")
STRIPE_WEBHOOK_SECRET = os.getenv("STRIPE_WEBHOOK_SECRET", "")

app = Flask(__name__)

def iso_now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")

def get_db():
    if "db" not in g:
        g.db = sqlite3.connect(DATABASE)
        g.db.row_factory = sqlite3.Row
    return g.db

@app.teardown_appcontext
def close_db(_exc):
    conn = g.pop("db", None)
    if conn:
        conn.close()

def init_db():
    DATABASE.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(DATABASE) as conn:
        conn.execute("PRAGMA foreign_keys = ON;")
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                email TEXT UNIQUE NOT NULL,
                password_hash TEXT NOT NULL,
                plan TEXT NOT NULL DEFAULT 'free' CHECK(plan IN ('free','paid')),
                stripe_customer_id TEXT,
                subscription_status TEXT,
                daily_usage INTEGER NOT NULL DEFAULT 0,
                usage_reset_on TEXT NOT NULL DEFAULT (DATE('now')),
                created_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS api_tokens (
                token TEXT PRIMARY KEY,
                user_id INTEGER NOT NULL,
                created_at TEXT NOT NULL,
                FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
            );

            CREATE INDEX IF NOT EXISTS ix_users_email ON users(email);
            """
        )

def parse_basic_auth(header: str):
    if not header or not header.startswith("Basic "):
        return None, None
    decoded = base64.b64decode(header.split(" ", 1)[1]).decode()
    username, _, password = decoded.partition(":")
    return username, password

@app.post("/v1/tokens")
def issue_token():
    username, password = parse_basic_auth(request.headers.get("Authorization"))
    if not username:
        return jsonify({"error": "Basic auth required"}), 401

    db = get_db()
    user = db.execute("SELECT * FROM users WHERE email = ?", (username,)).fetchone()
    if not user or not check_password_hash(user["password_hash"], password):
        return jsonify({"error": "invalid credentials"}), 401

    token = secrets.token_hex(32)
    db.execute(
        """INSERT OR REPLACE INTO api_tokens(token, user_id, created_at)
           VALUES(?, ?, ?)""",
        (token, user["id"], iso_now()),
    )
    db.commit()
    return jsonify({"token": token, "plan": user["plan"]})

def require_token(view):
    @wraps(view)
    def wrapped(*args, **kwargs):
        token = request.headers.get("X-API-Key")
        if not token:
            abort(401, description="X-API-Key header missing")
        db = get_db()
        row = db.execute(
            """SELECT users.* FROM api_tokens
               JOIN users ON users.id = api_tokens.user_id
               WHERE api_tokens.token = ?""",
            (token,),
        ).fetchone()
        if not row:
            abort(401, description="invalid token")
        g.current_user = dict(row)
        return view(*args, **kwargs)

    return wrapped

def enforce_quota(user):
    today = date.today().isoformat()
    db = get_db()
    if user["usage_reset_on"] != today:
        db.execute(
            "UPDATE users SET daily_usage = 0, usage_reset_on = ? WHERE id = ?",
            (today, user["id"]),
        )
        user["daily_usage"] = 0
    if user["plan"] == "free" and user["daily_usage"] >= DAILY_FREE_LIMIT:
        abort(402, description="free tier daily limit reached, upgrade required")

def increment_usage(user):
    db = get_db()
    db.execute("UPDATE users SET daily_usage = daily_usage + 1 WHERE id = ?", (user["id"],))
    db.commit()

@app.post("/v1/analyze")
@require_token
def metered_analyze():
    user = g.current_user
    enforce_quota(user)
    increment_usage(user)
    return jsonify({"status": "ok", "remaining_free_calls": max(0, DAILY_FREE_LIMIT - user["daily_usage"]) if user["plan"] == "free" else None})

@app.get("/v1/usage")
@require_token
def usage():
    user = g.current_user
    return jsonify(
        {
            "plan": user["plan"],
            "daily_usage": user["daily_usage"],
            "daily_limit": DAILY_FREE_LIMIT if user["plan"] == "free" else None,
            "subscription_status": user["subscription_status"],
        }
    )

@app.post("/stripe/webhook")
def stripe_webhook():
    payload = request.data
    sig = request.headers.get("Stripe-Signature")
    try:
        event = stripe.Webhook.construct_event(payload, sig, STRIPE_WEBHOOK_SECRET)
    except Exception as exc:
        return jsonify({"error": str(exc)}), 400

    if event["type"] in {"customer.subscription.created", "customer.subscription.updated"}:
        sub = event["data"]["object"]
        update_plan_for_customer(sub["customer"], sub["status"])
    elif event["type"] == "customer.subscription.deleted":
        sub = event["data"]["object"]
        update_plan_for_customer(sub["customer"], "canceled")

    return "", 200

def update_plan_for_customer(customer_id: str, status: str):
    plan = "paid" if status in {"active", "trialing"} else "free"
    db = sqlite3.connect(DATABASE)
    db.execute(
        """UPDATE users
           SET plan = ?, subscription_status = ?, daily_usage = CASE WHEN ? = 'free' THEN daily_usage ELSE daily_usage END
           WHERE stripe_customer_id = ?""",
        (plan, status, plan, customer_id),
    )
    db.commit()
    db.close()

if __name__ == "__main__":
    init_db()
    app.run(port=5001, debug=True)
