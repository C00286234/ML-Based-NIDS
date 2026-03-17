"""
Deliberately Vulnerable Web Application
=========================================
A minimal Flask app with intentionally vulnerable endpoints for testing
the NIDS against real web attacks from a Kali VM.

Author: C00286234

WARNING: This application is INTENTIONALLY INSECURE. It is designed to be
attacked. Never expose it to any network beyond your isolated test lab.

Endpoints:
    GET  /               — Homepage with links
    GET  /search?q=...   — Reflects input (XSS vulnerable)
    GET  /login          — Login form
    POST /login          — Login handler (brute-force target)
    GET  /users?id=...   — SQL injection vulnerable query
    GET  /health         — Health check for DoS monitoring

Usage:
    python testing/vulnerable_webapp.py                # Default port 8080
    python testing/vulnerable_webapp.py --port 9090    # Custom port

Attacks to test against this app:
    - DoS Slowloris:       slowloris <HOST_IP> -p 8080
    - DoS GoldenEye:       goldeneye http://<HOST_IP>:8080
    - DoS Hulk:            python hulk.py http://<HOST_IP>:8080
    - DDoS (SYN flood):    hping3 -S --flood -p 8080 <HOST_IP>
    - Brute Force:         hydra -l admin -P wordlist http-post-form://...
    - SQL Injection:       sqlmap -u "http://<HOST_IP>:8080/users?id=1" --batch
    - XSS:                 xsser -u "http://<HOST_IP>:8080/search?q=test"
"""

import argparse
import os
import sqlite3
import sys
import time
from pathlib import Path

try:
    from flask import Flask, request, render_template_string
except ImportError:
    print("Flask not installed. Run: pip install flask")
    sys.exit(1)

app = Flask(__name__)

# Simple in-memory request counter for monitoring
REQUEST_COUNT = 0
START_TIME = time.time()

# ==============================================================================
# DATABASE SETUP (deliberately vulnerable)
# ==============================================================================

DB_PATH = Path(__file__).parent / "test_vuln.db"


def init_db():
    """Create a test SQLite database with dummy user data."""
    conn = sqlite3.connect(str(DB_PATH))
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY,
            username TEXT NOT NULL,
            email TEXT NOT NULL,
            role TEXT DEFAULT 'user'
        )
    """)
    # Insert test data if empty
    cursor.execute("SELECT COUNT(*) FROM users")
    if cursor.fetchone()[0] == 0:
        test_users = [
            (1, "admin", "admin@test.local", "admin"),
            (2, "john", "john@test.local", "user"),
            (3, "jane", "jane@test.local", "user"),
            (4, "bob", "bob@test.local", "moderator"),
            (5, "alice", "alice@test.local", "user"),
        ]
        cursor.executemany("INSERT INTO users VALUES (?, ?, ?, ?)", test_users)
    conn.commit()
    conn.close()


# Hardcoded credentials (intentionally weak for brute-force testing)
VALID_USERS = {
    "admin": "password123",
    "user": "letmein",
    "test": "test",
}


# ==============================================================================
# HTML TEMPLATES
# ==============================================================================

HOME_HTML = """
<!DOCTYPE html>
<html>
<head><title>Vulnerable Test App</title></head>
<body>
    <h1>NIDS Test Target Application</h1>
    <p><strong>WARNING:</strong> This app is intentionally vulnerable for NIDS testing.</p>
    <ul>
        <li><a href="/search?q=test">Search</a> (XSS vulnerable)</li>
        <li><a href="/login">Login</a> (Brute-force target)</li>
        <li><a href="/users?id=1">Users</a> (SQL injection vulnerable)</li>
        <li><a href="/health">Health</a></li>
    </ul>
    <p>Request count: {{ count }}</p>
</body>
</html>
"""

SEARCH_HTML = """
<!DOCTYPE html>
<html>
<head><title>Search Results</title></head>
<body>
    <h1>Search Results</h1>
    <form action="/search" method="GET">
        <input type="text" name="q" value="{{ query }}" size="40">
        <button type="submit">Search</button>
    </form>
    <p>You searched for: {{ query | safe }}</p>
    <p>No results found for your query.</p>
    <p><a href="/">Back</a></p>
</body>
</html>
"""

LOGIN_HTML = """
<!DOCTYPE html>
<html>
<head><title>Login</title></head>
<body>
    <h1>Login</h1>
    {% if error %}
    <p style="color: red;">{{ error }}</p>
    {% endif %}
    {% if success %}
    <p style="color: green;">{{ success }}</p>
    {% endif %}
    <form action="/login" method="POST">
        <label>Username: <input type="text" name="username"></label><br><br>
        <label>Password: <input type="password" name="password"></label><br><br>
        <button type="submit">Login</button>
    </form>
    <p><a href="/">Back</a></p>
</body>
</html>
"""

USERS_HTML = """
<!DOCTYPE html>
<html>
<head><title>User Profile</title></head>
<body>
    <h1>User Lookup</h1>
    <form action="/users" method="GET">
        <label>User ID: <input type="text" name="id" value="{{ user_id }}"></label>
        <button type="submit">Lookup</button>
    </form>
    {% if error %}
    <p style="color: red;">{{ error }}</p>
    {% elif user %}
    <table border="1" cellpadding="8">
        <tr><th>ID</th><th>Username</th><th>Email</th><th>Role</th></tr>
        {% for row in user %}
        <tr><td>{{ row[0] }}</td><td>{{ row[1] }}</td><td>{{ row[2] }}</td><td>{{ row[3] }}</td></tr>
        {% endfor %}
    </table>
    {% else %}
    <p>Enter a user ID to look up.</p>
    {% endif %}
    <p><a href="/">Back</a></p>
</body>
</html>
"""


# ==============================================================================
# ROUTES
# ==============================================================================

@app.before_request
def count_requests():
    global REQUEST_COUNT
    REQUEST_COUNT += 1


@app.route("/")
def home():
    return render_template_string(HOME_HTML, count=REQUEST_COUNT)


@app.route("/search")
def search():
    """XSS vulnerable: reflects user input without sanitization."""
    query = request.args.get("q", "")
    # INTENTIONALLY VULNERABLE: using |safe in template renders raw HTML
    return render_template_string(SEARCH_HTML, query=query)


@app.route("/login", methods=["GET", "POST"])
def login():
    """Brute-force target: simple username/password with no rate limiting."""
    if request.method == "POST":
        username = request.form.get("username", "")
        password = request.form.get("password", "")

        # INTENTIONALLY VULNERABLE: no rate limiting, no lockout
        if username in VALID_USERS and VALID_USERS[username] == password:
            return render_template_string(LOGIN_HTML,
                                          success=f"Welcome, {username}!",
                                          error=None)
        else:
            return render_template_string(LOGIN_HTML,
                                          error="Invalid credentials",
                                          success=None), 401

    return render_template_string(LOGIN_HTML, error=None, success=None)


@app.route("/users")
def users():
    """SQL injection vulnerable: string concatenation in query."""
    user_id = request.args.get("id", "")

    if not user_id:
        return render_template_string(USERS_HTML, user_id="", user=None, error=None)

    try:
        conn = sqlite3.connect(str(DB_PATH))
        cursor = conn.cursor()

        # INTENTIONALLY VULNERABLE: raw string concatenation (SQL injection)
        query = f"SELECT * FROM users WHERE id = {user_id}"
        cursor.execute(query)
        results = cursor.fetchall()
        conn.close()

        if results:
            return render_template_string(USERS_HTML,
                                          user_id=user_id, user=results, error=None)
        else:
            return render_template_string(USERS_HTML,
                                          user_id=user_id, user=None,
                                          error="User not found")

    except Exception as e:
        return render_template_string(USERS_HTML,
                                      user_id=user_id, user=None,
                                      error=f"Database error: {e}")


@app.route("/health")
def health():
    """Health check endpoint."""
    uptime = time.time() - START_TIME
    return {
        "status": "running",
        "uptime_seconds": round(uptime, 1),
        "total_requests": REQUEST_COUNT,
        "requests_per_second": round(REQUEST_COUNT / max(uptime, 1), 2),
    }


# ==============================================================================
# MAIN
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Deliberately vulnerable web app for NIDS testing"
    )
    parser.add_argument("--port", type=int, default=8080,
                        help="Port to run on (default: 8080)")
    parser.add_argument("--host", default="0.0.0.0",
                        help="Host to bind to (default: 0.0.0.0)")
    args = parser.parse_args()

    # Initialize database
    init_db()
    print(f"Database initialized at {DB_PATH}")

    print(f"\n{'='*60}")
    print(f"  VULNERABLE WEB APP FOR NIDS TESTING")
    print(f"  WARNING: Intentionally insecure - lab use only!")
    print(f"{'='*60}")
    print(f"  Running on http://{args.host}:{args.port}")
    print(f"  Endpoints:")
    print(f"    GET  /search?q=...   (XSS)")
    print(f"    POST /login          (Brute force)")
    print(f"    GET  /users?id=...   (SQL injection)")
    print(f"    GET  /health         (Status)")
    print(f"{'='*60}\n")

    app.run(host=args.host, port=args.port, debug=False, threaded=True)


if __name__ == "__main__":
    main()
