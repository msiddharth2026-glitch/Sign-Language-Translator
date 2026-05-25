import sqlite3
import bcrypt
import os

DB_PATH = os.path.join(os.path.dirname(__file__), "users.db")

def init_db():
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            email TEXT UNIQUE NOT NULL,
            password TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.commit()
    conn.close()

def register_user(username: str, email: str, password: str) -> tuple[bool, str]:
    hashed = bcrypt.hashpw(password.encode(), bcrypt.gensalt()).decode()
    try:
        conn = sqlite3.connect(DB_PATH)
        conn.execute(
            "INSERT INTO users (username, email, password) VALUES (?, ?, ?)",
            (username.strip(), email.strip().lower(), hashed)
        )
        conn.commit()
        conn.close()
        return True, "Registration successful."
    except sqlite3.IntegrityError as e:
        if "username" in str(e):
            return False, "Username already exists."
        if "email" in str(e):
            return False, "Email already registered."
        return False, "Registration failed."

def login_user(identifier: str, password: str) -> tuple[bool, str, dict]:
    conn = sqlite3.connect(DB_PATH)
    row = conn.execute(
        "SELECT id, username, email, password FROM users WHERE username=? OR email=?",
        (identifier.strip(), identifier.strip().lower())
    ).fetchone()
    conn.close()

    if not row:
        return False, "User not found.", {}
    _, username, email, hashed = row
    if bcrypt.checkpw(password.encode(), hashed.encode()):
        return True, "Login successful.", {"username": username, "email": email}
    return False, "Incorrect password.", {}

def reset_password(email: str, new_password: str) -> tuple[bool, str]:
    hashed = bcrypt.hashpw(new_password.encode(), bcrypt.gensalt()).decode()
    conn = sqlite3.connect(DB_PATH)
    cur = conn.execute(
        "UPDATE users SET password=? WHERE email=?",
        (hashed, email.strip().lower())
    )
    conn.commit()
    conn.close()
    if cur.rowcount == 0:
        return False, "Email not found."
    return True, "Password reset successful."
