"""MAYA Gmail — read and send emails via Google OAuth2."""
import base64
import email as _email
import json
from email.mime.text import MIMEText
from pathlib import Path

CREDS_FILE  = Path(__file__).parent / "models/google_credentials.json"
TOKEN_FILE  = Path(__file__).parent / "models/gmail_tokens.json"
SCOPES      = [
    "https://www.googleapis.com/auth/gmail.readonly",
    "https://www.googleapis.com/auth/gmail.send",
]

_service = None


def _build_service():
    global _service
    try:
        from google.oauth2.credentials      import Credentials
        from google.auth.transport.requests import Request
        from googleapiclient.discovery      import build

        if not TOKEN_FILE.exists():
            return None

        creds = Credentials.from_authorized_user_file(str(TOKEN_FILE), SCOPES)
        if creds and creds.expired and creds.refresh_token:
            try:
                creds.refresh(Request())
                TOKEN_FILE.write_text(creds.to_json(), encoding="utf-8")
            except Exception as e:
                print(f"[gmail] token refresh failed: {e}", flush=True)
                return None

        if creds and creds.valid:
            _service = build("gmail", "v1", credentials=creds)
            return _service
    except Exception as e:
        print(f"[gmail] build_service error: {e}", flush=True)
    return None


def _get_service():
    global _service
    if _service is not None:
        return _service
    return _build_service()


def auth_url() -> str:
    """Return a Google OAuth2 URL for the user to visit. Blocking."""
    from google_auth_oauthlib.flow import Flow
    if not CREDS_FILE.exists():
        raise FileNotFoundError("google_credentials.json not found in Backend/models/")
    flow = Flow.from_client_secrets_file(
        str(CREDS_FILE),
        scopes=SCOPES,
        redirect_uri="urn:ietf:wg:oauth:2.0:oob",
    )
    url, _ = flow.authorization_url(prompt="consent")
    # Store flow state for the callback
    _flow_cache["flow"] = flow
    return url


_flow_cache: dict = {}


def handle_callback(code: str) -> None:
    """Exchange the auth code for a token and save it. Blocking."""
    global _service
    flow = _flow_cache.pop("flow", None)
    if flow is None:
        # Build a fresh flow (e.g. if server restarted between auth_url and callback)
        from google_auth_oauthlib.flow import Flow
        if not CREDS_FILE.exists():
            raise FileNotFoundError("google_credentials.json not found in Backend/models/")
        flow = Flow.from_client_secrets_file(
            str(CREDS_FILE),
            scopes=SCOPES,
            redirect_uri="urn:ietf:wg:oauth:2.0:oob",
        )
    flow.fetch_token(code=code)
    creds = flow.credentials
    TOKEN_FILE.write_text(creds.to_json(), encoding="utf-8")
    from googleapiclient.discovery import build
    _service = build("gmail", "v1", credentials=creds)
    print("[gmail] authenticated and token saved", flush=True)


def fetch_emails(count: int = 5, unread_only: bool = True) -> list[dict]:
    """Return a list of email summaries. Blocking — call from executor."""
    svc = _get_service()
    if not svc:
        raise RuntimeError("Gmail not authenticated. Ask MAYA to authenticate Gmail first.")

    query = "is:unread" if unread_only else ""
    results = (
        svc.users().messages()
        .list(userId="me", q=query, maxResults=count)
        .execute()
    )
    messages = results.get("messages", [])
    if not messages:
        return []

    emails = []
    for m in messages:
        msg = svc.users().messages().get(
            userId="me", id=m["id"], format="metadata",
            metadataHeaders=["From", "Subject", "Date"],
        ).execute()

        headers = {h["name"]: h["value"] for h in msg.get("payload", {}).get("headers", [])}
        snippet = msg.get("snippet", "")[:300]
        emails.append({
            "id":      m["id"],
            "from":    headers.get("From", "Unknown"),
            "subject": headers.get("Subject", "(no subject)"),
            "date":    headers.get("Date", ""),
            "snippet": snippet,
        })
    return emails


def send_email(to: str, subject: str, body: str) -> None:
    """Send an email. Blocking — call from executor."""
    svc = _get_service()
    if not svc:
        raise RuntimeError("Gmail not authenticated. Ask MAYA to authenticate Gmail first.")

    message = MIMEText(body)
    message["to"]      = to
    message["subject"] = subject
    raw = base64.urlsafe_b64encode(message.as_bytes()).decode()
    svc.users().messages().send(userId="me", body={"raw": raw}).execute()
    print(f"[gmail] sent to {to}: {subject}", flush=True)


# Try to load service on module import
try:
    _build_service()
    if _service:
        print("[gmail] service loaded from saved token", flush=True)
except Exception:
    pass
