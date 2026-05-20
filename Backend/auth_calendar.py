"""Authenticate Maya with Google Calendar accounts.

Usage:
  python auth_calendar.py          # authenticate account 1
  python auth_calendar.py 2        # authenticate account 2
"""
import sys
from pathlib import Path

CREDS_FILE   = Path(__file__).parent / "models/google_credentials.json"
TOKENS_FILES = [
    Path(__file__).parent / "models/google_tokens_1.json",
    Path(__file__).parent / "models/google_tokens_2.json",
]
SCOPES = ["https://www.googleapis.com/auth/calendar"]


def auth(account: int) -> None:
    idx = account - 1
    if idx < 0 or idx >= len(TOKENS_FILES):
        print(f"ERROR: account must be 1 or 2")
        return

    if not CREDS_FILE.exists():
        print(f"ERROR: {CREDS_FILE} not found.")
        return

    from google_auth_oauthlib.flow import InstalledAppFlow
    from googleapiclient.discovery import build

    print(f"Opening browser for account {account}...")
    flow  = InstalledAppFlow.from_client_secrets_file(str(CREDS_FILE), SCOPES)
    creds = flow.run_local_server(port=0, open_browser=True)

    TOKENS_FILES[idx].write_text(creds.to_json(), encoding="utf-8")
    print(f"Token saved to {TOKENS_FILES[idx]}")

    service = build("calendar", "v3", credentials=creds)
    result  = service.calendarList().list().execute()
    cals    = result.get("items", [])
    print(f"\nAccount {account} connected! Found {len(cals)} calendar(s):")
    for c in cals:
        print(f"  - {c.get('summary', c['id'])}")


if __name__ == "__main__":
    account = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    auth(account)
