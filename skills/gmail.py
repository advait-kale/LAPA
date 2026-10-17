import json
from pathlib import Path

from google_auth_oauthlib.flow import InstalledAppFlow
from google.oauth2.credentials import Credentials
from simplegmail import Gmail

project_root = Path(__file__).resolve().parents[1]
token_path = project_root / "token.json"
credentials_path = project_root / "credentials.json"
scopes = ["https://www.googleapis.com/auth/gmail.send"]

credentials = None
if token_path.exists():
    with token_path.open(encoding="utf-8") as f:
        token_info = json.load(f)
    credentials = Credentials.from_authorized_user_info(token_info)

if not credentials or not credentials.has_scopes(scopes):
    flow = InstalledAppFlow.from_client_secrets_file(str(credentials_path), scopes)
    credentials = flow.run_local_server(port=0, prompt="consent")
    token_path.write_text(credentials.to_json(), encoding="utf-8")



# message = gmail.send_message(
#     sender="advaitkale9@gmail.com",
#     to="advaitkale9@gmail.com",
#     subject="Hello",
#     msg_plain="Hello from simplegmail.",
#     msg_html="<p>Hello from <strong>simplegmail</strong>.</p>",
# )

print("Email sent.")
class Mail:
    def __init__(self):
        self.gmail = Gmail(credentials=credentials)

    def send_mail(self, sender : str, to : str, subject : str, body : str):
        self.gmail.send_message(
            sender=sender,
            to=to,
            subject=subject,
            msg_plain=body,
            msg_html=f"<p>{body}</p>",
        )
