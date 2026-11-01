from typing import Literal
from platform import node
import os
from pathlib import Path
from IPython.display import Image, display
from langchain.tools import tool
from pydantic import BaseModel, Field
from langgraph.graph import StateGraph, START, END
from typing import TypedDict, Optional, Set, List
from langchain.chat_models import init_chat_model
from langchain_groq import ChatGroq
from typing_extensions import Annotated, List
import operator
from langchain.messages import HumanMessage, SystemMessage

# Load the Groq key from the project-root .env without adding a runtime dependency.
if "__file__" in globals():
    project_root = Path(__file__).resolve().parents[2]
else:
    # Jupyter does not define __file__; locate the project from the notebook's cwd.
    project_root = Path.cwd().resolve()
    for candidate in (project_root, *project_root.parents):
        if (candidate / ".env").is_file():
            project_root = candidate
            break

env_path = project_root / ".env"
if env_path.exists():
    for line in env_path.read_text(encoding="utf-8").splitlines():
        key, separator, value = line.partition("=")
        if separator and key.strip().lower() == "groq_api_key":
            os.environ.setdefault("GROQ_API_KEY", value.strip().strip("\"'"))
            break

groq_api_key = os.getenv("GROQ_API_KEY")
if not groq_api_key:
    raise RuntimeError(
        "Groq API key not found. Add GROQ_API_KEY=<your key> to the project-root .env file."
    )

llm = ChatGroq(
    model="openai/gpt-oss-120b",
    api_key=groq_api_key,
    temperature=0.7,
)

# sender
# to
# subject
# msg_plain
# msg_html

#state
class User_Input(BaseModel):
    query : str = Field(description="The user's query")

#state
class IntentResult(BaseModel):
    intent: Literal["email", "calendar", "general"] = Field(
        description="The primary intent of the user's request"
    )

#node
def intent(query: str) -> str:
    system_message = """
    You are the intent classifier for LAPA, a personal AI assistant.

    Classify the user's request into exactly one category:

    - email: Reading, searching, drafting, sending, replying to,
      or managing emails.
    - calendar: Creating, scheduling, viewing, updating, or
      cancelling calendar events, meetings, or appointments.
    - general: Questions, explanations, coding help, advice,
      and requests unrelated to email or calendar operations.

    Identify the primary intent, not merely keywords.
    If the user requests both email and calendar operations,
    choose the primary action. If neither takes priority, choose calendar.
    Questions about email or calendars as concepts are general.

    Do not execute actions. Only classify the request.
    """

    result = llm.with_structured_output(
        IntentResult
    ).invoke([
        SystemMessage(content=system_message),
        HumanMessage(content=query),
    ])

    return result.intent

#state
class EmailDraft(BaseModel):
    sender: str = Field(
        description="Authenticated sender email address"
    )
    to: str = Field(
        description="Recipient email address"
    )
    subject: str = Field(
        description="Email subject"
    )
    msg_plain: str = Field(
        description="Plain-text email body"
    )
    msg_html: str = Field(
        description="HTML version of the same email body"
    )

class mail_dict(TypedDict):
    sender: str
    to: str
    subject: str
    msg_plain: str

def email_details(query: User_Input, state: mail_dict) -> EmailDraft:
    system_message = """
        Fields:
        - sender: Use the authenticated sender provided in the state.
        - to: Extract the recipient email address if provided.
        - subject: Extract the requested subject, or create a concise
          subject if the user requests an email but provides none.
        - msg_plain: Extract or draft the plain-text email body.
        - msg_html: Create an HTML version of the same email body.

        Rules:
        1. Preserve the user's intended meaning.
        2. Never invent email addresses.
        3. Never guess the sender's identity.
        4. If the recipient is unknown, leave it empty.
        5. Do not send the email.
    """

    result = llm.with_structured_output(
        EmailDraft
    ).invoke([
        SystemMessage(content=system_message),
        HumanMessage(content=query.query),
    ])
    
    return {
        "sender": state["sender"],
        "to": result.to,
        "subject": result.subject,
        "msg_plain": result.msg_plain
    }

chain = StateGraph(User_Input)
chain.add_node("intent", intent)
chain.add_node("email_details", email_details)
chain.add_edge(START, "intent")
chain.add_conditional_edges("intent", intent,{"email" : "email_details"})
chain.add_edge("email_details", END)

chain = chain.compile()
# import matplotlib.pyplot as plt
# plt.figure(figsize=(20, 20))    
# display.imshow(Image(chain.get_graph().draw_mermaid_png()))

chain.invoke({"query" : "Write a mail to arvind@gmail.com to remind him about tommorow's meeting from me my mail is advaitkale9@gmail.com"})j