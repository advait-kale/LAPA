from typing import Literal, TypedDict
import os
from pathlib import Path
from pydantic import BaseModel, Field
from langgraph.graph import StateGraph, START, END
from langchain_groq import ChatGroq
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


class AssistantState(TypedDict, total=False):
    """Shared state passed between orchestrator graph nodes."""

    query: str
    sender: str
    intent: Literal["email", "calendar", "general"]
    to: str
    subject: str
    msg_plain: str
    msg_html: str
    response: str


class User_Input(BaseModel):
    query: str = Field(description="The user's query")

class IntentResult(BaseModel):
    intent: Literal["email", "calendar", "general"] = Field(
        description="The primary intent of the user's request"
    )

def classify_intent(query: str) -> Literal["email", "calendar", "general"]:
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

def email_details(state: AssistantState) -> AssistantState:
    system_message = """
        Fields:
        - sender: Use the authenticated sender provided in the state. If it is
          missing, leave it empty; never infer it from the request.
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

    sender = state.get("sender", "")
    result = llm.with_structured_output(
        EmailDraft
    ).invoke([
        SystemMessage(content=system_message),
        HumanMessage(content=f"Authenticated sender: {sender}\n\nRequest: {state['query']}"),
    ])

    return {
        "sender": sender,
        "to": result.to,
        "subject": result.subject,
        "msg_plain": result.msg_plain,
        "msg_html": result.msg_html,
    }


def route_intent(state: AssistantState) -> Literal["email", "calendar", "general"]:
    """Choose the next node from the intent stored by the classifier."""
    return state["intent"]


def calendar_request(state: AssistantState) -> AssistantState:
    """Placeholder for the calendar integration, which is not wired yet."""
    return {
        "response": (
            "I recognized this as a calendar request, but calendar actions "
            "are not connected yet."
        )
    }


def general_response(state: AssistantState) -> AssistantState:
    result = llm.invoke([HumanMessage(content=state["query"])])
    return {"response": str(result.content)}


def build_graph():
    workflow = StateGraph(AssistantState)
    workflow.add_node(
        "classify_intent",
        lambda state: {"intent": classify_intent(state["query"])},
    )
    workflow.add_node("email_details", email_details)
    workflow.add_node("calendar_request", calendar_request)
    workflow.add_node("general_response", general_response)

    workflow.add_edge(START, "classify_intent")
    workflow.add_conditional_edges(
        "classify_intent",
        route_intent,
        {
            "email": "email_details",
            "calendar": "calendar_request",
            "general": "general_response",
        },
    )
    workflow.add_edge("email_details", END)
    workflow.add_edge("calendar_request", END)
    workflow.add_edge("general_response", END)
    return workflow.compile()


chain = build_graph()
