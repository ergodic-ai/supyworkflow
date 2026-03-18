# Workflow: Email Digest
# Fetches recent emails, analyzes them, and sends a summary to Slack.

# --- Fetch recent emails
emails = gmail_list_messages(maxResults=5)

# --- Analyze emails
class EmailDigest(BaseModel):
    total_count: int
    unread_count: int
    highlights: list[str]
    action_items: list[str]
    summary: str

digest = llm(
    "Analyze these emails. Identify highlights, action items, and write a brief summary.",
    data=emails,
    format=EmailDigest,
)

# --- Format and send to Slack
lines = [f"*Email Digest* ({digest.total_count} emails, {digest.unread_count} unread)\n"]
lines.append(f"_{digest.summary}_\n")

if digest.highlights:
    lines.append("*Highlights:*")
    for h in digest.highlights:
        lines.append(f"  - {h}")

if digest.action_items:
    lines.append("\n*Action Items:*")
    for a in digest.action_items:
        lines.append(f"  - {a}")

message = "\n".join(lines)
slack_send_message(channel="general", text=message)
