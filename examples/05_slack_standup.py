# Workflow: Slack Standup Summary
# Reads recent Slack messages from a channel, summarizes the standup updates.

# --- Fetch recent messages
messages = slack_get_channel_messages(channel=channel_id, limit=50)

# --- Analyze standup updates
class PersonUpdate(BaseModel):
    person: str
    yesterday: str
    today: str
    blockers: str

class StandupSummary(BaseModel):
    updates: list[PersonUpdate]
    team_blockers: list[str]
    key_progress: list[str]
    overall_status: str

summary = llm(
    "Parse these Slack messages as a team standup. Extract each person's update "
    "(what they did yesterday, what they're doing today, any blockers). "
    "Identify team-wide blockers and key progress.",
    data=messages,
    format=StandupSummary,
)

# --- Post summary back to Slack
lines = [f"*Standup Summary* ({len(summary.updates)} updates)\n"]
lines.append(f"Status: {summary.overall_status}\n")

for u in summary.updates:
    lines.append(f"*{u.person}*")
    lines.append(f"  Yesterday: {u.yesterday}")
    lines.append(f"  Today: {u.today}")
    if u.blockers and u.blockers.lower() != "none":
        lines.append(f"  :warning: Blocker: {u.blockers}")
    lines.append("")

if summary.team_blockers:
    lines.append("*Team Blockers:*")
    for b in summary.team_blockers:
        lines.append(f"  :red_circle: {b}")

lines.append("\n*Key Progress:*")
for p in summary.key_progress:
    lines.append(f"  :white_check_mark: {p}")

slack_send_message(channel=channel_id, text="\n".join(lines))
