# Workflow: Calendar Briefing
# Checks today's calendar and creates a briefing with prep notes.

import datetime

# --- Get today's events
today = datetime.date.today().isoformat()
events = calendar_list_events(timeMin=today + "T00:00:00Z", timeMax=today + "T23:59:59Z", maxResults=20)

# --- Generate briefing
class MeetingPrep(BaseModel):
    title: str
    time: str
    attendees: list[str]
    prep_notes: str

class DayBriefing(BaseModel):
    meeting_count: int
    free_hours: float
    meetings: list[MeetingPrep]
    day_summary: str

briefing = llm(
    "Create a daily briefing from these calendar events. "
    "For each meeting, suggest brief prep notes based on the title and attendees.",
    data=events,
    format=DayBriefing,
)

# --- Format output
lines = [f"*Daily Briefing for {today}*", f"{briefing.meeting_count} meetings, ~{briefing.free_hours}h free\n"]
lines.append(f"_{briefing.day_summary}_\n")

for m in briefing.meetings:
    lines.append(f"*{m.time} - {m.title}*")
    lines.append(f"  Attendees: {', '.join(m.attendees)}")
    lines.append(f"  Prep: {m.prep_notes}\n")

output = "\n".join(lines)
