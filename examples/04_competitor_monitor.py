# Workflow: Competitor Monitor
# Searches for news about competitors, analyzes sentiment, emails a report.

# --- Search for competitor news
news = search_news(q=competitor_name, num=10)

# --- Analyze coverage
class NewsItem(BaseModel):
    headline: str
    source: str
    sentiment: str
    key_takeaway: str

class CompetitorReport(BaseModel):
    items: list[NewsItem]
    overall_sentiment: str
    strategic_implications: list[str]
    recommended_actions: list[str]

analysis = llm(
    f"Analyze recent news coverage about {competitor_name}. "
    "Assess sentiment for each item and identify strategic implications for us.",
    data=news,
    format=CompetitorReport,
)

# --- Build and send email report
lines = [f"Competitor Intelligence: {competitor_name}\n"]
lines.append(f"Overall Sentiment: {analysis.overall_sentiment}\n")

lines.append("Recent Coverage:")
for item in analysis.items:
    lines.append(f"  [{item.sentiment.upper()}] {item.headline} ({item.source})")
    lines.append(f"    -> {item.key_takeaway}")

lines.append(f"\nStrategic Implications:")
for imp in analysis.strategic_implications:
    lines.append(f"  - {imp}")

lines.append(f"\nRecommended Actions:")
for action in analysis.recommended_actions:
    lines.append(f"  - {action}")

report_text = "\n".join(lines)

gmail_send_message(
    to=recipient_email,
    subject=f"Competitor Report: {competitor_name}",
    body=report_text,
)
