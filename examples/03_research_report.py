# Workflow: Research Report
# Searches the web for a topic, synthesizes findings, and saves to Google Sheets.

# --- Search for information
results = search_web(q=topic, num=10)

# --- Analyze and synthesize
class ResearchFinding(BaseModel):
    source: str
    key_insight: str
    relevance: str

class ResearchReport(BaseModel):
    findings: list[ResearchFinding]
    executive_summary: str
    key_themes: list[str]
    gaps: list[str]

report = llm(
    f"Analyze these search results about '{topic}'. "
    "Extract key findings from each source, identify themes, and note gaps in coverage.",
    data=results,
    format=ResearchReport,
)

# --- Save to Google Sheets
rows = [["Source", "Key Insight", "Relevance"]]
for f in report.findings:
    rows.append([f.source, f.key_insight, f.relevance])

rows.append([])
rows.append(["Executive Summary"])
rows.append([report.executive_summary])
rows.append([])
rows.append(["Key Themes"])
for theme in report.key_themes:
    rows.append([theme])

sheets_update_values(
    spreadsheetId=spreadsheet_id,
    range="Sheet1!A1",
    values=rows,
)
