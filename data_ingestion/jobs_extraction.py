import requests
import re
from bs4 import BeautifulSoup

RAW_URL = "https://raw.githubusercontent.com/vanshb03/New-Grad-2026/main/README.md"

def extract_jobs_from_table():
    response = requests.get(RAW_URL)
    response.raise_for_status()
    md = response.text

    pattern = re.compile(
        r"\| Company \| Role \| Location \| Application/Link \| Date Posted \|\s*\|[-| :]+\|\s*((?:\|.*\|\s*)+)",
        re.MULTILINE
    )

    match = pattern.search(md)
    if not match:
        print("No job table found.")
        return []

    table_block = match.group(1)

    jobs = []

    for line in table_block.strip().split("\n"):
        line = line.strip()
        if not line.startswith("|"):
            continue

        # Split columns
        cols = [c.strip() for c in line.split("|")[1:-1]]  # ignore left & right empty parts
        if len(cols) < 5:
            continue

        company, role, location, apply_html, date_posted = cols[:5]

        # Extract URL from HTML <a> tag
        soup = BeautifulSoup(apply_html, "html.parser")
        link_tag = soup.find("a")
        apply_link = link_tag["href"] if link_tag else ""

        jobs.append({
            "company": company,
            "role": role,
            "location": location,
            "apply_link": apply_link,
            "date_posted": date_posted
        })

    return jobs



