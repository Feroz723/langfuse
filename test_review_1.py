import os
import sys
import time
import base64
import httpx
from dotenv import load_dotenv

load_dotenv()

token = os.getenv("GITHUB_TOKEN")
if not token:
    print("GITHUB_TOKEN not found in .env")
    sys.exit(1)

headers = {
    "Authorization": f"Bearer {token}",
    "Accept": "application/vnd.github.v3+json",
}

repo = "Feroz723/langfuse"
print(f"Target repository: {repo}")

# 1. Get default branch SHA
res = httpx.get(f"https://api.github.com/repos/{repo}/git/ref/heads/main", headers=headers)
if res.status_code != 200:
    # Try master
    res = httpx.get(f"https://api.github.com/repos/{repo}/git/ref/heads/master", headers=headers)

if res.status_code != 200:
    print(f"Error fetching default branch ref for {repo}: {res.status_code} - {res.text}")
    sys.exit(1)

ref_data = res.json()
base_sha = ref_data["object"]["sha"]
default_branch = ref_data["ref"].split("/")[-1]
print(f"Default branch ({default_branch}) SHA: {base_sha}")

# 2. Create new branch name
branch_name = f"ai-review-test-{int(time.time())}"
res = httpx.post(
    f"https://api.github.com/repos/{repo}/git/refs",
    headers=headers,
    json={"ref": f"refs/heads/{branch_name}", "sha": base_sha},
)
if res.status_code != 201:
    print(f"Error creating branch {branch_name}: {res.status_code} - {res.text}")
    sys.exit(1)
print(f"Created branch: {branch_name}")

# 3. Create file content in branch
file_path = "ai_review_demo.py"
content = '''"""AI Code Review Demo Module."""

import os
import sys

def parse_user_query(raw_query: str):
    """Parse query with potential security vulnerabilities."""
    secret_key = "hardcoded_demo_api_secret_key_98765"
    if "exec" in raw_query:
        # Potential dangerous eval execution
        return eval(raw_query)
    return {"query": raw_query, "status": "processed"}

def calculate_analytics_metrics(data_points: list):
    """Calculate average metric values without zero check."""
    total = sum(data_points)
    average = total / len(data_points)  # Potential ZeroDivisionError
    return {"total": total, "average": average}
'''

encoded_content = base64.b64encode(content.encode("utf-8")).decode("utf-8")

res = httpx.put(
    f"https://api.github.com/repos/{repo}/contents/{file_path}",
    headers=headers,
    json={
        "message": "add ai_review_demo.py to test automated code reviewer",
        "content": encoded_content,
        "branch": branch_name,
    },
)
if res.status_code not in (200, 201):
    print(f"Error creating file {file_path}: {res.status_code} - {res.text}")
    sys.exit(1)
print(f"Created file {file_path} in branch {branch_name}")

# 4. Open Pull Request
res = httpx.post(
    f"https://api.github.com/repos/{repo}/pulls",
    headers=headers,
    json={
        "title": "Automated Code Review Demo PR",
        "head": branch_name,
        "base": default_branch,
        "body": "Test Pull Request created to verify automated AI Code Review pipeline on `Feroz723/langfuse`.",
    },
)
if res.status_code != 201:
    print(f"Error creating PR: {res.status_code} - {res.text}")
    sys.exit(1)

pr_data = res.json()
print("=" * 60)
print(f"SUCCESS! Created PR #{pr_data['number']}: {pr_data['html_url']}")
print("=" * 60)
