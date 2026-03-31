"""
Integration tests against a live deployed instance.

Usage:
    export API_BASE_URL=https://your-app.up.railway.app
    export API_TOKEN=your_token_here
    python test_api.py
"""

import os

import requests

BASE_URL = os.environ.get("API_BASE_URL", "http://localhost:8000")
TOKEN = os.environ.get("API_TOKEN", "")
HEADERS = {
    "accept": "application/json",
    "Authorization": f"Bearer {TOKEN}",
    "Content-Type": "application/json",
}
SAMPLE_PDF = "https://www.w3.org/WAI/ER/tests/xhtml/testfiles/resources/pdf/dummy.pdf"


def check(label, condition, detail=""):
    status = "PASS" if condition else "FAIL"
    print(f"  [{status}] {label}" + (f" - {detail}" if detail else ""))
    return condition


def validate_answers(data, expected_count):
    answers = data.get("answers")
    if not check("answers is a list", isinstance(answers, list)):
        return False
    if not check("Answer count matches", len(answers) == expected_count):
        return False

    valid = True
    for index, answer in enumerate(answers, start=1):
        prefix = f"Answer {index}"
        valid &= check(f"{prefix} has question", "question" in answer)
        valid &= check(f"{prefix} has answer", "answer" in answer)
        valid &= check(f"{prefix} has status", "status" in answer)
        valid &= check(f"{prefix} has sources", "sources" in answer)
        valid &= check(f"{prefix} sources is a list", isinstance(answer.get("sources"), list))
    return valid


def test_health():
    print("\n--- Health check ---")
    response = requests.get(f"{BASE_URL}/health", timeout=10)
    check("Status 200", response.status_code == 200)
    check("Body has status=healthy", response.json().get("status") == "healthy")
    check("Version is 2.1.0", response.json().get("version") == "2.1.0")


def test_auth_rejected():
    print("\n--- Auth rejection ---")
    response = requests.post(
        f"{BASE_URL}/hackrx/run",
        headers={"Content-Type": "application/json"},
        json={"documents": SAMPLE_PDF, "questions": ["hi"]},
        timeout=10,
    )
    check("No token -> 401", response.status_code == 401)

    response = requests.post(
        f"{BASE_URL}/hackrx/run",
        headers={**HEADERS, "Authorization": "Bearer bad-token"},
        json={"documents": SAMPLE_PDF, "questions": ["hi"]},
        timeout=10,
    )
    check("Wrong token -> 401", response.status_code == 401)


def test_too_many_questions():
    print("\n--- Question limit ---")
    response = requests.post(
        f"{BASE_URL}/hackrx/run",
        headers=HEADERS,
        json={"documents": SAMPLE_PDF, "questions": ["q"] * 11},
        timeout=10,
    )
    check("11 questions -> 400", response.status_code == 400)


def test_document_caching():
    print("\n--- Document caching ---")
    payload = {"documents": SAMPLE_PDF, "questions": ["Summarise this document."]}
    first = requests.post(f"{BASE_URL}/hackrx/run", headers=HEADERS, json=payload, timeout=120)
    second = requests.post(f"{BASE_URL}/hackrx/run", headers=HEADERS, json=payload, timeout=120)

    check("First call status 200", first.status_code == 200, first.text[:120] if first.status_code != 200 else "")
    check("Second call status 200", second.status_code == 200, second.text[:120] if second.status_code != 200 else "")
    if first.status_code == 200 and second.status_code == 200:
        validate_answers(first.json(), 1)
        validate_answers(second.json(), 1)
        check("First call cache header is MISS", first.headers.get("X-Document-Cache") == "MISS")
        check("Second call cache header is HIT", second.headers.get("X-Document-Cache") == "HIT")


def test_single_question():
    print("\n--- Single question ---")
    payload = {"documents": SAMPLE_PDF, "questions": ["What is this document about?"]}
    response = requests.post(f"{BASE_URL}/hackrx/run", headers=HEADERS, json=payload, timeout=120)
    check("Status 200", response.status_code == 200, response.text[:120] if response.status_code != 200 else "")
    if response.status_code == 200:
        validate_answers(response.json(), 1)


def test_multi_question():
    print("\n--- Multi-question parallel ---")
    questions = ["What is this document about?", "Who authored this?", "What date was it created?"]
    response = requests.post(
        f"{BASE_URL}/hackrx/run",
        headers=HEADERS,
        json={"documents": SAMPLE_PDF, "questions": questions},
        timeout=180,
    )
    check("Status 200", response.status_code == 200, response.text[:120] if response.status_code != 200 else "")
    if response.status_code == 200:
        validate_answers(response.json(), len(questions))


if __name__ == "__main__":
    print(f"Testing against: {BASE_URL}\n{'=' * 50}")
    test_health()
    test_auth_rejected()
    test_too_many_questions()
    test_document_caching()
    test_single_question()
    test_multi_question()
    print(f"\n{'=' * 50}\nDone.")
