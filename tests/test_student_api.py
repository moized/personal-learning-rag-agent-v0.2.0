from fastapi.testclient import TestClient
from app.main import app
from app.models import Job, Source
from app.services.overview import source_status
from tests.helpers import add_course


def status(job_status, stage="ingest", current=0, total=0, retries=0, error=None):
    src = Source(id=1, name="Week 1.pdf", source_type="file", status="processing")
    job = Job(source_id=1, task="process_source", status=job_status, stage=stage,
              progress_current=current, progress_total=total, retry_count=retries, error=error)
    return source_status(src, job, (12, 2))


def test_status_speaks_plain_language():
    assert status("queued")["step"] == "Waiting in line"
    working = status("running", "organize", 20, 80)
    assert (working["state"], working["step"], working["progress"]) == ("working", "Sorting it into topics", 0.25)
    assert status("running", "dedupe")["progress"] is None
    assert status("queued", retries=1)["step"] == "Hit a problem, trying again (attempt 2 of 5)"
    ready = status("done")
    assert (ready["state"], ready["summary"]) == ("ready", "12 passages, 2 repeated passages merged")
    failed = status("failed", error="Transcript unavailable for abc: no captions\nTraceback ...")
    assert (failed["state"], failed["error"]) == ("failed", "Transcript unavailable for abc: no captions")


def test_failed_material_can_be_retried(db):
    client = TestClient(app)
    course = add_course(db)
    src = Source(name="Lecture", source_type="youtube", status="failed", course_id=course.id, error="boom")
    db.add(src); db.flush()
    job = Job(source_id=src.id, task="process_source", status="failed", stage="transcript", retry_count=5, error="boom")
    db.add(job); db.commit()

    assert client.get(f"/courses/{course.id}/sources").json()[0]["state"] == "failed"
    assert client.post(f"/sources/{src.id}/retry").json()["status"] == "queued"
    db.expire_all()
    assert (job.status, job.retry_count, job.error, src.status) == ("queued", 0, None, "queued")
    assert client.post(f"/sources/{src.id}/retry").status_code == 409
    assert client.post("/sources/999/retry").status_code == 404


def test_the_student_page_is_served_at_the_root():
    client = TestClient(app)
    page = client.get("/")
    assert page.status_code == 200 and "<main" in page.text
    assert client.get("/app.js").status_code == 200
    # API routes and the interactive docs still win over the page.
    assert client.get("/health").json()["status"] == "ok"
    assert client.get("/docs").status_code == 200
