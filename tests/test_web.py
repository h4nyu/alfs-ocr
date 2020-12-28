from alfs_char.web import app
from alfs_char.store import ImageRepository


def test_detect() -> None:
    repo = ImageRepository()
    rows = repo.filter()
    if len(rows) == 0:
        return
    row = repo.find(rows[0]["id"])
    with app.test_client() as client:
        res = client.post("/api/upload-image", json=dict(data=row["data"]))
        assert res.status_code == 200
