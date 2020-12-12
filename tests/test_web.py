from alfs_char.web import detect, UploadPayload
from alfs_char.store import ImageRepository


def test_detect() -> None:
    repo = ImageRepository()
    rows = repo.filter()
    if len(rows) == 0:
        return
    row = repo.find(rows[0]["id"])
    payload = UploadPayload(data=row["data"])
    detect(payload)
