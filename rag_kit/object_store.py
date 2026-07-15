import hashlib
from pathlib import Path
from typing import Dict, Iterable

from minio import Minio
from minio.error import S3Error


class ObjectStore:
    def __init__(
        self,
        endpoint: str,
        access_key: str,
        secret_key: str,
        bucket: str,
        secure: bool = False,
    ) -> None:
        self.bucket = bucket
        self.client = Minio(endpoint, access_key=access_key, secret_key=secret_key, secure=secure)

    def ensure_bucket(self) -> None:
        if not self.client.bucket_exists(self.bucket):
            self.client.make_bucket(self.bucket)

    def upload_sources(self, sources: Iterable[str]) -> Dict[str, str]:
        self.ensure_bucket()
        object_keys: Dict[str, str] = {}
        for source in sorted(set(sources)):
            path = Path(source)
            if not path.is_file():
                continue
            object_key = self._object_key(path)
            content_type = _content_type(path)
            self.client.fput_object(self.bucket, object_key, str(path), content_type=content_type)
            object_keys[source] = object_key
        return object_keys

    def healthcheck(self) -> bool:
        try:
            self.ensure_bucket()
        except S3Error:
            return False
        return True

    def _object_key(self, path: Path) -> str:
        digest = hashlib.sha1(str(path.resolve()).encode("utf-8")).hexdigest()[:12]
        return f"documents/{digest}/{path.name}"


def _content_type(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        return "application/pdf"
    if suffix == ".docx":
        return "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
    if suffix == ".md":
        return "text/markdown"
    return "text/plain"

