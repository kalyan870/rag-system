from pathlib import Path
from typing import BinaryIO

from fastapi import HTTPException


def parse_allowed_extensions(value: str) -> set[str]:
    return {item.strip().lower() for item in value.split(",") if item.strip()}


def save_upload(source: BinaryIO, destination: str, max_bytes: int) -> None:
    total_bytes = 0
    with open(destination, "wb") as output:
        while chunk := source.read(min(1024 * 1024, max_bytes + 1 - total_bytes)):
            total_bytes += len(chunk)
            if total_bytes > max_bytes:
                raise HTTPException(413, "File exceeds the upload size limit.")
            output.write(chunk)


def supported_extension(filename: str, allowed_extensions: str) -> str:
    extension = Path(filename).suffix.lower()
    if extension not in parse_allowed_extensions(allowed_extensions):
        raise HTTPException(400, f"Unsupported file type: {extension}. Allowed: {allowed_extensions}")
    return extension
