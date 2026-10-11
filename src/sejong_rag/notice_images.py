"""저장된 공지 본문 이미지를 수집하고 로컬 OCR 결과를 검토용으로 보존한다."""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path
from urllib.parse import urlparse

import requests

from sejong_rag.ingest import USER_AGENT, parse_notice_html
from sejong_rag.recover_attachments import parse_tesseract_tsv, run_command


def validate_image_url(url: str) -> None:
    parsed = urlparse(url)
    if (parsed.scheme != "https" or parsed.hostname != "www.sejong.ac.kr"
            or parsed.port not in (None, 443) or parsed.username or parsed.password):
        raise ValueError("공식 HTTPS 본문 이미지 주소만 허용합니다.")


def download_image(url: str, path: Path) -> None:
    validate_image_url(url)
    with requests.get(url, headers={"User-Agent": USER_AGENT}, timeout=60,
                      stream=True, allow_redirects=False) as response:
        response.raise_for_status()
        if response.status_code != 200:
            raise ValueError("본문 이미지 이동 응답은 자동으로 따라가지 않습니다.")
        if not response.headers.get("Content-Type", "").lower().startswith("image/"):
            raise ValueError("이미지 형식의 응답이 아닙니다.")
        payload = bytearray()
        for block in response.iter_content(65536):
            payload.extend(block)
            if len(payload) > 20 * 1024 * 1024:
                raise ValueError("본문 이미지가 20MiB 제한을 초과했습니다.")
        if not payload:
            raise ValueError("본문 이미지가 비어 있습니다.")
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(".part")
        try:
            temporary.write_bytes(payload)
            temporary.replace(path)
        finally:
            temporary.unlink(missing_ok=True)


def ocr_image(path: Path) -> tuple[str, float | None]:
    # 작은 화면 캡처는 문단으로 인식되지 않을 수 있어 흩어진 글자 모드도 시도한다.
    for mode in ("3", "11"):
        result = run_command(["tesseract", str(path), "stdout", "-l", "kor+eng", "--psm", mode, "tsv"])
        text, confidence = parse_tesseract_tsv(result.stdout)
        if text.strip():
            return text, confidence
    return "", None


def process_images(data_dir: Path, *, delay: float = 10, dry_run: bool = False, refresh: bool = False, offline: bool = False,
                   downloader=download_image, reader=ocr_image) -> dict:
    if delay < 10:
        raise ValueError("실제 요청 간격은 10초 이상이어야 합니다.")
    if offline and refresh:
        raise ValueError("offline과 refresh는 함께 사용할 수 없습니다.")
    output = data_dir / "processed/notice-images"
    summary = {"notices": 0, "images": 0, "processed": 0, "failed": 0}
    last_request = None
    for detail_path in sorted((data_dir / "metadata/details").glob("*.json")):
        detail = json.loads(detail_path.read_text(encoding="utf-8"))
        document_id = detail["document_id"]
        raw = data_dir / "raw/notices" / f"{document_id}.html"
        images = detail.get("body_images", [])
        if raw.is_file():
            images = parse_notice_html(raw.read_text(encoding="utf-8"), detail["source_url"]).body_images
        if not images:
            continue
        summary["notices"] += 1
        summary["images"] += len(images)
        if dry_run:
            continue
        report_path = output / f"{document_id}.json"
        previous = json.loads(report_path.read_text(encoding="utf-8")) if report_path.exists() else {}
        cached = {item["image_url"]: item for item in previous.get("images", [])}
        records = []
        for number, image in enumerate(images, 1):
            url = image["url"]
            item = {"image_url": url, "alt": image.get("alt", ""), "image": number,
                    "status": "failed", "text": ""}
            try:
                validate_image_url(url)
                path = data_dir / "raw/notice-images" / (hashlib.sha256(url.encode()).hexdigest() + ".img")
                if offline and not path.is_file():
                    raise FileNotFoundError("확보된 이미지 원본이 없어 OCR하지 않았습니다.")
                if refresh or not path.exists():
                    if last_request is not None:
                        time.sleep(max(0, delay - (time.monotonic() - last_request)))
                    last_request = time.monotonic()
                    downloader(url, path)
                digest = hashlib.sha256(path.read_bytes()).hexdigest()
                old = cached.get(url, {})
                if old.get("content_hash") == digest and old.get("status") == "review_required":
                    item = {**old, "image": number}
                else:
                    text, confidence = reader(path)
                    if not text.strip():
                        raise ValueError("이미지에서 글을 추출하지 못했습니다.")
                    item.update(text=text, mean_confidence=confidence, content_hash=digest,
                                raw_path=str(path), status="review_required")
                    summary["processed"] += 1
            except Exception as error:
                item["error"] = f"{type(error).__name__}: {error}"
                summary["failed"] += 1
            records.append(item)
        output.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps({"document_id": document_id,
            "source_url": detail["source_url"], "images": records}, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def main():
    parser = argparse.ArgumentParser(description="공지 본문 이미지 수집 및 검토용 OCR")
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--delay", type=float, default=10)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--refresh", action="store_true", help="같은 주소의 이미지 교체 여부도 재확인")
    parser.add_argument("--offline", action="store_true", help="저장된 이미지 원본만 OCR하며 네트워크 요청하지 않음")
    args = parser.parse_args()
    print(json.dumps(process_images(args.data_dir, delay=args.delay, dry_run=args.dry_run, refresh=args.refresh, offline=args.offline), ensure_ascii=False))


if __name__ == "__main__":
    main()
