"""페이지 텍스트를 검색 가능한 작은 단위로 나누는 기능."""

from __future__ import annotations


def chunk_text(text: str, max_chars: int = 1600, overlap_chars: int = 200) -> list[str]:
    """줄 경계를 우선 보존하며 텍스트를 겹치는 조각으로 나눈다."""
    if max_chars <= 0:
        raise ValueError("max_chars는 1 이상이어야 합니다.")
    if overlap_chars < 0 or overlap_chars >= max_chars:
        raise ValueError("overlap_chars는 0 이상 max_chars 미만이어야 합니다.")

    lines = [line.rstrip() for line in text.splitlines() if line.strip()]
    if not lines:
        return []

    chunks: list[str] = []
    current: list[str] = []

    def length(parts: list[str]) -> int:
        return sum(len(part) for part in parts) + max(0, len(parts) - 1)

    def flush() -> None:
        nonlocal current
        if not current:
            return
        chunks.append("\n".join(current).strip())
        if overlap_chars == 0:
            current = []
            return
        overlap: list[str] = []
        for line in reversed(current):
            candidate = [line, *overlap]
            if overlap and length(candidate) > overlap_chars:
                break
            overlap = candidate
            if length(overlap) >= overlap_chars:
                break
        current = overlap

    for line in lines:
        if len(line) > max_chars:
            flush()
            step = max_chars - overlap_chars
            for start in range(0, len(line), step):
                piece = line[start : start + max_chars]
                if piece:
                    chunks.append(piece)
                if start + max_chars >= len(line):
                    break
            current = []
            continue

        candidate = [*current, line]
        if current and length(candidate) > max_chars:
            flush()
            candidate = [*current, line]
            current = candidate if length(candidate) <= max_chars else [line]
        else:
            current = candidate

    flush()
    return chunks


def chunk_page_text(
    text: str,
    max_chars: int = 1600,
    overlap_chars: int = 200,
    context_lines: int = 5,
) -> list[str]:
    """페이지 상단 문맥을 각 chunk에 반복해 표 제목과 헤더를 보존한다."""
    lines = [line.rstrip() for line in text.splitlines() if line.strip()]
    if not lines:
        return []

    context = "\n".join(lines[:context_lines]).strip()
    body = "\n".join(lines[context_lines:]).strip()
    if not body:
        return [context[:max_chars]]

    available = max_chars - len(context) - 1
    if available <= 1:
        return chunk_text(text, max_chars=max_chars, overlap_chars=overlap_chars)

    body_overlap = min(overlap_chars, available - 1)
    body_chunks = chunk_text(
        body,
        max_chars=available,
        overlap_chars=body_overlap,
    )
    return [f"{context}\n{chunk}".strip() for chunk in body_chunks]
