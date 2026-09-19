from __future__ import annotations

from pathlib import Path

from crag_engine import process_file_path


def process_and_store_file(file_path: str) -> int:
    """Read, chunk, embed, and store a local PDF/TXT file."""
    chunk_count = process_file_path(file_path)
    print(f"✅ Indexed '{Path(file_path).name}' into {chunk_count} chunk(s).")
    return chunk_count


if __name__ == "__main__":
    sample = Path(__file__).resolve().parent / "sample.txt"
    if not sample.exists():
        sample.write_text(
            "Project Beta uses PostgreSQL database and React frontend framework.",
            encoding="utf-8",
        )

    process_and_store_file(str(sample))
