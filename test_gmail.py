"""Safe configuration check; real Gmail access requires the explicit --connect flag."""
from __future__ import annotations

import argparse
import os
from pathlib import Path

from dotenv import load_dotenv


REQUIRED = ("GOOGLE_CLIENT_ID", "GOOGLE_CLIENT_SECRET", "GOOGLE_REFRESH_TOKEN")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Gmail 설정 존재 여부 또는 명시적 읽기 전용 연결을 점검합니다.")
    parser.add_argument("--connect", action="store_true", help="기존 OAuth 자격 증명으로 실제 Gmail 목록을 조회합니다.")
    args = parser.parse_args(argv)
    load_dotenv(Path(__file__).resolve().parent / ".env", override=False)
    present = {name: bool(os.getenv(name, "").strip()) for name in REQUIRED}
    for name, exists in present.items():
        print(f"{name}: {'있음' if exists else '없음'}")
    if not args.connect:
        print("외부 연결은 실행하지 않았습니다. 실제 조회는 --connect를 명시해야 합니다.")
        return 0
    if not all(present.values()):
        print("기존 Gmail 자격 증명이 부족합니다. 연결을 실행하지 않았습니다.")
        return 1
    try:
        from google.oauth2.credentials import Credentials
        from mail_service import build

        creds = Credentials(
            None,
            refresh_token=os.environ["GOOGLE_REFRESH_TOKEN"],
            token_uri="https://oauth2.googleapis.com/token",
            client_id=os.environ["GOOGLE_CLIENT_ID"],
            client_secret=os.environ["GOOGLE_CLIENT_SECRET"],
            scopes=["https://www.googleapis.com/auth/gmail.readonly"],
        )
        service = build("gmail", "v1", credentials=creds, cache_discovery=False)
        result = service.users().messages().list(userId="me", maxResults=5).execute()
        print(f"읽기 전용 조회 성공: {len(result.get('messages', []))}건. 메일 ID·본문·계정 주소는 출력하지 않습니다.")
    except Exception as exc:
        print(f"Gmail 조회를 완료하지 못했습니다 ({type(exc).__name__}). 의존성·자격 증명·연결 상태를 확인하세요.")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
