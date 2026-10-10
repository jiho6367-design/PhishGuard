# PhishGuard 무료 온라인 모드

기존 저장소 루트의 로컬 앱과 별개인 실제 분석 모드입니다. 사용자가 입력한 메일을 승인된 `2026-10-10.rules.v2` Python 규칙으로 매번 계산합니다. 고정 예제의 점수를 재생하지 않고, 유료 AI API·ChatGPT 구독 인증·Gmail·HF 모델·DB·서버 이력/설정을 사용하지 않습니다. 점수는 **규칙 위험 지수**이고 피싱 확률·정확도·안전 보장이 아닙니다.

## 데이터 흐름과 범위

- 브라우저가 제목·본문·발신자·인증 여부·첨부명·추가 URL을 같은 앱의 `POST /api/analyze`로 전송합니다. 비밀키나 개인정보가 포함된 메일은 붙여 넣지 마세요.
- 서버 메모리에서 계산한 뒤 원본 제목·본문·발신자·전체 URL을 응답하지 않습니다. 근거에는 **URL 호스트·발신자 도메인·첨부명·특정 표현**이 포함될 수 있습니다.
- 입력 URL은 문자열로 파싱할 뿐 방문·DNS 조회·외부 API 호출을 하지 않습니다. 이 성질을 바꾸려면 SSRF 방어와 별도 검토가 필요합니다.
- 공개 모드의 API는 상태 확인과 분석뿐입니다. `/api/history`, `/api/settings`, `/api/gmail`, `/api/batch` 등 공유 상태·Gmail 경로는 제공하지 않습니다. 기존 로컬 앱의 파일을 import하지 않습니다.
- 서버에 메일·설정·이력을 저장하지 않습니다. 브라우저 이력·설정의 보관 동의, 삭제, 기간·개수 제한, 저장 실패 처리는 이 모드의 UI에서 제공합니다. 같은 브라우저 프로필을 공유하는 공용 PC는 같은 저장 공간을 공유하며 브라우저 저장은 암호화·계정 격리가 아닙니다.
- 앱 코드가 본문이나 분석 오류의 원문을 로그에 남기지 않도록 했습니다. 호스팅 사업자의 요청/접속/운영 로그·보관 정책은 별도이며, 앱의 `persistent_storage:false`가 플랫폼 로그까지 없다는 의미는 아닙니다. 분석 입력을 쿼리스트링에 넣지 마세요.

## 로컬 실행

이 폴더에서 Python 3.13으로 실행합니다. 아래 `.venv`는 로컬 개발 전용이고 게시·함수 번들에서 제외합니다.

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe app.py
```

화면은 `http://127.0.0.1:5000` 또는 `http://localhost:5000`입니다. 기본 실행은 loopback에만 바인딩하고 디버그를 끕니다. 다른 로컬 포트의 검증에서는 `create_app(["http://127.0.0.1:해당포트"], vercel_runtime=False)`로 정확한 출처를 지정합니다. `app.run`은 로컬 개발 실행이고 온라인 운영에서는 Vercel이 WSGI `app`을 실행합니다.

## API 계약과 경계

```json
{
  "item": {
    "title": "합성 메일",
    "body": "확인해 주세요: https://portal.example.com",
    "sender": "sample@example.com",
    "auth_results": {"spf_pass": null, "dkim_pass": null, "dmarc_pass": null},
    "attachments": ["invoice.zip"],
    "links": []
  },
  "threshold": 20
}
```

UTF-8 `application/json` 요청의 전체 크기는 64 KiB 이하, 제목 300자·본문 16,000자·발신자 500자·첨부명 20개(각 260자)·추가 URL 배열 50개 이하여야 합니다. **제목·본문에서 추출한 URL과 추가 URL을 정규화·중복 제외한 전체 URL도 합계 50개 이하, 각 URL도 2,048자 이하**여야 합니다. 단서를 조용히 잘라내지 않고 초과 시 400 오류를 반환합니다. 임계값은 5~95의 5 단위 정수이며 생략 시 20입니다. 인증 결과는 `true`/`false`/`null`이고, 모르는 결과는 실패로 계산하지 않습니다.

추가 키·Gmail ID·원문 HTML 모드·사용자 `parser_warnings`, 중복 JSON 키·NaN/Infinity·잘못된 UTF-8·타입 오류를 거부합니다. 성공 응답은 `risk_score`(0~1), `label`(`phishing`/`normal`), `threshold`(백분율 정수), `signals`/`reasons`/`warnings`/`feedback`, 규칙 버전, 요청 소요 시간을 포함합니다. `normal`은 임계값 미만이라는 뜻입니다. 오류는 `error` 코드와 고정 한국어 `detail`을 반환하며 입력 원문·예외 원문을 반영하지 않습니다.

URL의 2,048자 제한은 브라우저 입력과 동일한 UTF-16 길이 단위로 계산하므로 일부 이모지·비 BMP 문자는 2칸을 차지합니다(`health.limits.url_length_unit: utf16_code_units`). 서버의 제목·본문·발신자·첨부명 한도는 Python 유니코드 문자 수이며 브라우저 입력 제한은 UTF-16 기준이어서 일부 문자에서 UI가 더 일찍 제한할 수 있습니다. 전체 근거는 `reasons`/`signals`에 유지하고 피드백은 대표 근거 2개만 요약하여 표시합니다. 근거를 삭제하거나 규칙 점수를 바꾸는 요약이 아닙니다.

Unicode 호스트명의 `casefold` 정규화는 일부 문자를 최대 3개로 확장합니다(예: `ﬃ` → `ffi`). 따라서 파생 근거는 입력 URL보다 길어질 수 있습니다. UI의 응답 검증은 개별 근거·신호 설명 6,400 UTF-16 단위, 요약 피드백 20,000 단위를 허용하며 원문 URL의 2,048 단위 제한은 유지합니다. 두 개의 서로 다른 최대 길이 확장 호스트와 `İ` 호스트를 실제 API 요청으로 검증합니다.

모든 API 응답은 `Cache-Control`, CDN cache control을 `no-store`로 지정하고 CORS 허용 헤더·쿠키를 추가하지 않습니다. 브라우저가 보낸 Origin은 허용된 **현재 Host의 출처와 정확히 같아야** 합니다. 다른 허용 별칭도 교차 출처 요청으로 허용하지 않습니다. Host 확인은 인증이 아니며, 다른 HTTP 클라이언트는 Origin을 꾸밀 수 있습니다.

## 요청 제한의 실제 한계

기본 제한은 **현재 Python 프로세스마다 분석 요청 30개/60초, 동시에 2개**입니다. 같은 출처에서 보낸 잘못된 분석 요청도 횟수에 포함합니다. 초과 시 429와 `Retry-After`를 반환합니다. 메모리 사용은 30개 타임스탬프로 제한하고 잠금으로 동시 요청을 관리합니다. IP/X-Forwarded-For를 수집·저장하지 않으므로 헤더를 바꾸어 같은 프로세스 제한을 피할 수 없습니다.

그러나 서버리스 인스턴스 추가·재시작·콜드 스타트마다 카운터가 별개입니다. **지속되거나 전 세계적으로 보장되는 사용자별 할당량·비용 상한·DDoS 방어가 아닙니다.** 테스트는 첫 인스턴스의 31번째 요청이 429인데 새 인스턴스에서는 200인 반례도 확인합니다. 공개 배포 전 Hobby 사용량/정지 조건, 플랫폼 Firewall/Rate limiting, 실제 Vercel 동작과 비용/사용량 모니터링을 검토해야 합니다. 이 구현은 새 외부 저장소나 API 키를 생성하지 않습니다.

## 승인 후 Vercel 설정

이 문서는 로컬 후보의 실행 방법입니다. GitHub 게시·로그인·프로젝트 생성·운영 설정 변경·배포는 아직 실행하지 않았습니다. 승인한 정확한 변경 목록과 SHA에 대해서만 진행합니다.

| 설정 | 값 |
|---|---|
| Repository | 기존 PhishGuard 저장소 |
| Root Directory | `online` |
| Framework Preset | `Flask` (`vercel.json`의 `framework: "flask"`) |
| Python | `.python-version`의 `3.13` |
| Build Command | Override ON, 빈 값 |
| Install Command | 이전 Override 해제, Python framework 기본 설치 (`requirements.txt`) |
| Output Directory | 이전 Override 해제, Flask 기본값. 정적 제안의 `.` 값을 사용하지 않음 |
| Function | `app.py`의 `app`, `maxDuration: 15` |
| Static files | `public/**`에서 Vercel CDN 제공 |

`vercel.json`은 빌드·프레임워크·보안 헤더와 함수 설정을 포함합니다. Root Directory는 원격 프로젝트 설정으로 별도로 지정해야 합니다. `.vercelignore` 및 함수 `excludeFiles`로 테스트·로컬 환경을 배포 대상에서 제외합니다. 기존 저장소 루트 앱·다른 프로젝트를 변경할 필요가 없습니다. 실제 Vercel 빌드/미리보기/TLS/헤더 처리는 로그인 뒤 최종 검증이 필요하며 로컬 테스트로 증명하지 않았습니다.

**정확한 앱 출처:** Vercel에서 `Enable access to System Environment Variables`가 활성일 때 `VERCEL=1`과 `VERCEL_URL`, `VERCEL_BRANCH_URL`, `VERCEL_PROJECT_PRODUCTION_URL`의 정확한 도메인만 허용합니다. 시스템 변수를 노출할 수 없다면 `PHISH_PUBLIC_ORIGINS`에 최종 HTTPS 출처 목록을 쉼표로 명시하고 Vercel 플랫폼 여부도 실제 런타임에서 확인해야 합니다. 추가 custom domain 별칭은 자동 포함되지 않으므로 정확히 추가해야 합니다. wildcard `*.vercel.app`나 요청의 `X-Forwarded-Host`로 허용 목록을 만들지 않습니다. 미설정 도메인은 403으로 실패합니다. Deployment Protection을 해제해서 이 검사를 우회하지 마세요.

Vercel이 TLS를 종료한 뒤 WSGI에 전달하는 공식 `X-Forwarded-Proto` 계약은 **`VERCEL=1` 플랫폼 모드에서만** 사용합니다. 일반 로컬/다른 WSGI 서버는 해당 헤더를 무시합니다. 다른 서버에서 `VERCEL=1`을 임의로 설정하는 것은 프록시 신뢰 경계를 바꾸므로 하지 마세요. 플랫폼 헤더의 처리·허용 출처와 실제 방문 도메인은 배포 뒤 별도로 확인해야 합니다.

Hobby의 무료 제공은 개인·비상업 목적 및 사용량 제한에 따릅니다. 무료 무제한 운영을 보장하지 않고 유료 플랜·추가 과금·새 비밀키 발급은 별도 승인 사항입니다. ChatGPT 구독은 이 제품의 런타임 API 크레딧이 아닙니다.

복구하려면 원격에서 기록해 둔 이전 Root/Framework/Build/Install/Output/환경 변수 설정을 복원하고 이전 승인 SHA를 배포해야 합니다. 새 `online/` 소스를 철회한다면 별도 revert 후 **소스와 원격 설정을 함께** 확인합니다. 소스만 되돌리면 Root Directory가 없는 상태가 될 수 있습니다.

## 검증 및 공식 근거

테스트 의존성은 런타임 requirements에 포함하지 않습니다. 별도 개발 환경에서 `pytest online/tests`를 실행합니다. 합성 데이터로 strict input, 95개 임계값 조합, URL/DNS/DB 호출 차단, Origin/Host/프록시, 캐시, 예외 비공개, 반복·동시 요청·재시작 제한을 검증합니다. 별도 브라우저 QA와 독립 검토 결과는 작업 보고서에 기록합니다. 제품의 피싱 탐지 정확도를 측정한 결과는 아닙니다.

- [Vercel Flask 진입점, public CDN, 단일 함수](https://vercel.com/docs/frameworks/backend/flask)
- [Python 지원 버전, 의존성, 전체 파일 번들 범위](https://vercel.com/docs/functions/runtimes/python)
- [vercel.json framework, functions, headers](https://vercel.com/docs/project-configuration/vercel-json)
- [Vercel 시스템 환경 변수와 노출 조건](https://vercel.com/docs/environment-variables/system-environment-variables)
- [Host 및 X-Forwarded-Proto 공식 요청 헤더](https://vercel.com/docs/headers/request-headers)
- [Hobby 플랜](https://vercel.com/docs/plans/hobby)

## 브라우저 저장과 취소의 정확한 범위

저장은 기본 OFF입니다. 처음 ON으로 적용하면 이후 분석의 요약만 저장하며, 이전에 탭 메모리에 있던 이력을 복사하지 않습니다. 저장 요약은 제목·발신자·시간·점수·임계값·정해진 신호 종류뿐입니다. 본문·별도 링크 목록·첨부명·원문 분석 근거는 저장하지 않습니다. 제목과 발신자에서는 알려진 URL 형식을 제거하지만 임의로 숨긴 주소나 민감 정보를 모두 찾아낸다는 보장은 없습니다. 제목과 발신자 자체도 민감할 수 있습니다.

IndexedDB에 최대 50개, 7일 이내의 요약을 보관합니다. 만료 자료는 앱 열기·조회·쓰기 때 정리하며 브라우저가 닫혀 있는 동안 자동 삭제되지는 않습니다. 별도 프로필/출처의 저장소는 분리되지만 같은 프로필의 탭은 공유합니다. 로그인 사용자별 격리나 암호화 저장이 아니므로 공용 PC에서는 저장하지 않는 편이 좋습니다.

각 저장 트랜잭션은 최신 동의와 revision을 다시 확인합니다. 이미 ON이었던 탭의 늦은 분석 결과·오래된 설정은 다른 탭의 삭제 뒤 다시 저장되지 않습니다. BroadcastChannel을 지원하면 삭제 알림으로 OFF 상태의 오래된 설정 초안도 재확인하도록 막습니다. 지원하지 않는 환경에서는 포커스·화면 재활성화·뒤로 복귀·명시적 새로고침 때 동기화합니다. 영구 삭제 epoch/tombstone은 남기지 않으므로 **알림 없는 OFF → 삭제 → OFF는 세대 구분이 불가능**합니다. 그런 탭에서 사용자가 새로 저장 적용을 누르면 빈 이력으로 동의를 다시 켤 수 있으나 과거 메모리 이력을 복원하지는 않습니다.

취소하면 늦은 응답과 아직 완료되지 않은 브라우저 저장을 AbortSignal로 차단합니다. 이미 서버로 전송된 요청이나 이미 commit된 요약까지 소급 취소하는 것은 아닙니다. 저장이 완료된 자료는 개별/전체 삭제를 사용하세요. 전체 삭제는 이력·설정·저장 동의와 결과 표시를 지웁니다. 작성 중인 입력 칸은 별도의 **입력 비우기**로 정리하세요. 저장/삭제가 실패하면 기존 자료가 남아 있을 수 있음을 알리고 성공했다고 표시하지 않습니다.

분석 fetch는 `same-origin` 쿠키만 전달하며 앱 자체는 인증 쿠키를 만들지 않습니다. 이는 Vercel Deployment Protection의 접근 쿠키를 보존하기 위한 동작입니다. 실제 보호된 배포의 로그인·쿠키·TLS 검증은 운영 설정 승인 이후 별도로 확인해야 합니다.
