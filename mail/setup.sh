#!/usr/bin/env bash
set -euo pipefail

# 디렉토리 생성
mkdir -p mail/{mail-data,mail-state,mail-logs,config}

# 샘플 env 확인
[ -f mail/mailserver.env ] || { echo "mail/mailserver.env 를 .env.example로부터 생성하세요"; exit 1; }

# ---- 컨테이너 기동 ----
docker compose up -d mailserver

# ---- 계정 생성 (support: 수/발신, no-reply: 발신 전용) ----
docker compose exec -T mailserver setup email add support@energy.meowning.kr '1q2w3e4r!'
docker compose exec -T mailserver setup email add no-reply@energy.meowning.kr '1q2w3e4r!'

# ---- 별칭 → support 포워드 ----
docker compose exec -T mailserver setup alias add contact@energy.meowning.kr support@energy.meowning.kr
docker compose exec -T mailserver setup alias add info@meowning.kr            support@energy.meowning.kr

# ---- DKIM 키 생성 (selector: mail) ----
docker compose exec -T mailserver setup config dkim

echo
echo "[다음 단계]"
echo "1) DKIM 공개키 파일: mail/config/opendkim/keys/energy.meowning.kr/mail.txt"
echo "2) DNS: mail._domainkey.energy.meowning.kr TXT에 mail.txt 전체 붙여넣기"
echo "3) Cloudflare는 메일 레코드 모두 DNS only(회색 구름)"
