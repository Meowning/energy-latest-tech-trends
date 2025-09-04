#!/usr/bin/env bash
set -euo pipefail

mkdir -p mail/{mail-data,mail-state,mail-logs,config}
[ -f mail/.env ] || cp mail/.env.example mail/.env

docker compose up -d mailserver

docker compose exec -T mailserver setup email add support@energy.meowning.kr '1q2w3e4r!'
docker compose exec -T mailserver setup email add no-reply@energy.meowning.kr '1q2w3e4r!'
docker compose exec -T mailserver setup alias add contact@energy.meowning.kr support@energy.meowning.kr
docker compose exec -T mailserver setup alias add info@meowning.kr            support@energy.meowning.kr
docker compose exec -T mailserver setup alias add postmaster@energy.meowning.kr support@energy.meowning.kr

docker compose exec -T mailserver setup config dkim

echo "(1) mail/config/opendkim/keys/energy.meowning.kr/mail.txt (DKIM 내용) 열어서 내용 복사"
echo "(2) mail._domainkey.energy.meowning.kr TXT로 등록"
