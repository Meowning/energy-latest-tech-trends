#!/usr/bin/env bash
set -euo pipefail

echo "== 포트 (25/587/993) =="
nc -vz energy.meowning.kr 25  || true
nc -vz energy.meowning.kr 587 || true
nc -vz energy.meowning.kr 993 || true

echo
echo "== STARTTLS(587) 인증서(자체서명 경고 정상) =="
openssl s_client -starttls smtp -connect energy.meowning.kr:587 -servername energy.meowning.kr </dev/null 2>/dev/null \
  | openssl x509 -noout -subject -dates || true

echo
echo "== IMAPS(993) 인증서(자체서명 경고 정상) =="
openssl s_client -connect energy.meowning.kr:993 -servername energy.meowning.kr </dev/null 2>/dev/null \
  | openssl x509 -noout -subject -dates || true

echo
echo "== 발신 테스트 (no-reply → support) =="
swaks --to support@energy.meowning.kr \
      --from no-reply@energy.meowning.kr \
      --server energy.meowning.kr \
      --port 587 --tls --auth LOGIN \
      --auth-user no-reply@energy.meowning.kr \
      --auth-password '1q2w3e4r!' \
      --header "Subject: self-signed test $(date +%F_%T)" \
      --body "OK"

echo
echo "== 로그 팔로우 =="
echo "docker compose logs -f mailserver | egrep -i 'postfix|dovecot|opendkim|fail2ban'"
