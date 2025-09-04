#!/usr/bin/env bash
set -euo pipefail

echo "== 포트 확인 (25/587/993) =="
nc -vz energy.meowning.kr 25  || true
nc -vz energy.meowning.kr 587 || true
nc -vz energy.meowning.kr 993 || true

echo
echo "== STARTTLS(587) 인증서 확인 =="
openssl s_client -starttls smtp -connect energy.meowning.kr:587 -servername energy.meowning.kr </dev/null 2>/dev/null \
  | openssl x509 -noout -issuer -subject -dates || true

echo
echo "== IMAPS(993) 인증서 확인 =="
openssl s_client -connect energy.meowning.kr:993 -servername energy.meowning.kr </dev/null 2>/dev/null \
  | openssl x509 -noout -issuer -subject -dates || true

echo
echo "== 발신 테스트 (no-reply → support) =="
swaks --to support@energy.meowning.kr \
      --from no-reply@energy.meowning.kr \
      --server energy.meowning.kr \
      --port 587 --tls --auth LOGIN \
      --auth-user no-reply@energy.meowning.kr \
      --auth-password '1q2w3e4r!' \
      --header "Subject: [DMS] no-reply submission $(date +%F_%T)" \
      --body "no-reply → support 테스트"

echo
echo "== 발신 테스트 (support → no-reply) =="
swaks --to no-reply@energy.meowning.kr \
      --from support@energy.meowning.kr \
      --server energy.meowning.kr \
      --port 587 --tls --auth LOGIN \
      --auth-user support@energy.meowning.kr \
      --auth-password '1q2w3e4r!' \
      --header "Subject: [DMS] support submission $(date +%F_%T)" \
      --body "support 계정 발신 테스트"

echo
echo "== 수신/전송 로그 팔로우 =="
echo "docker compose logs -f mailserver | egrep -i 'postfix|dovecot|opendkim|fail2ban'"
