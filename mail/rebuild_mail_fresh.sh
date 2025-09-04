# rebuild_mail_fresh.sh
set -euo pipefail

HOST=${}
PASS='1q2w3e4r!'  # 데모 비번(원하면 바꿔서 실행)

echo "[1] 컨테이너 중지"
docker compose down mailserver || true

echo "[2] 설정 초기화 (키/계정/별칭만 삭제)"
rm -rf mail/config
mkdir -p mail/{mail-data,mail-state,mail-logs,config}

#  (선택) 메일박스까지 전체 초기화하려면 아래 주석 해제
# rm -rf mail/mail-data mail/mail-state mail/mail-logs
# mkdir -p mail/{mail-data,mail-state,mail-logs}

echo "[3] mail/.env: SSL_TYPE=self-signed 강제"
if grep -q '^SSL_TYPE=' mail/.env 2>/dev/null; then
  sed -i 's/^SSL_TYPE=.*/SSL_TYPE=self-signed/' mail/.env
else
  echo 'SSL_TYPE=self-signed' >> mail/.env
fi

echo "[4] 기동"
docker compose up -d mailserver
sleep 3

echo "[5] 계정/별칭 재생성"
docker compose exec -T mailserver setup email add support@$HOST "$PASS" || true
docker compose exec -T mailserver setup email add no-reply@$HOST "$PASS" || true
docker compose exec -T mailserver setup alias add contact@$HOST support@$HOST || true
docker compose exec -T mailserver setup alias add info@meowning.kr support@$HOST || true
docker compose exec -T mailserver setup alias add postmaster@$HOST support@$HOST || true

echo "[6] DKIM 2048-bit 재생성(selector: mail)"
docker compose exec -T mailserver bash -lc "
set -e
D=/tmp/docker-mailserver/opendkim/keys/$HOST
mkdir -p \"\$D\" && cd \"\$D\"
rm -f mail.private mail.txt
opendkim-genkey -b 2048 -d $HOST -s mail
chmod 600 mail.private
chown opendkim:opendkim mail.private 2>/dev/null || true
echo '[DKIM] mail.txt 아래 한 줄값:'; grep -o 'v=DKIM1;.*' mail.txt
"

echo "[7] 가비아 TXT 값(250자 단위로 자동 줄바꿈)"
V=$(awk 'BEGIN{RS="\"";ORS=""} NR%2==0{printf "%s",$0}' mail/config/opendkim/keys/$HOST/mail.txt)
echo "$V" | fold -w 250 | sed 's/^/\"/; s/$/\"/'

cat <<'TIP'

[가비아 DNS 입력]
- 타입: TXT
- 호스트: mail._domainkey.energy
- 값: 위에 출력된 따옴표 줄 2~3줄 그대로 붙여넣기(레코드 1개에 값 여러 줄)
- MX/SPF/DMARC도 확인:
  A      : energy → 122.199.76.193
  MX     : energy → energy.meowning.kr (prio 10)
  SPF TXT: energy → v=spf1 a mx ip4:122.199.76.193 ~all
  DMARC  : _dmarc.energy → v=DMARC1; p=quarantine; rua=mailto:support@energy.meowning.kr; ruf=mailto:support@energy.meowning.kr; fo=1; adkim=s; aspf=s

[확인]
dig +short TXT mail._domainkey.energy.meowning.kr
docker compose exec -T mailserver opendkim-testkey -d energy.meowning.kr -s mail -vvv
TIP
