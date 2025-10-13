#!/usr/bin/env bash
# 최신기술동향 메일서버 전용 커스텀 파이프라인입니다. 메일서버 시작 전에 실행되므로 건들지 말아주세요
set -euo pipefail

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BOLD='\033[1m'
RESET='\033[0m'
log(){ echo "[$(date +%F_%T)] $*"; }

# ===== DMS 관련 환경변수 =====
: "${DOMAIN:-energy-latest-tech-trends.com}"  # 메일 주소의 @뒤 도메인 (예: meowning.kr 또는 energy.meowning.kr)
MAIL_FQDN="${HOSTNAME_FQDN:-${DOMAIN}}"       # 메일서버 FQDN (예: mail.meowning.kr 또는 energy.meowning.kr)

DMS_SETUP="/usr/local/bin/setup"
CFG_DIR="/tmp/docker-mailserver"
ACCOUNTS_CF="${CFG_DIR}/postfix-accounts.cf"
VIRTUAL_CF="${CFG_DIR}/postfix-virtual.cf"
DKIM_DIR="${CFG_DIR}/opendkim/keys/${MAIL_FQDN}"
SSL_DIR="${CFG_DIR}/ssl"

# ===== 계정 정보, 별칭 =====
# ⚠️ 유저/별칭은 서버 FQDN이 아니라 "메일 주소 도메인"을 사용해야 함
SUPPORT_USER="support@${DOMAIN}"
NOREPLY_USER="no-reply@${DOMAIN}"
SUPPORT_PASS="${SUPPORT_PASS:-password}"
NOREPLY_PASS="${NOREPLY_PASS:-password}"

ALIAS_CONTACT="contact@${DOMAIN}"
ALIAS_POSTMASTER="postmaster@${DOMAIN}"
ALIAS_INFO_EXT="${ALIAS_INFO_EXT:-info@${DOMAIN}}"
ALLOW_INFO_ALIAS="${ALLOW_INFO_ALIAS:-1}"

# ===== 디렉토리 및 파일 보장 =====
mkdir -p "${CFG_DIR}" "${SSL_DIR}"
touch "${ACCOUNTS_CF}" "${VIRTUAL_CF}"    # 계정/별칭 파일이 없으면 생성

# ===== helper 함수 =====
has_user() {
  # postfix-accounts.cf 파일 내 포맷 = user@domain|password (or hash)
  grep -Eiq "^[[:space:]]*$1[[:space:]]*\|" "${ACCOUNTS_CF}"
}
has_alias() {
  # postfix-virtual.cf 파일 내 포맷 = source destination
  # 공백 정규화 한다음 비교
  awk '{print $1" "$2}' "${VIRTUAL_CF}" | sed 's/[[:space:]]\+/ /g' | grep -Fxq "$1 $2"
}
safe_add_user() {
  local user="$1" pass="$2"
  if has_user "$user"; then
    log "[SKIP] 유저 존재: $user"
  else
    log "[ADD] 유저 추가: $user"
    ${DMS_SETUP} email add "$user" "$pass" || true
  fi
}
safe_add_alias() {
  local src="$1" dst="$2"
  if has_alias "$src" "$dst"; then
    log "[SKIP] 별칭 존재: $src -> $dst"
  else
    log "[ADD] 별칭 추가: $src -> $dst"
    ${DMS_SETUP} alias add "$src" "$dst" || true
  fi
}

# 여기부터 파이프라인 시작입니다.
# ===== (1) 계정, 별칭 추가 =====
safe_add_user  "${SUPPORT_USER}" "${SUPPORT_PASS}"
safe_add_user  "${NOREPLY_USER}" "${NOREPLY_PASS}"
safe_add_alias "${ALIAS_CONTACT}"    "${SUPPORT_USER}"
safe_add_alias "${ALIAS_POSTMASTER}" "${SUPPORT_USER}"
[ "${ALLOW_INFO_ALIAS}" = "1" ] && safe_add_alias "${ALIAS_INFO_EXT}" "${SUPPORT_USER}"

# ===== (2) DKIM 키 없으면 생성 (서버 FQDN 기준) =====
if [ ! -f "${DKIM_DIR}/mail.private" ]; then
  log "[INFO] DKIM 생성 (setup 사용): ${MAIL_FQDN}"
  ${DMS_SETUP} config dkim
  printf '\n'
  printf '%b\n' "${YELLOW}${BOLD}==================================================${RESET}"
  printf '%b\n' "${YELLOW}${BOLD} DKIM DNS 레코드를 등록해주세요:${RESET}"
  printf '\n'
  printf '%b\n' "${GREEN} 1. 호스트:${RESET}"
  printf '%b\n' "    mail._domainkey.${DOMAIN}"
  printf '\n'
  printf '%b\n' "${GREEN} 2. 값(TXT):${RESET}"
  DKIM_VALUE=$(grep -o 'p=.*' "${DKIM_DIR}/mail.txt" | tr -d '"' )
  printf '%b\n' "    \"${DKIM_VALUE}\""
  printf '\n'
  printf '%b\n' "${YELLOW}${BOLD} 등록 후 'docker compose up -d' 로 다시 실행하세요.${RESET}"
  printf '%b\n' "${YELLOW}${BOLD}==================================================${RESET}"
  exit 78
else
  log "[SKIP] DKIM 이미 존재: ${DKIM_DIR}/mail.private"
fi

# ===== (3) self-signed 인증서 (옵션) =====
if [ "${USE_LETSENCRYPT:-0}" = "1" ]; then
  SSL_TYPE="letsencrypt"
fi

if [ "${SSL_TYPE:-self-signed}" = "self-signed" ]; then
  KEY_FILE="${SSL_DIR}/${DOMAIN}-key.pem"
  CERT_FILE="${SSL_DIR}/${DOMAIN}-cert.pem"
  CA_DIR="${SSL_DIR}/demoCA"
  CA_CERT="${CA_DIR}/cacert.pem"

  mkdir -p "${CA_DIR}"

  if [ ! -f "${KEY_FILE}" ] || [ ! -f "${CERT_FILE}" ]; then
    log "[INFO] self-signed 인증서 생성: ${DOMAIN}"
    openssl req -x509 -newkey rsa:2048 -nodes \
      -keyout "${KEY_FILE}" \
      -out    "${CERT_FILE}" \
      -days 365 \
      -subj "/CN=${MAIL_FQDN}" >/dev/null 2>&1
  else
    log "[SKIP] self-signed 인증서 이미 존재"
  fi

  if [ ! -f "${CA_CERT}" ]; then
    log "[INFO] CA cert 생성 -> ${CA_CERT}"
    cp -f "${CERT_FILE}" "${CA_CERT}"
  else
    log "[SKIP] CA cert 이미 존재 -> ${CA_CERT}"
  fi
fi

log "[INFO] 사용자 설정 스크립트 완료! 메일서버 가동을 시작합니다."
