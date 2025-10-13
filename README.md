# 에너지 최신기술동향 알리미 📡⚡
> _여러 기관에서 발행하는 국내·외 에너지 정책 및 산업 동향 소식을 매일 빠르게 받아보세요._

## 💼 개요
**에너지 최신기술동향 알리미**는 공개된 공공기관 웹사이트에서 에너지 관련 **산업기술 최신 동향 자료**를 자동으로 수집하여, 이용자의 이메일로 발송해주는 RPA 시스템입니다!

## 📌 핵심 지원 기능
- 여러 기관의 **최신 간행물, 보도자료** 등 자동 수집
- 수집한 자료를 **제목 + 내용 + 표지 사진 + pdf 파일 + 원문 링크** 형태로 정리
- 이용자가 사전에 입력한 이메일 주소로 자동 발송 (추후 다른 경로 추가 예정)
- **OCR 및 요약 서비스**



## ⚙️ 작동 방식

1. **수집 대상 확인**  
   아래 게시된 기관 웹사이트

2. **콘텐츠 수집 및 정리**  
   - 제목, 요약, 발행일, 원문 링크 정리
   - robots.txt에 위반되지 않는 범위에서만 수집

3. **메일 발송**  
   - 템플릿 형식의 이메일로 수신자에게 전달

## ⏰ 서버 가동 정책

> 본 서비스는 **지나친 트래픽 유발 방지**와 **불필요한 자원 낭비 최소화**를 위해 아래와 같은 기준에 따라 운영됩니다.

---

### 🕘 운영 시간  
**월요일 ~ 금요일, 09:00 ~ 18:00**  

⛔️ 그 외 시간 및 주말에는 **수집 및 발송이 일시 중지**

---

### 📅 운영 기간 및 이용 안내


> _6개월간 꾸준히 소식을 받아보시고,
기간 종료 후에는 안내 메일과 함께 깔끔하게 정리됩니다._

- 한 번 이메일을 등록하면 **등록일 기준 6개월**간 최신 소식 자동 발송

- 6개월 경과 시, 등록된 이메일 정보는 DB에서 안전하게 삭제됨

- 6개월 이전에 소식 받기를 중단하고 싶은 경우, 사이트에 방문하여 취소해야 함  

- ***계속 이용을 희망하는 경우, 이메일 재등록 필요***  



## 🏢 지원 사이트

현재 알리미가 지원 중인 주요 공공기관 사이트입니다.  
(업데이트 희망 시 meowning@kumoh.ac.kr로 연락 부탁드립니다.)

| 번호 | 기관명                 | 수집 가능 여부 | 비고                              |
|------|---------------------|------------|--------------------------------|
| 1    | 한국원자력안전재단         | ✅          |                                |
| 2    | 한국원자력산업협회         | ✅          |                                |
| 3    | 한국수력원자력             | ❌          | **robots.txt 전체 차단**으로 수집 제외 |
| 4    | 한수원 중앙연구원          | ❌          | **robots.txt 전체 차단**으로 수집 제외 |
| 5    | 산업통상자원부             | ✅          |                                |
| 6    | 에너지경제연구원           | ✅          |                                |
| 7    | 한전경영연구원 (KEMRI)     | ✅          |                                |
| 8    | 산업통상자원부             | ✅          |                                |
| 9    | 한국원자력연구원 (KAERI)   | ✅          |                                |
| 10   | 원자력안전위원회           | ✅          |                                |
| 11   | 한국신재생에너지협회       | ❌          | **robots.txt 전체 차단**으로 수집 제외 |
| 12   | 한국원자력협력재단         | ❌          | **robots.txt 전체 차단**으로 수집 제외 |

> ⚠️ robots.txt로 인해 ***자동 수집 불가능한 자료는 수동 확인 필요***

## 기술 스택
### Backend
<div>
  <img src="https://img.shields.io/badge/python-3776AB?style=for-the-badge&logo=python&logoColor=white"> 
  <img src="https://img.shields.io/badge/FastAPI-009485.svg?style=for-the-badge&logo=fastapi&logoColor=white">
  <img src="https://img.shields.io/badge/mysql-4479A1.svg?style=for-the-badge&logo=mysql&logoColor=white">
</div>

### Frontend
<div>
  <img src="https://img.shields.io/badge/Svelte-%23f1413d.svg?style=for-the-badge&logo=svelte&logoColor=white">
  <img src="https://img.shields.io/badge/Vite-646CFF?style=for-the-badge&logo=vite&logoColor=fff">
  <img src="https://img.shields.io/badge/node.js-339933?style=for-the-badge&logo=Node.js&logoColor=white">
  <img src="https://img.shields.io/badge/Nginx-009639?logo=nginx&logoColor=white&style=for-the-badge">
</div>

### Tools
<div>
  <img src="https://img.shields.io/badge/Git-%23F05032.svg?style=for-the-badge&logo=git&logoColor=white">
  <img src="https://img.shields.io/badge/github-181717?style=for-the-badge&logo=github&logoColor=white">
  <img src="https://img.shields.io/badge/docker-%230db7ed.svg?style=for-the-badge&logo=docker&logoColor=white"> 
</div>


#

## 세팅 방법
### 1. 메일서버 세팅 방법
> _docker-mailserver 이미지를 베이스로 기능을 추가했습니다._

요구사항
- **포트 번호 :** 25, 587, 993 (tcp) (변경 가능)
- **구축자 소유 도메인 필요**
- 호스트 컴퓨터에 **Certbot** 설치 (Let's Encrypt 인증서 사용 시)

#

### ① .env 파일 생성
- .env.example 파일을 참고하여 작성해주세요
```
# ========================= 메일서버 =========================
# ===== 도메인 / 호스트 =====
DOMAIN=example.com
HOSTNAME_FQDN=mail.example.com

# ===== 기본 계정 패스워드 =====
SUPPORT_PASS=verysecurepassword123!
NOREPLY_PASS=verysecurepassword123!

# ===== 별칭 옵션 =====
ALLOW_INFO_ALIAS=1          # 1이면 info@DOMAIN 별칭을 허용 (지원/문의용)

# ===== 부트스트랩 디버그 옵션 =====
BOOTSTRAP_SMOKE=0           # 1이면 swaks 발신 테스트 수행
BOOTSTRAP_VERIFY_CERTS=0    # 1이면 openssl 인증서 정보 출력

# ===== letsencrypt 옵션 =====
USE_LETSENCRYPT=0          # (SSL) 1이면 letsencrypt 사용
# ===========================================================
```

### ② 도커 컴포즈 실행
```docker comopose up```을 통해 실행

```

```


### ③ DNS 레코드 설정

### 변수 (치환해서 사용하세요)
- `<MAIL_DOMAIN>`     : **메일 주소의 도메인**                  _(ex: example.com)_
- `<MAIL_HOST>`       : **메일서버 FQDN**                       _(ex: mail.example.com)_
- `<WEB_IP>`          : **웹 서버 IPv4**                        _(웹과 메일 IP가 같다면 둘 다 같은 값을 사용하세요)_
- `<MAIL_IP>`         : **메일 서버 IPv4**                      _(웹과 메일 IP가 같다면 둘 다 같은 값을 사용하세요)_
- `<DKIM_TXT>`        : **DKIM TXT 값**                         _(<프로젝트 루트>/mail/config/opendkim/keys/<MAIL_DOMAIN>의 mail.txt 전체 내용)_
- `<REPORT_EMAIL>`    : **DMARC/TLS-RPT 리포트 수신 메일**       _(ex: postmaster@<MAIL_DOMAIN>)_

### A. 메인 메일 도메인 (<MAIL_DOMAIN>) — 필수 레코드
| 타입 | 호스트                         | 값/위치                         | 비고                                 |
|------|------------------------------|-------------------------------|--------------------------------------|
| A    | `<MAIL_DOMAIN>`                | `<WEB_IP>`                      |                                           |
| A    | `<MAIL_HOST>`                  | `<MAIL_IP>`                     | 메일서버 FQDN                             |
| MX   | `<MAIL_DOMAIN>`                | `10 <MAIL_HOST>.`               | **끝에 점(.) 필수**, 10은 우선순위를 뜻함 |
| TXT  | `<MAIL_DOMAIN>`                | `"v=spf1 mx ~all"`              |                                          |
| TXT  | `mail._domainkey.<MAIL_DOMAIN>` | `"<DKIM_TXT>" `                | DKIM: mail.txt의 값 전체             |
| TXT  | `_dmarc.<MAIL_DOMAIN>`         | `"v=DMARC1; p=none; rua=mailto:<REPORT_EMAIL>; adkim=r; aspf=r"` |  |

