# 메일서버 스터디
메일서버 구축하면서 머리에 자꾸 과부하가 와서 한번 정리하는 것...

## 1. 도메인과 메일서버
### (1) 도메인 (Domain)
- 메일 주소의 골뱅이(@) 뒤에 오는 부분 (ex: `user@example.com -> example.com`)
- SPF, DKIM, DMARC 정책(후술)은 도메인을 기준으로 동작
- **정책적/논리적 단위 (메일주소, DKIM, DMARC)**

### (2) FQDN (Fully Qualified Domain Name)
- 메일 서버가 네트워크에서 식별되는 이름 (ex: `mail.example.com`)
- SMTP, IMAP, Submission 접속 시 클라이언트가 FQDN으로 접속
- **네트워크 접속 단위 (TLS(SSL), SMTP)**

## 2. DNS와 네트워크 개념
### (1) 주요 레코드
- A레코드 : 호스트명 -> IPv4 주소 변환 `mail.example.com. IN A 192.0.2.10`
- AAAA 레코드: 호스트명 -> IPv6 `mail.example.com. IN AAAA 2001:db8::1`
- TXT 레코드: 텍스트 형태의 임의 정보 저장, 메일 보안 정책(SPF, DKIM, DMARc) 정책 정의에 활용함
- MX 레코드: 특정 도메인으로 들어온 메일을 어떤 메일 서버(FQDN)로 전달할지 지정 `example.com MX 10 mail.example.com.`
- PTR 레코드: IP 주소 -> 호스트명 역방향 매핑. 메일서버 신뢰도 검증 可
- CNAME 레코드: 한 도메인을 다른 도메인의 별칭으로 지정 (MX와 같이 사용 불가)
- CAA 레코드: 어떤 인증기관이(CA) 해당 도메인의 TLS 인증서를 발급할 수 있는지 지정
- SRV 레코드: 특정 서비스와 포트 지정

## 3. TLS(SSL) 인증서
- SSL은 옛 명칭이며, 현재는 TLS가 표준임
- FQDN

## 4. 메일 보안 정책
### SPF (Sender Policy Framework)
- 어떤 서버가 메일을 보낼 수 있는지 지정
- TXT 레코드로 발신 허용 서버 IP/FQDN 목록을 선언
- `v=spf1 mx ~all` -> MX에 등록된 서버(전체)만 허용

### DKIM (DomainKeys Identified Mail)
- 발신 서버가 메일 헤더와 본문 일부에 전자서명 추가 (공개키 암호화)
- 발신 서버는 메일에 **비밀키** 포함
- 수신 서버는 DNS에 등록된 **공개키**(`selector._domainkey.example.com`)를 조회하여 서명 검증
- 메일 내용이나 헤더가 중간에 변경되면 서명이 일치하지 않아 FAIL 처리

### DMARC (Domain-based Message Authentication, Reporting & Conformance)
- SPF와 DKIM의 결과를 종합하여 최종 정책 결정
- 

## 2. DNS 레코드와 메일 발송 흐름
### MX 레코드
- 도메인으로 들어오는 메일을 어떤 서버로 보낼지 지정
- 예: `example.com MX 10 mail.example.com.`

### 메일 발송 흐름 (SMTP)
1. 발신 서버가 수신 도메인(`example.com`)의 MX 레코드 조회
2. `mail.example.com` 확인 후 그 서버로 TCP 연결
3. TLS 협상(STARTTLS 같은...) 시 `mail.example.com`의 인증서 제시
4. 인증서 CN/SAN이 접속한 FQDN과 일치해야 신뢰

