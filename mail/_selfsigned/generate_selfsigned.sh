# 1) 컨테이너 내려두기
sudo docker compose down

# 2) 디렉터리 구조 정확히 생성 (demoCA 폴더 포함!)
mkdir -p ./mail/config/ssl/demoCA

# 3) 인증서/키 생성 (3650일 유효, CN=energy.meowning.kr)
openssl req -x509 -newkey rsa:4096 -sha256 -days 3650 -nodes \
  -keyout ./mail/config/ssl/energy.meowning.kr-key.pem \
  -out   ./mail/config/ssl/energy.meowning.kr-cert.pem \
  -subj "/CN=energy.meowning.kr"

# 4) demoCA/cacert.pem 위치에 "루트(자체) CA"로 쓸 파일 배치
#    => docker-mailserver는 이 '정확한 경로/파일명'을 요구합니다.
cp ./mail/config/ssl/energy.meowning.kr-cert.pem ./mail/config/ssl/demoCA/cacert.pem

# 5) 권한 깔끔하게 (필수는 아니지만 권장)
chmod 600 ./mail/config/ssl/energy.meowning.kr-key.pem
chmod 644 ./mail/config/ssl/energy.meowning.kr-cert.pem ./mail/config/ssl/demoCA/cacert.pem

# 6) 생성 확인 (이 3개 파일이 꼭 보여야 합니다)
ls -l ./mail/config/ssl ./mail/config/ssl/demoCA
