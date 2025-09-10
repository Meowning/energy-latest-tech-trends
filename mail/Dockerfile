FROM ghcr.io/docker-mailserver/docker-mailserver:15.1.0

USER root
RUN apt-get update && apt-get install -y --no-install-recommends \
      tini swaks openssl ca-certificates \
  && rm -rf /var/lib/apt/lists/*

COPY bootstrap.sh /usr/local/bin/bootstrap.sh
RUN chmod +x /usr/local/bin/bootstrap.sh

ENTRYPOINT ["/usr/bin/tini","--","/usr/local/bin/bootstrap.sh"]
