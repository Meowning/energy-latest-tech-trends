FROM ghcr.io/docker-mailserver/docker-mailserver:15.1.0

COPY mailserver-setup.sh /usr/local/bin/mailserver-setup.sh
RUN chmod +x /usr/local/bin/mailserver-setup.sh

ENTRYPOINT [ "/usr/local/bin/mailserver-setup.sh" ]