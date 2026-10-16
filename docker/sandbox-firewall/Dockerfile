# Sets a sandbox's outgoing-traffic rules from outside the sandbox. For every sandbox the API starts this image once in
# the sandbox's network namespace (docker run --network container:<sandbox> --cap-add NET_ADMIN); it reads the rules on
# stdin, loads them with iptables-restore and exits. The sandbox itself has no NET_ADMIN, so even root in there (agents
# have sudo) cannot change or remove the rules.
FROM debian:trixie-slim
RUN apt-get update \
    && apt-get install -y --no-install-recommends iptables \
    && rm -rf /var/lib/apt/lists/*
ENTRYPOINT ["iptables-restore"]
