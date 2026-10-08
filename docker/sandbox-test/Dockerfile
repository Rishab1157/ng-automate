# Sandbox for running and healing test projects: the OpenHands agent server (same version as the SDK in
# third_party/) plus the build tools test-automation projects need. Python 3.13, Node 24, Chromium and git
# come with the base image.
#
# Live view: browsers the tests drive show on a virtual display (Xvfb :99) that users watch, and can take over,
# in the web UI through the API (see ngauto-entrypoint.sh).
#
# Build (from the repo root):
#   docker build -t ng-automate/sandbox-test:1.53.0 docker/sandbox-test
FROM ghcr.io/openhands/agent-server:1.53.0-python

USER root
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        openjdk-21-jdk-headless \
        maven \
        chromium-driver \
        unzip \
        xvfb \
        x11-utils \
        fluxbox \
        x11vnc \
        websockify \
        novnc \
        xdotool \
    && rm -rf /var/lib/apt/lists/*

ENV JAVA_HOME=/usr/lib/jvm/java-21-openjdk-amd64 \
    MAVEN_OPTS="-Xmx1024m"

# Build caches are mounted here as per-organization Docker volumes. They must exist owned by the sandbox
# user: Docker copies a folder's owner into a new, empty volume mounted on it.
RUN mkdir -p /home/openhands/.m2 /home/openhands/.npm /home/openhands/.cache/pip \
    && chown -R openhands:openhands /home/openhands/.m2 /home/openhands/.npm /home/openhands/.cache

# The X server runs as the sandbox user, so its socket folder must already exist, world-writable and sticky.
RUN mkdir -p /tmp/.X11-unix && chmod 1777 /tmp/.X11-unix

# ngauto-find-elements: lists a page's interactive elements with locators checked unique (used to write tests).
COPY ngauto-entrypoint.sh ngauto-find-elements /usr/local/bin/
# A Windows checkout can turn line endings into CRLF, which bash and the #! line cannot read.
RUN sed -i 's/\r$//' /usr/local/bin/ngauto-entrypoint.sh /usr/local/bin/ngauto-find-elements \
    && chmod 755 /usr/local/bin/ngauto-entrypoint.sh /usr/local/bin/ngauto-find-elements

# Commands the agent server runs see the display. Port 8001 belongs to websockify (the live view), so the agent
# server's own VSCode, which also wants 8001, stays off.
ENV DISPLAY=:99 \
    OH_ENABLE_VSCODE=false \
    NGAUTO_LIVE_VIEW=1

# Back to the base image's unprivileged user. Same entrypoint as the base image (tini + agent server), with the
# live view started in between; DockerWorkspace passes the agent server's arguments (--host, --port).
USER openhands
ENTRYPOINT ["tini", "--", "/usr/local/bin/ngauto-entrypoint.sh", "/usr/local/bin/openhands-agent-server"]
