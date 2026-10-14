# Sandbox for running and healing test projects: OUR OpenHands agent server, built from the vendored SDK in
# third_party/ (so every change made there runs in the sandbox), plus the build tools test-automation projects need.
# Python 3.13, Node 24, Chromium and git come with the base image.
#
# Live view: browsers the tests drive show on a virtual display (Xvfb :99) that users watch, and can take over,
# in the web UI through the API (see ngauto-entrypoint.sh).
#
# Build (from the repo root), the agent server first:
#   docker compose --profile build build agent-server
#   docker compose --profile build build sandbox-test
# The earlier image on OpenHands' prebuilt agent server stays as ng-automate/sandbox-test:1.53.0: to go back to it,
# set SANDBOX_TEST_IMAGE=ng-automate/sandbox-test:1.53.0 in .env.
FROM ng-automate/agent-server:1.53.0-source

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

# Chrome for the projects' tests. Inside the container Chrome's own sandbox cannot start ("Chrome instance exited"),
# so every start gets --no-sandbox: Debian's chromium launcher reads its flags from /etc/chromium.d (the container is
# the isolation). Projects that look for Google Chrome (Selenium, WebDriverManager, chromedriver) find Chromium under
# its usual names, so their own code and drivers need no change.
RUN printf '%s\n' \
        '# NG Automate sandbox: Chrome runs in a container, which is the isolation; its own sandbox cannot start here.' \
        'export CHROMIUM_FLAGS="$CHROMIUM_FLAGS --no-sandbox"' \
        > /etc/chromium.d/ngauto-container \
    && ln -sf /usr/bin/chromium /usr/bin/google-chrome \
    && ln -sf /usr/bin/chromium /usr/bin/google-chrome-stable

# The window manager's default style asks fbsetbg for a wallpaper, fbsetbg finds no wallpaper tool in the image and
# puts an error dialog on the live view. The sandbox's screen needs no wallpaper: fbsetbg does nothing here
# (/usr/local/bin comes before /usr/bin in PATH).
RUN ln -sf /bin/true /usr/local/bin/fbsetbg

# ngauto-find-elements: lists a page's interactive elements with locators checked unique (used to write tests).
COPY ngauto-entrypoint.sh ngauto-find-elements /usr/local/bin/
# A Windows checkout can turn line endings into CRLF, which bash and the #! line cannot read.
RUN sed -i 's/\r$//' /usr/local/bin/ngauto-entrypoint.sh /usr/local/bin/ngauto-find-elements \
    && chmod 755 /usr/local/bin/ngauto-entrypoint.sh /usr/local/bin/ngauto-find-elements

# noVNC focuses its screen when it connects. Without preventScroll the browser scrolls the page that embeds the live
# view down to it, so NG Automate's step pages jumped away from what the user was reading.
RUN sed -i 's/UI\.rfb\.focus();/UI.rfb.focus({ preventScroll: true });/' /usr/share/novnc/app/ui.js

# Commands the agent server runs see the display. Port 8001 belongs to websockify (the live view), so the agent
# server's own VSCode, which also wants 8001, stays off.
ENV DISPLAY=:99 \
    OH_ENABLE_VSCODE=false \
    NGAUTO_LIVE_VIEW=1

# Back to the base image's unprivileged user. Same entrypoint as the base image (tini + the agent server from
# third_party/, run from source), with the live view started in between; DockerWorkspace passes the agent server's
# arguments (--host, --port).
USER openhands
ENTRYPOINT ["tini", "--", "/usr/local/bin/ngauto-entrypoint.sh", "/agent-server/.venv/bin/python", "-m", "openhands.agent_server"]
