FROM us-central1-docker.pkg.dev/amp-prod-0/amp-orb/amp-orb:latest
USER user
RUN curl -fsSL https://ampcode.com/install.sh | bash \
    && mkdir -p /home/user/.config/amp \
    && [ -f /home/user/.config/amp/settings.json ] || printf '{}\n' > /home/user/.config/amp/settings.json
