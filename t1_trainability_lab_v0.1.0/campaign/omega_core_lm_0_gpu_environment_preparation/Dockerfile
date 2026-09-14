# OMEGA pod template. Digest is intentional: never replace it with a floating tag.
FROM pytorch/pytorch:2.11.0-cuda12.8-cudnn9-runtime@sha256:eee11b3b3872a8c838e35ef48f08b2d5def2080902c7f666831310ca1a0ef2be

ARG DEBIAN_FRONTEND=noninteractive

RUN apt-get update \
    && apt-get install --no-install-recommends -y openssh-server openssh-client rsync tini ca-certificates \
    && rm -rf /var/lib/apt/lists/* \
    && mkdir -p /run/sshd /workspace/omega

COPY docker/sshd_config /etc/ssh/sshd_config
COPY docker/runpod-entrypoint.sh /usr/local/bin/omega-runpod-entrypoint
RUN chmod 0755 /usr/local/bin/omega-runpod-entrypoint

EXPOSE 22
VOLUME ["/workspace/omega"]

ENTRYPOINT ["/usr/bin/tini", "--", "/usr/local/bin/omega-runpod-entrypoint"]
