# OMEGA pod template. Build context must be t1_trainability_lab_v0.1.0.
# Digest is intentional: never replace it with a floating tag.
FROM pytorch/pytorch:2.11.0-cuda12.8-cudnn9-runtime@sha256:eee11b3b3872a8c838e35ef48f08b2d5def2080902c7f666831310ca1a0ef2be

ARG DEBIAN_FRONTEND=noninteractive

RUN apt-get update \
    && apt-get install --no-install-recommends -y openssh-server openssh-client rsync tini ca-certificates python3.12-venv \
    && rm -rf /var/lib/apt/lists/* \
    && rm -f /etc/ssh/ssh_host_*_key /etc/ssh/ssh_host_*_key.pub \
    && test -z "$(find /etc/ssh -maxdepth 1 -type f -name 'ssh_host_*_key' -print -quit)" \
    && mkdir -p /run/sshd /workspace/omega

COPY campaign/omega_core_lm_0_gpu_environment_preparation/requirements.lock /opt/omega/requirements.lock
RUN python -m venv --system-site-packages /opt/omega/venv
ENV PATH="/opt/omega/venv/bin:${PATH}"
RUN python -m pip install --no-cache-dir -r /opt/omega/requirements.lock \
    && python -m pip check \
    && python -c "import datasets, huggingface_hub, psutil, safetensors, tokenizers, torch, transformers; assert torch.__version__.split('+')[0] == '2.11.0'; print('OMEGA dependency versions: PASS')"

COPY campaign/omega_core_lm_0_gpu_environment_preparation/docker/sshd_config /etc/ssh/sshd_config
COPY campaign/omega_core_lm_0_gpu_environment_preparation/docker/runpod-entrypoint.sh /usr/local/bin/omega-runpod-entrypoint
COPY campaign/omega_core_lm_0_gpu_environment_preparation/omega_nominal_microbatch_runner.py /opt/omega/omega_nominal_microbatch_runner.py
COPY campaign/omega_core_lm_0_gpu_environment_preparation/budget_controller.py /opt/omega/budget_controller.py
COPY campaign/omega_core_lm_0_gpu_environment_preparation/vps_control.py /opt/omega/vps_control.py
COPY campaign/omega_core_lm_0_gpu_environment_preparation/test_omega_nominal_microbatch_runner.py /opt/omega/test_omega_nominal_microbatch_runner.py
COPY campaign/omega_core_lm_0_gpu_environment_preparation/test_budget_controller.py /opt/omega/test_budget_controller.py
COPY scripts/run_omega_core_lm_0_r1_training_technical_preflight.py /opt/omega/scripts/run_omega_core_lm_0_r1_training_technical_preflight.py
COPY scripts/audit_omega_core_lm_0_design.py /opt/omega/audit_omega_core_lm_0_design.py
COPY t1_trainability/model.py /opt/omega/t1_trainability/model.py
COPY campaign/omega_core_lm_0_gpu_environment_preparation/docker/omega_image_package_init.py /opt/omega/t1_trainability/__init__.py
RUN chmod 0755 /usr/local/bin/omega-runpod-entrypoint
RUN cd /opt/omega \
    && python -m py_compile omega_nominal_microbatch_runner.py budget_controller.py vps_control.py \
    && python -c "from t1_trainability.model import CoreMLP, RMSNorm, SlotMix; print('OMEGA package import: PASS')" \
    && python test_budget_controller.py \
    && python test_omega_nominal_microbatch_runner.py

EXPOSE 22

ENTRYPOINT ["/usr/bin/tini", "--", "/usr/local/bin/omega-runpod-entrypoint"]
