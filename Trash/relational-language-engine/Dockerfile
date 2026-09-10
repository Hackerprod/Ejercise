FROM debian:bookworm-slim AS build
RUN apt-get update && apt-get install -y --no-install-recommends cmake g++ make ca-certificates && rm -rf /var/lib/apt/lists/*
WORKDIR /src
COPY . .
RUN cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DRLM_BUILD_TESTS=ON \
 && cmake --build build --parallel \
 && ctest --test-dir build --output-on-failure

FROM debian:bookworm-slim
RUN useradd --system --home /opt/rlm --shell /usr/sbin/nologin rlm \
 && mkdir -p /opt/rlm/bin /opt/rlm/config /opt/rlm/data \
 && chown -R rlm:rlm /opt/rlm
COPY --from=build /src/build/rlmctl /opt/rlm/bin/rlmctl
COPY --from=build /src/config/production.toml /opt/rlm/config/production.toml
USER rlm
WORKDIR /opt/rlm
EXPOSE 9087
ENTRYPOINT ["/opt/rlm/bin/rlmctl"]
CMD ["serve", "--config", "/opt/rlm/config/production.toml"]
