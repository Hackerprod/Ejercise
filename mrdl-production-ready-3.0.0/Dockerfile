FROM ubuntu:24.04 AS build
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
      build-essential cmake ninja-build libsqlite3-dev ca-certificates \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /src
COPY . .
RUN cmake -S . -B build/container -G Ninja \
      -DCMAKE_BUILD_TYPE=Release \
      -DMRDL_NATIVE_ARCH=OFF \
      -DMRDL_ENABLE_LTO=ON \
      -DMRDL_BUILD_TESTS=ON \
      -DMRDL_BUILD_BENCHMARKS=ON \
    && cmake --build build/container --parallel 2 \
    && ctest --test-dir build/container --output-on-failure \
    && cmake --install build/container --prefix /opt/mrdl

FROM ubuntu:24.04
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends libsqlite3-0 ca-certificates \
    && rm -rf /var/lib/apt/lists/* \
    && groupadd --system mrdl \
    && useradd --system --gid mrdl --home /var/lib/mrdl --shell /usr/sbin/nologin mrdl \
    && install -d -o mrdl -g mrdl /var/lib/mrdl/model /data \
    && install -d -o root -g mrdl -m 0750 /etc/mrdl
COPY --from=build /opt/mrdl /opt/mrdl
COPY --chown=root:mrdl --chmod=0640 config/docker.ini /etc/mrdl/mrdl.ini
ENV PATH="/opt/mrdl/bin:${PATH}"
USER mrdl:mrdl
WORKDIR /var/lib/mrdl
VOLUME ["/var/lib/mrdl/model"]
ENTRYPOINT ["mrdl"]
CMD ["doctor", "--config", "/etc/mrdl/mrdl.ini", "--allow-unprepared"]
