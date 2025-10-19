FROM rust:1.82 as builder
WORKDIR /app
ENV RUSTFLAGS="-C target-cpu=native"

COPY Cargo.toml Cargo.lock ./
RUN mkdir -p src && echo 'fn main(){}' > src/main.rs
RUN cargo fetch --locked

COPY src ./src
RUN cargo build --locked --release

FROM debian:bookworm-slim
RUN apt-get update && apt-get install -y libgomp1 ca-certificates && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY --from=builder /app/target/release/api /usr/local/bin/api
ENV RUST_LOG=info
CMD ["/usr/local/bin/api"]
