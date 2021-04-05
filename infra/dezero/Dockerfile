FROM rust:1.51

# Prerequisites
#RUN apt-get update -y

# Install development tools
RUN rustup component add rustfmt clippy

# Install packages
RUN cargo install cargo-edit
