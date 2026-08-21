FROM ghcr.io/prefix-dev/pixi:0.75.0

WORKDIR /app

RUN apt-get update \
	&& apt-get install -y --no-install-recommends git \
	&& rm -rf /var/lib/apt/lists/*

COPY . .

RUN pixi install --locked

CMD ["pixi", "run", "main"]
