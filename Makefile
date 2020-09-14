docker_build:
		docker build -t stad:latest .

docker_run:
		docker run --runtime nvidia -it --rm \
			--network host \
			--workdir /app \
			--name stad \
			--hostname stad \
			stad:latest /bin/bash
