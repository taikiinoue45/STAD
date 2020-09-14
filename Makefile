docker_build:
		docker build -t stad:latest -f docker/Dockerfile .

docker_run:
		docker run --runtime nvidia -it --rm \
			--network host \
			--workdir /app/STAD \
			--name stad \
			--hostname stad \
			stad:latest /bin/bash

python_run:
		python stad/run.py yamls/mvtec.yaml
