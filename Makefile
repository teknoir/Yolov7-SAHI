.PHONY: help install run test docker-build docker-run gcloud-build

help:
	@echo "YOLOv7 with SAHI Project Makefile"
	@echo ""
	@echo "Choose a command run in camera-capture:"
	@echo "  install        - Install the necessary dependencies"
	@echo "  run            - Run the app.py script"
	@echo "  test           - Run tests (command to be added)"
	@echo "  docker-build   - Build the Docker image"
	@echo "  docker-run     - Run the Docker container with necessary environment variables"
	@echo "  gcloud-build   - Build the project using Google Cloud Builder"
	@echo ""

install:
	@echo "Installing dependencies..."
	pip install -r requirements.txt

run:
	@echo "Running application..."
	python src/yolov7app-SAHI

test:
	@echo "Running tests..."
	@pytest tests/test_app.py

lint-fix:
	@python -m isort -l 79 --profile black .
	@python -m black -l 79 .

docker-build:
	@echo "Building Docker image..."
	docker build -t yolov7app_sahi -f Dockerfile .

docker-run:
	@echo "Running Docker container..."
	docker run -e LOG_LEVEL=DEBUG yolov7app_sahi

gcloud-build:
	@echo "Building with Google Cloud Builder..."
	gcloud --project=teknoir builds submit . --config=cloudbuild.yaml --substitutions=SHORT_SHA="0.0.1",BRANCH_NAME="main"

