setup:
	docker run -it -d -v $$PWD:/usr/local/app -w /usr/local/app  --name voxels python
	docker exec -it voxels apt-get update
	docker exec -it voxels apt-get install ffmpeg libsm6 libxext6  -y
	docker exec -it voxels pip install -r requirements.txt
start:
	docker start voxels
	docker exec -it voxels bash
stop:
	docker stop voxels
destroy:
	docker rm voxels