build: Dockerfile
	docker build -t mlhw4 .

run: build
	docker run -d --rm --name mlhw4c -p 8888:8888 -v `pwd`:/home/jovyan/work mlhw4 start-notebook.sh --NotebookApp.token=''

stop:
	docker stop mlhw4c

terminal:
	docker run --rm -it -v `pwd`:/home/jovyan/work mlhw4 /bin/bash

clean:
	docker rmi mlhw4