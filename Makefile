
IMAGE := datathon
MODEL ?= woodcock
CONTAINER := recommender-$(MODEL)

.PHONY: build run stop restart logs status remove

build:
	podman build -t $(IMAGE) .

run:
	podman run -d \
		--name $(CONTAINER) \
		--env-file .env \
		--restart=unless-stopped \
		$(IMAGE) $(MODEL)

stop:
	podman stop $(CONTAINER)

restart:
	podman restart $(CONTAINER)

logs:
	podman logs -f $(CONTAINER)

status:
	podman ps -a --filter name=recommender-

remove:
	podman rm -f $(CONTAINER)
