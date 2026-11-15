IMAGE := datathon
MODEL ?= lynch-v8
CONTAINER := recommender-$(MODEL)

.PHONY: build test run stop restart logs status remove

build:
	podman build -t $(IMAGE) .

test:	# 2-minute shadow test of lynch-v8 (writes nothing)
	podman run --rm --env-file .env $(IMAGE) lynch-v8-shadow

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
