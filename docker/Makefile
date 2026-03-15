# Docker image build chain
#
# Usage:
#   make all        # build full chain up to runner
#   make train      # build up to train image
#   make debug      # build debug image (current source on top of train)
#   make infer      # build inference image (DeepEP/DeepGEMM/FlashMLA)
#   make nviz       # build NVIZ image
#   make deps       # rebuild from deps down (after adding pip packages)
#   make push       # push all images
#   make push-runner # push only runner to GHCR
#   make push-nviz   # push only NVIZ
#
# Images:
#   base   → noumena-network/nmoe:base       (CUDA + system, ~6GB, rarely changes)
#   system → noumena-network/nmoe:system     (torch + triton + vendors, ~24GB, almost never)
#   deps   → noumena-network/nmoe:deps       (pip deps, ~500MB, sometimes)
#   train  → noumena-network/nmoe_train      (built kernels, changes with code)
#   debug  → noumena-network/nmoe_debug      (debug pod image, current source on top of train)
#   infer  → noumena-network/nmoe_infer      (inference: DeepEP/DeepGEMM/FlashMLA/nixl)
#   nviz   → noumena-network/nmoe_nviz       (Bun/Next.js monitoring UI)
#   runner → noumena-network/nmoe:runner     (CI runner)

.PHONY: all build base system deps train debug infer nviz runner push push-debug push-runner push-infer push-nviz clean

# Default target
.DEFAULT_GOAL := all

DOCKER_BUILD = docker build
CONTEXT = ..

# Base: CUDA + system packages + uv + venv
base:
	$(DOCKER_BUILD) -f Dockerfile.base -t noumena-network/nmoe:base $(CONTEXT)

# System: torch + triton + vendors (depends on base)
system: base
	$(DOCKER_BUILD) -f Dockerfile.system -t noumena-network/nmoe:system $(CONTEXT)

# Deps: pip packages (depends on system)
deps: system
	$(DOCKER_BUILD) -f Dockerfile.deps -t noumena-network/nmoe:deps $(CONTEXT)

# Train: copy source + build kernels (depends on deps)
train: deps
	$(DOCKER_BUILD) -f Dockerfile.train -t noumena-network/nmoe_train:latest $(CONTEXT)

# Debug: current source + debug tools on top of train
debug: train
	$(DOCKER_BUILD) -f Dockerfile.debug -t noumena-network/nmoe_debug:latest $(CONTEXT)

# Infer: inference with DeepEP/DeepGEMM/full FlashMLA (depends on deps)
infer: deps
	$(DOCKER_BUILD) -f Dockerfile.infer -t noumena-network/nmoe_infer:latest $(CONTEXT)

# NVIZ: monitoring UI and API surface
nviz:
	$(DOCKER_BUILD) -f Dockerfile.nviz -t noumena-network/nmoe_nviz:latest $(CONTEXT)

# Runner: add GitHub Actions runner (depends on train)
runner: train
	$(DOCKER_BUILD) -f Dockerfile.runner -t noumena-network/nmoe:runner $(CONTEXT)

# Build everything
build: debug nviz runner

# Build and push everything
all: build push

# Push all images to registries
push: push-base push-system push-deps push-train push-debug push-infer push-nviz push-runner

push-base:
	docker push noumena-network/nmoe:base

push-system:
	docker push noumena-network/nmoe:system

push-deps:
	docker push noumena-network/nmoe:deps

push-train:
	docker push noumena-network/nmoe_train:latest

push-debug:
	docker push noumena-network/nmoe_debug:latest

push-infer:
	docker push noumena-network/nmoe_infer:latest

push-nviz:
	docker push noumena-network/nmoe_nviz:latest

push-runner:
	docker push noumena-network/nmoe:runner

# Force rebuild from a specific stage (ignores dependencies)
force-%:
	$(DOCKER_BUILD) --no-cache -f Dockerfile.$* -t noumena-network/nmoe:$* $(CONTEXT)

# Show image sizes
sizes:
	@docker images --format "table {{.Repository}}:{{.Tag}}\t{{.Size}}" | grep -E "nmoe|REPOSITORY"

# Clean up dangling images
clean:
	docker image prune -f
