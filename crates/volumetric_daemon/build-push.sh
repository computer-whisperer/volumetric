#!/bin/bash
set -euo pipefail
# Build context is the repo root so the workspace path deps resolve.
cd "$(dirname "$0")/../.."

# Tag = crate version + git state, e.g. 0.1.0-d363c83 or 0.1.0-d363c83-dirty.
VERSION=$(grep -m1 '^version' crates/volumetric_daemon/Cargo.toml | cut -d'"' -f2)
TAG=${VERSION}-$(git describe --always --dirty)
IMAGE=registry.cjbal.com/volumetric-daemon:${TAG}

buildah build --layers -t "${IMAGE}" -t registry.cjbal.com/volumetric-daemon:latest \
  -f crates/volumetric_daemon/Containerfile .
buildah push --tls-verify=false "${IMAGE}"
buildah push --tls-verify=false registry.cjbal.com/volumetric-daemon:latest
echo "Pushed ${IMAGE} and :latest"
echo "Deployment yaml: /ceph/public/k8s/apps/volumetric-daemon/ — pin it to ${TAG}"
