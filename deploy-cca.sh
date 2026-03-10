#!/bin/bash
# Deploy CCA to Spark1 (DGX)
# Usage: ./deploy-cca.sh [build|run|shell|dev|logs]
#
# Workflow:
#   1. Push code from node5 to GitHub
#   2. Pull on Spark1 (including submodule)
#   3. Build container on Spark1
#   4. Run CCA agent
set -e

SPARK1="${CCA_DEPLOY_HOST:?Set CCA_DEPLOY_HOST env var (e.g. 192.168.1.100)}"
SPARK1_USER="${CCA_DEPLOY_USER:-seli}"
SPARK1_PASS="${CCA_DEPLOY_PASS:?Set CCA_DEPLOY_PASS env var}"
REMOTE_DIR="docker-swarm-stacks/nvidia-dgx-spark/cca"

ssh_cmd() {
    sshpass -p "$SPARK1_PASS" ssh -o StrictHostKeyChecking=no "$SPARK1_USER@$SPARK1" "$@"
}

ACTION="${1:-build}"

case "$ACTION" in
    pull)
        echo "=== Pulling latest code on Spark1 ==="
        ssh_cmd "cd docker-swarm-stacks && git pull && git submodule update --init --recursive nvidia-dgx-spark/cca"
        ;;

    build)
        echo "=== Building CCA on Spark1 ==="
        ssh_cmd "cd docker-swarm-stacks && git pull && git submodule update --init --recursive nvidia-dgx-spark/cca"
        ssh_cmd "cd $REMOTE_DIR && docker compose -f cca-compose.yml build"
        echo ""
        echo "Build complete. Run with: ./deploy-cca.sh run"
        ;;

    run)
        echo "=== Running CCA REPL on Spark1 ==="
        sshpass -p "$SPARK1_PASS" ssh -tt -o StrictHostKeyChecking=no "$SPARK1_USER@$SPARK1" \
            "cd $REMOTE_DIR && docker compose -f cca-compose.yml run --rm cca"
        ;;

    shell)
        echo "=== Opening shell in CCA container on Spark1 ==="
        sshpass -p "$SPARK1_PASS" ssh -tt -o StrictHostKeyChecking=no "$SPARK1_USER@$SPARK1" \
            "cd $REMOTE_DIR && docker compose -f cca-compose.yml run --rm cca bash"
        ;;

    dev)
        echo "=== Running CCA dev container (source mounted) ==="
        sshpass -p "$SPARK1_PASS" ssh -tt -o StrictHostKeyChecking=no "$SPARK1_USER@$SPARK1" \
            "cd $REMOTE_DIR && docker compose -f cca-compose.yml --profile dev run --rm cca-dev"
        ;;

    logs)
        echo "=== CCA container logs ==="
        ssh_cmd "docker logs --tail 100 cca 2>/dev/null || echo 'No running CCA container'"
        ;;

    *)
        echo "Usage: $0 [pull|build|run|shell|dev|logs]"
        echo ""
        echo "  pull   - Pull latest code on Spark1 (git + submodule)"
        echo "  build  - Pull + build CCA Docker image on Spark1"
        echo "  run    - Launch CCA interactive REPL on Spark1"
        echo "  shell  - Open bash shell inside CCA container"
        echo "  dev    - Run with source mounted (no rebuild for code changes)"
        echo "  logs   - View CCA container logs"
        exit 1
        ;;
esac
