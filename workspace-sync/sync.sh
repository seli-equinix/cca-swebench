#!/bin/sh
# Workspace-Sync: Clone/pull repos from GitLab to Spark1 local storage.
# EVA (source): periodic pull (CCA reads for tracing/analysis).
# EVA-migration: clone only (CCA manages commits/push directly).
# NOTE: Uses POSIX sh (alpine image has ash, not bash).

set -eu

GITLAB_URL="${GITLAB_URL:-http://192.168.4.204:8929}"
GITLAB_USER="${GITLAB_USER:-root}"
GITLAB_PASS="${GITLAB_PASS:-Loveme-sex64}"
LOCAL_PATH="/workspace"
SYNC_INTERVAL="${SYNC_INTERVAL:-60}"
CCA_URL="${CCA_URL:-http://localhost:8500}"
SOURCE_PROJECTS="${SOURCE_PROJECTS:-EVA}"
MIGRATION_PROJECTS="${MIGRATION_PROJECTS:-EVA-migration}"

REMOTE_BASE="http://${GITLAB_USER}:${GITLAB_PASS}@$(echo "$GITLAB_URL" | sed 's|http://||')/${GITLAB_USER}"

# Install deps
apk add --no-cache curl git 2>/dev/null || true

# Clone source project repos if not already cloned
OLD_IFS="$IFS"
IFS=','
for project in $SOURCE_PROJECTS; do
    IFS="$OLD_IFS"
    project=$(echo "$project" | xargs)
    if [ ! -d "${LOCAL_PATH}/${project}/.git" ]; then
        echo "[$(date -Iseconds)] Cloning source repo: ${project}..."
        git clone "${REMOTE_BASE}/${project}.git" "${LOCAL_PATH}/${project}" 2>&1 || \
            echo "[$(date -Iseconds)] WARN: clone failed for ${project} (repo may not exist yet)"
    fi
done
IFS="$OLD_IFS"

# Clone migration repos if not already cloned
OLD_IFS="$IFS"
IFS=','
for project in $MIGRATION_PROJECTS; do
    IFS="$OLD_IFS"
    project=$(echo "$project" | xargs)
    if [ ! -d "${LOCAL_PATH}/${project}/.git" ]; then
        echo "[$(date -Iseconds)] Cloning migration repo: ${project}..."
        git clone "${REMOTE_BASE}/${project}.git" "${LOCAL_PATH}/${project}" 2>&1 || \
            echo "[$(date -Iseconds)] WARN: clone failed for ${project} (repo may not exist yet)"
        if [ -d "${LOCAL_PATH}/${project}/.git" ]; then
            cd "${LOCAL_PATH}/${project}"
            git config user.name "cca"
            git config user.email "cca@local"
            cd "$LOCAL_PATH"
        fi
    fi
done
IFS="$OLD_IFS"

echo "[$(date -Iseconds)] Starting pull loop (interval=${SYNC_INTERVAL}s, sources=${SOURCE_PROJECTS})"

# Periodic pull loop (source repos only — CCA manages migration repos)
while true; do
    sleep "$SYNC_INTERVAL"
    CHANGED=0

    OLD_IFS="$IFS"
    IFS=','
    for project in $SOURCE_PROJECTS; do
        IFS="$OLD_IFS"
        project=$(echo "$project" | xargs)
        cd "${LOCAL_PATH}/${project}" 2>/dev/null || continue
        BEFORE=$(git rev-parse HEAD 2>/dev/null || echo "none")
        git pull --ff-only 2>&1 | tail -3
        AFTER=$(git rev-parse HEAD 2>/dev/null || echo "none")
        if [ "$BEFORE" != "$AFTER" ]; then
            CHANGED=1
            echo "[$(date -Iseconds)] ${project} updated"
        fi
    done
    IFS="$OLD_IFS"

    # Trigger CCA re-index if source code changed
    if [ "$CHANGED" -eq 1 ]; then
        echo "[$(date -Iseconds)] Source updated, triggering re-index..."
        curl -sf -X POST "${CCA_URL}/workspace/reindex" \
            -H 'Content-Type: application/json' \
            -d '{"force": false}' \
            --connect-timeout 5 --max-time 60 2>/dev/null || \
            echo "[$(date -Iseconds)] WARN: re-index trigger failed (CCA may not be running)"
    fi
done
