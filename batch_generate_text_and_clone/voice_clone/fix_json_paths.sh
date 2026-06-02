#!/bin/bash
set -euo pipefail

# Wait for all clone workers to finish
echo "$(date) Waiting for clone workers to finish..."
while pgrep -f "clone_dataset.py" > /dev/null 2>&1; do
    sleep 60
done
echo "$(date) All clone workers finished. Starting JSON path fix..."

OLD_PREFIX="/root/code/github_repos/OmniVoice-fork/batch_cloned_voices"
NEW_PREFIX="/root/group-shared/voiceprint/data/speech/voice_activity_detection/batch_cloned_voices_ommivoice_kids_finetuned"

DIR="/root/group-shared/voiceprint/data/speech/voice_activity_detection/batch_cloned_voices_ommivoice_kids_finetuned"

# Find all JSON files that contain the old path
echo "$(date) Scanning JSON files with old paths..."
files_to_fix=$(grep -rl "$OLD_PREFIX" "$DIR" --include="*.json" 2>/dev/null | wc -l)
echo "$(date) Found $files_to_fix JSON files to fix."

# Fix paths using sed in parallel
echo "$(date) Fixing paths..."
grep -rl "$OLD_PREFIX" "$DIR" --include="*.json" 2>/dev/null | xargs -P 32 -I {} sed -i "s|$OLD_PREFIX|$NEW_PREFIX|g" {}

echo "$(date) Done! Fixed $files_to_fix files."
