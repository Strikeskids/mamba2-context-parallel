#!/bin/bash

set -euo pipefail

dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [[ "$#" -ne 1 ]]; then
    echo "Usage: $0 TARGET"
fi

target="$1"

sync () {
    rsync -arP --filter ':- .gitignore' --exclude .git "$dir/" "$target"
}

sync

fswatch --exclude .git -o "$dir" \
| while read count; do 
    sync
done
