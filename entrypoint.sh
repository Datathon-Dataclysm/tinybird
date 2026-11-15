#!/bin/sh
set -eu

MODEL="${1:-lynch-v8}"

case "$MODEL" in
    lynch-v8)
        set -- python -u lynch/live.py --submit
        ;;
    lynch-v8-shadow)
        # 2-minute test: computes answers, writes NOTHING
        set -- python -u lynch/live.py --minutes 2
        ;;
    woodcock)
        set -- python -u starter/live.py --version v1-woodcock
        ;;
    eagle)
        set -- python -u starter/live.py --version v2-eagle
        ;;
    experimental)
        set -- python -u starter/experimental.py
        ;;
    *)
        echo "Modelo desconocido: $MODEL (lynch-v8 | lynch-v8-shadow | woodcock | eagle | experimental)" >&2
        exit 1
        ;;
esac

exec "$@"
