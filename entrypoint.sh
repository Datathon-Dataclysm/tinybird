
#!/bin/sh
set -eu

MODEL="${1:-woodcock}"

case "$MODEL" in
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
        echo "Modelo desconocido: $MODEL" >&2
        exit 1
        ;;
esac

exec "$@"
