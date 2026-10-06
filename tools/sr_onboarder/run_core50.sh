set -euo pipefail
exec "$1" -m scientific_intelligent_modelling.onboarding.cli accept --manifest "$2" --stage core50 --datasets-root "$3" --output-root "$4" --workers "$5" --tmux > "$6" 2>&1
