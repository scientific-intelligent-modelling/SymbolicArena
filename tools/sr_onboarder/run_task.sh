set -euo pipefail
exec "$1" -m scientific_intelligent_modelling.onboarding.task --request "$2" > "$3" 2>&1
