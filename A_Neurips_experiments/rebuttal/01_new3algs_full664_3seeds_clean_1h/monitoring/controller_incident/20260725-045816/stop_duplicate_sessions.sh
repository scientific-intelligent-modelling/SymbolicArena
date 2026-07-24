#!/usr/bin/env bash
set -u -o pipefail

PLAN="${1:?usage: stop_duplicate_sessions.sh PLAN_TSV}"
LOG="${2:?usage: stop_duplicate_sessions.sh PLAN_TSV LOG}"

: > "$LOG"
while IFS=$'\t' read -r host session task_id canonical_host; do
  if [[ "$host" == "host" ]]; then
    continue
  fi
  expected_session="neurips_rebuttal_new3_1h_${task_id}"
  if [[ "$session" != "$expected_session" || "$host" == "$canonical_host" ]]; then
    printf 'REFUSED\t%s\t%s\t%s\t%s\n' "$host" "$session" "$task_id" "$canonical_host" | tee -a "$LOG"
    continue
  fi

  if [[ "$host" == "iaaccn22" ]]; then
    if tmux has-session -t "$session" 2>/dev/null; then
      tmux kill-session -t "$session"
      printf 'STOPPED\t%s\t%s\t%s\t%s\n' "$host" "$session" "$task_id" "$canonical_host" | tee -a "$LOG"
    else
      printf 'ALREADY_GONE\t%s\t%s\t%s\t%s\n' "$host" "$session" "$task_id" "$canonical_host" | tee -a "$LOG"
    fi
    continue
  fi

  suffix="${host#iaaccn}"
  target="10.10.100.${suffix}"
  if timeout 15 ssh -n -o BatchMode=yes -o ConnectTimeout=8 "$target" \
    "tmux has-session -t '$session' 2>/dev/null && tmux kill-session -t '$session'"; then
    printf 'STOPPED\t%s\t%s\t%s\t%s\n' "$host" "$session" "$task_id" "$canonical_host" | tee -a "$LOG"
  else
    printf 'ALREADY_GONE_OR_UNREACHABLE\t%s\t%s\t%s\t%s\n' \
      "$host" "$session" "$task_id" "$canonical_host" | tee -a "$LOG"
  fi
done < "$PLAN"
