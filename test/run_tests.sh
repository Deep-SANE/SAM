#!/usr/bin/env bash
# Run the SAM test programs and print a summary table at the end.
# Called by the test Makefile:  run_tests.sh "<mpirun command>" program...
#
# Each program runs as one MPI process with SAM_DATA_DIR set to a fresh
# temporary directory, removed afterwards. Exit status is nonzero if any
# program fails or crashes.
set -u

mpirun="$1"
shift

rows=()
total_passed=0
total_failed=0
total_time=0
all_ok=1

for program in "$@"; do
  name="$(basename "$program")"
  echo "=== $name ==="

  data="$(mktemp -d)"
  log="$(mktemp)"
  start=$SECONDS
  SAM_DATA_DIR="$data" $mpirun "$program" > "$log" 2>&1
  rc=$?
  elapsed=$((SECONDS - start))
  cat "$log"

  # each program ends with "Summary: <p> passed, <f> failed"
  summary="$(grep '^Summary:' "$log" | tail -1)"
  rm -rf "$data" "$log"

  if [[ -z "$summary" ]]; then
    passed="-"; failed="-"; result="CRASHED (exit $rc)"; all_ok=0
  else
    passed="$(echo "$summary" | sed 's/^Summary: \([0-9]*\) passed.*/\1/')"
    failed="$(echo "$summary" | sed 's/.*, \([0-9]*\) failed.*/\1/')"
    total_passed=$((total_passed + passed))
    total_failed=$((total_failed + failed))
    if [[ $rc -ne 0 || $failed -ne 0 ]]; then
      result="FAIL"; all_ok=0
    else
      result="ok"
    fi
  fi
  total_time=$((total_time + elapsed))
  rows+=("$(printf '%-48s %7s %7s %9s   %s' "$name" "$passed" "$failed" "$elapsed" "$result")")
done

line="$(printf '%.0s-' {1..86})"
echo
echo "SAM test summary"
echo "$line"
printf '%-48s %7s %7s %9s   %s\n' "Test program" "Passed" "Failed" "Time (s)" "Result"
echo "$line"
for row in "${rows[@]}"; do echo "$row"; done
echo "$line"
if [[ $all_ok -eq 1 ]]; then overall="PASS"; else overall="FAIL"; fi
printf '%-48s %7s %7s %9s   %s\n' "Total ($# programs)" "$total_passed" "$total_failed" "$total_time" "$overall"
echo "$line"

[[ $all_ok -eq 1 ]]
