#!/usr/bin/env bash
# Run the SAM test programs and print a summary table at the end.
# Called by the test Makefile:  run_tests.sh "<mpirun command>" program...
#
# Each program runs as one MPI process with SAM_DATA_DIR set to a fresh
# temporary directory, removed afterwards. Tests that need released data read
# it from SAM_TEST_DATA_DIR (the configured data directory, set by the
# Makefile) and skip if it is missing. Exit status is nonzero if any program
# fails or crashes.
set -u

mpirun="$1"
shift

rows=()
total_passed=0
total_failed=0
total_skipped=0
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

  # each program ends with "Summary: <p> passed, <f> failed, <s> skipped"
  summary="$(grep '^Summary:' "$log" | tail -1)"
  rm -rf "$data" "$log"

  if [[ -z "$summary" ]]; then
    passed="-"; failed="-"; skipped="-"; result="CRASHED (exit $rc)"; all_ok=0
  else
    passed="$(echo "$summary" | sed 's/^Summary: \([0-9]*\) passed.*/\1/')"
    failed="$(echo "$summary" | sed 's/.*, \([0-9]*\) failed.*/\1/')"
    skipped="$(echo "$summary" | sed -n 's/.*, \([0-9]*\) skipped.*/\1/p')"
    skipped="${skipped:-0}"
    total_passed=$((total_passed + passed))
    total_failed=$((total_failed + failed))
    total_skipped=$((total_skipped + skipped))
    if [[ $rc -ne 0 || $failed -ne 0 ]]; then
      result="FAIL"; all_ok=0
    elif [[ $skipped -ne 0 ]]; then
      result="ok (some skipped)"
    else
      result="ok"
    fi
  fi
  total_time=$((total_time + elapsed))
  rows+=("$(printf '%-46s %7s %7s %8s %9s   %s' "$name" "$passed" "$failed" "$skipped" "$elapsed" "$result")")
done

line="$(printf '%.0s-' {1..98})"
echo
echo "SAM test summary"
echo "$line"
printf '%-46s %7s %7s %8s %9s   %s\n' "Test program" "Passed" "Failed" "Skipped" "Time (s)" "Result"
echo "$line"
for row in "${rows[@]}"; do echo "$row"; done
echo "$line"
if [[ $all_ok -eq 1 ]]; then overall="PASS"; else overall="FAIL"; fi
printf '%-46s %7s %7s %8s %9s   %s\n' "Total ($# programs)" "$total_passed" "$total_failed" "$total_skipped" "$total_time" "$overall"
echo "$line"
if [[ $total_skipped -ne 0 ]]; then
  echo "Skipped cases need data that is not in the data directory; see their [SKIP] lines."
fi

[[ $all_ok -eq 1 ]]
