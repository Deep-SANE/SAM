#!/usr/bin/env bash
# Download and verify SAM's released data (operator expansions).
# Run through make:  make data [OPS="..."] [FORCE=1]   |   make data-verify
#
#   fetch.sh get [OPERATOR...]   download files (all, or only these operators)
#   fetch.sh verify              check files on disk against the release
#   fetch.sh list                show what the release contains
#
# Environment:
#   SAM_DATA_DIR   where the data lives (set by make from `make config`)
#   SAM_DATA_URL   download location [default: the GitHub release in manifest]
#   FORCE=1        replace files that differ from the release (files extended
#                  locally to a larger maxShell are never replaced)
#
# The README and manifest are also copied into <data dir>/data_release/, so the
# list of released files sits next to the data (also when data is on /work).
set -uo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
MANIFEST="$HERE/manifest.tsv"
REPO_URL="https://github.com/Deep-SANE/SAM"

say()  { printf '\033[1;34m==>\033[0m %s\n' "$*"; }
warn() { printf '\033[1;33mnote:\033[0m %s\n' "$*" >&2; }
die()  { printf '\033[1;31merror:\033[0m %s\n' "$*" >&2; exit 1; }

[[ -f "$MANIFEST" ]] || die "missing $MANIFEST"
DATA="${SAM_DATA_DIR:-}"
RELEASE="$(sed -n 's/^# release: *//p' "$MANIFEST")"
URL="${SAM_DATA_URL:-$REPO_URL/releases/download/$RELEASE}"
URL="${URL// /%20}"   # a local mirror path may contain spaces

# sha256 of a file (Linux: sha256sum, macOS: shasum)
if command -v sha256sum >/dev/null 2>&1; then
  sha() { sha256sum "$1" | cut -d' ' -f1; }
elif command -v shasum >/dev/null 2>&1; then
  sha() { shasum -a 256 "$1" | cut -d' ' -f1; }
else
  die "need sha256sum or shasum to check the files"
fi

download() {   # download URL to FILE
  if command -v curl >/dev/null 2>&1; then
    curl -fsSL --retry 3 -o "$2" "$1"
  elif command -v wget >/dev/null 2>&1; then
    wget -q -O "$2" "$1"
  else
    die "need curl or wget to download data"
  fi
}

rows() { grep -v '^#' "$MANIFEST" | tail -n +2; }

maxshell_of() { od -An -tu1 -N1 "$1" | tr -d ' '; }   # first byte of the file

# copy the catalog (README + manifest) next to the data
publish_catalog() {
  mkdir -p "$DATA/data_release" &&
    cp -f "$HERE/README.md" "$MANIFEST" "$DATA/data_release/" ||
    warn "could not copy the release catalog into $DATA/data_release"
}

wanted() {   # is OPERATOR in the requested list (empty list = all)?
  local op="$1"; shift
  [[ $# -eq 0 ]] && return 0
  for x in "$@"; do [[ "$x" == "$op" ]] && return 0; done
  return 1
}

cmd_list() {
  printf '%-18s %-22s %8s %10s %10s\n' body operator maxShell tensors bytes
  rows | while IFS=$'\t' read -r asset path bytes sum body op hw ms nt; do
    printf '%-18s %-22s %8s %10s %10s\n' "$body" "$op" "$ms" "$nt" "$bytes"
  done
}

cmd_get() {
  [[ -n "$DATA" ]] || die "SAM_DATA_DIR is not set; run 'make config', then use 'make data'."
  mkdir -p "$DATA" || die "cannot create $DATA"
  say "Release $RELEASE -> $DATA"
  local n_new=0 n_ok=0 n_kept=0 n_fail=0 matched=0
  while IFS=$'\t' read -r asset path bytes sum body op hw ms nt; do
    wanted "$op" "$@" || continue
    matched=$((matched + 1))
    local target="$DATA/$path"
    if [[ -f "$target" ]]; then
      if [[ "$(sha "$target")" == "$sum" ]]; then
        n_ok=$((n_ok + 1)); continue
      elif [[ "$(maxshell_of "$target")" -gt "$ms" ]]; then
        # a costly local extension: never replaced, not even with FORCE=1
        warn "$path was extended locally (maxShell $(maxshell_of "$target") > $ms); left untouched (delete it by hand to get the released one)"
        n_kept=$((n_kept + 1)); continue
      elif [[ "${FORCE:-0}" != 1 ]]; then
        warn "$path differs from the release; left untouched (FORCE=1 replaces it)"
        n_kept=$((n_kept + 1)); continue
      fi
    fi
    mkdir -p "$(dirname "$target")"
    local tmp="$target.part"
    printf '    %-60s ' "$path"
    if download "$URL/$asset" "$tmp.gz" && gzip -dc "$tmp.gz" > "$tmp" \
       && [[ "$(sha "$tmp")" == "$sum" ]]; then
      chmod a-w "$tmp"                 # protect against accidental edits
      mv -f "$tmp" "$target"           # a rename also replaces a read-only file
      rm -f "$tmp.gz"
      echo "ok"; n_new=$((n_new + 1))
    else
      rm -f "$tmp" "$tmp.gz"
      echo "FAILED"; n_fail=$((n_fail + 1))
    fi
  done < <(rows)

  [[ $# -gt 0 && $matched -eq 0 ]] && die "no released file matches: $*  (see 'make data-list')"
  publish_catalog
  say "downloaded $n_new, already present $n_ok, left untouched $n_kept, failed $n_fail"
  [[ $n_fail -eq 0 ]]
}

cmd_verify() {
  [[ -n "$DATA" ]] || die "SAM_DATA_DIR is not set; run 'make config', then use 'make data'."
  say "Checking $DATA against release $RELEASE"
  local n_ok=0 n_missing=0 n_bad=0 n_ext=0
  while IFS=$'\t' read -r asset path bytes sum body op hw ms nt; do
    local target="$DATA/$path"
    if [[ ! -f "$target" ]]; then
      n_missing=$((n_missing + 1))
    elif [[ "$(sha "$target")" == "$sum" ]]; then
      n_ok=$((n_ok + 1))
    elif [[ "$(maxshell_of "$target")" -gt "$ms" ]]; then
      # a deliberate extension to larger maxShell (expand_exist_MPI) is not
      # an error; we just cannot check it against the release any more
      echo "    EXTENDED  $path (maxShell $(maxshell_of "$target") > released $ms)"
      n_ext=$((n_ext + 1))
    else
      echo "    MODIFIED  $path"; n_bad=$((n_bad + 1))
    fi
  done < <(rows)
  say "ok $n_ok, extended locally $n_ext, modified $n_bad, not downloaded $n_missing"
  if [[ $n_bad -gt 0 ]]; then
    warn "modified files give wrong results. Restore them with: make data FORCE=1"
    return 1
  fi
}

case "${1:-}" in
  get)    shift; cmd_get "$@" ;;
  verify) cmd_verify ;;
  list)   cmd_list ;;
  *)      sed -n '2,13p' "$0"; exit 2 ;;
esac
