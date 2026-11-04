#!/usr/bin/env bash
# SAM configuration — resolves prerequisites (u3lib, u4lib, untou3, Spectra, MPI)
# and writes config/config.generated.mk. Run via `make config` (or directly).
#
# Usage:
#   ./configure.sh [--u3lib=PATH] [--u4lib=PATH] [--untou3=PATH]
#                  [--spectra=PATH] [--cxx=PATH] [--no-mpi]
#                  [--data-dir=PATH] [--yes|-y]
set -uo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
GEN="$ROOT/config/config.generated.mk"

say()  { printf '\033[1;34m==>\033[0m %s\n' "$*"; }
warn() { printf '\033[1;33mnote:\033[0m %s\n' "$*" >&2; }
die()  { printf '\033[1;31merror:\033[0m %s\n' "$*" >&2; exit 1; }

YES=0
OPT_u3lib=""
OPT_u4lib=""
OPT_untou3=""
OPT_spectra=""
OPT_cxx=""
OPT_data_dir=""
NO_MPI=0

for arg in "$@"; do
  case "$arg" in
    --yes|-y) YES=1 ;;
    --u3lib=*) OPT_u3lib="${arg#*=}" ;;
    --u4lib=*) OPT_u4lib="${arg#*=}" ;;
    --untou3=*) OPT_untou3="${arg#*=}" ;;
    --spectra=*) OPT_spectra="${arg#*=}" ;;
    --cxx=*) OPT_cxx="${arg#*=}" ;;
    --data-dir=*) OPT_data_dir="${arg#*=}" ;;
    --no-mpi) NO_MPI=1 ;;
    --help|-h)
      cat << 'HELP'
Usage: ./configure.sh [OPTIONS]

Options:
  --u3lib=PATH      Path to u3lib root [default: ../u3lib]
  --u4lib=PATH      Path to u4lib root [default: ../u4lib]
  --untou3=PATH     Path to untou3 root [default: ../untou3]
  --spectra=PATH    Path to Spectra include directory
  --cxx=PATH        C++ compiler to use [default: mpicxx or compiler from u3lib]
  --no-mpi          Disable MPI (compile in serial / OpenMP mode)
  --data-dir=PATH   Path to SAM data directory [default: ./data]
  --yes, -y         Non-interactive mode
HELP
      exit 0 ;;
    *) echo "unknown option: $arg" >&2; exit 2 ;;
  esac
done

say "Configuring SAM (DeepSANE)"

# 1. Resolve and validate u3lib
U3LIB_DIR="${OPT_u3lib:-${U3LIB_DIR:-}}"
if [[ -z "$U3LIB_DIR" ]]; then
  if [[ -f "$ROOT/../u3lib/include/u3lib.h" ]]; then
    U3LIB_DIR="$(cd "$ROOT/../u3lib" 2>/dev/null && pwd || true)"
  fi
fi

if [[ -z "$U3LIB_DIR" || ! -f "$U3LIB_DIR/include/u3lib.h" ]]; then
  die "u3lib was not found!
SAM requires u3lib as a foundation prerequisite.
Please ensure u3lib is cloned and configured:
  git clone https://github.com/Deep-SANE/u3lib.git ../u3lib
  cd ../u3lib && make config && make"
fi

# 2. Resolve and validate u4lib
U4LIB_DIR="${OPT_u4lib:-${U4LIB_DIR:-}}"
if [[ -z "$U4LIB_DIR" ]]; then
  if [[ -f "$ROOT/../u4lib/include/u4lib.h" ]]; then
    U4LIB_DIR="$(cd "$ROOT/../u4lib" 2>/dev/null && pwd || true)"
  fi
fi

if [[ -z "$U4LIB_DIR" || ! -f "$U4LIB_DIR/include/u4lib.h" ]]; then
  die "u4lib was not found!
SAM requires u4lib as a foundation prerequisite.
Please ensure u4lib is cloned and configured:
  git clone https://github.com/Deep-SANE/u4lib.git ../u4lib
  cd ../u4lib && make config"
fi

# 3. Resolve and validate untou3
UNTOU3_REPO="https://github.com/Deep-SANE/untou3.git"
UNTOU3_DIR="${OPT_untou3:-${UNTOU3_DIR:-}}"
if [[ -z "$UNTOU3_DIR" ]]; then
  if [[ -f "$ROOT/../untou3/include/UNtoU3.h" ]]; then
    UNTOU3_DIR="$(cd "$ROOT/../untou3" 2>/dev/null && pwd || true)"
  else
    UNTOU3_DIR="$(cd "$ROOT/.." && pwd)/untou3"
  fi
fi

if [[ ! -f "$UNTOU3_DIR/include/UNtoU3.h" ]]; then
  command -v git >/dev/null 2>&1 || die "git required to fetch untou3"
  say "untou3 not found; cloning -> $UNTOU3_DIR"
  git clone "$UNTOU3_REPO" "$UNTOU3_DIR" >/dev/null 2>&1 || die "untou3 clone failed ($UNTOU3_REPO).
Please clone it manually:
  git clone $UNTOU3_REPO \"$UNTOU3_DIR\""
fi

if [[ ! -f "$UNTOU3_DIR/include/UNtoU3.h" ]]; then
  die "untou3 was not found at $UNTOU3_DIR (missing include/UNtoU3.h)"
fi

say "Found u3lib at $U3LIB_DIR"
say "Found u4lib at $U4LIB_DIR"
say "Found untou3 at $UNTOU3_DIR"

# Read inherited configurations from u4lib or u3lib
read_var() {
  local var="$1"
  local file="$2"
  if [[ -f "$file" ]]; then
    grep "^${var}[[:space:]]*:=" "$file" 2>/dev/null | head -1 | sed "s/^${var}[[:space:]]*:=[[:space:]]*//" || true
  fi
}

U4_GEN="$U4LIB_DIR/config/config.generated.mk"
U3_GEN="$U3LIB_DIR/config/config.generated.mk"
SRC_GEN="$U4_GEN"
if [[ ! -f "$SRC_GEN" ]]; then
  SRC_GEN="$U3_GEN"
fi

if [[ ! -f "$SRC_GEN" ]]; then
  die "Prerequisite libraries must be configured first! Neither $U4_GEN nor $U3_GEN exists."
fi

INHERIT_CXX="$(read_var CXX "$SRC_GEN")"
INHERIT_CXXFLAGS="$(read_var CXXFLAGS "$SRC_GEN")"
BOOST_ROOT="$(read_var BOOST_ROOT "$SRC_GEN")"
EIGEN_DIR="$(read_var EIGEN_DIR "$SRC_GEN")"
FMT_DIR="$(read_var FMT_DIR "$SRC_GEN")"
WIGNERNJ_DIR="$(read_var WIGNERNJ_DIR "$SRC_GEN")"
USE_WIGNERNJ="$(read_var USE_WIGNERNJ "$SRC_GEN")"
KRANS_DIR="$(read_var KRANS_DIR "$SRC_GEN")"
USE_KRANS="$(read_var USE_KRANS "$SRC_GEN")"
NULLSPACE_SOLVER="$(read_var NULLSPACE_SOLVER "$SRC_GEN")"
USE_OPENMP="$(read_var USE_OPENMP "$SRC_GEN")"
OMP_CXXFLAGS="$(read_var OMP_CXXFLAGS "$SRC_GEN")"
OMP_LDFLAGS="$(read_var OMP_LDFLAGS "$SRC_GEN")"

# 4. MPI & Toolchain
#
# The underlying compiler always follows the prerequisite libraries (u3lib /
# u4lib) -- never Apple clang. When MPI is enabled we keep the mpicxx wrapper as
# CXX but point it at that inherited compiler through OMPI_CXX / MPICH_CXX, so
# the OpenMP flags (-fopenmp) and ABI inherited from u3lib/u4lib stay valid.
USE_MPI=0
CXX=""
MPI_UNDERLYING_CXX=""

# Compiler inherited from the prerequisites (overridable with --cxx).
UNDERLYING_CXX="${OPT_cxx:-$INHERIT_CXX}"
if [[ -z "$UNDERLYING_CXX" ]]; then
  die "Could not determine a C++ compiler from the prerequisite libraries.
Configure u3lib/u4lib first (they record CXX in config/config.generated.mk),
or pass --cxx=PATH."
fi

if [[ "$UNDERLYING_CXX" == *clang* ]]; then
  warn "Prerequisite compiler looks like clang ($UNDERLYING_CXX); u3lib/u4lib use GCC with -fopenmp. If the build rejects -fopenmp, reconfigure u3lib/u4lib with GCC."
fi

if [[ $NO_MPI -eq 0 ]]; then
  # Look for mpicxx
  MPICXX_BIN=""
  if command -v mpicxx >/dev/null 2>&1; then
    MPICXX_BIN="$(command -v mpicxx)"
  elif [[ -x "/opt/homebrew/bin/mpicxx" ]]; then
    MPICXX_BIN="/opt/homebrew/bin/mpicxx"
  fi

  if [[ -n "$MPICXX_BIN" ]]; then
    CXX="$MPICXX_BIN"
    MPI_UNDERLYING_CXX="$UNDERLYING_CXX"   # make mpicxx wrap the inherited GCC
    USE_MPI=1
    say "Using MPI compiler: $CXX (wrapping $MPI_UNDERLYING_CXX)"
  else
    warn "mpicxx not found; configuring in serial / OpenMP mode."
    CXX="$UNDERLYING_CXX"
    USE_MPI=0
  fi
else
  say "MPI explicitly disabled (--no-mpi)."
  CXX="$UNDERLYING_CXX"
  USE_MPI=0
fi

# 5. Resolve Spectra (required header-only eigensolver)
SPECTRA_DIR="${OPT_spectra:-}"

valid_spectra() {
  local p="$1"
  [[ -n "$p" && -f "$p/Spectra/SymEigsSolver.h" ]]
}

if ! valid_spectra "$SPECTRA_DIR"; then
  CANDIDATES=(
    "$ROOT/external-libs/spectra/include"
    "/opt/homebrew/include"
    "/opt/homebrew/opt/spectra/include"
    "/usr/local/include"
    "/usr/include"
    "/opt/local/include"
    "$HOME/.local/include"
  )

  if command -v pkg-config >/dev/null 2>&1; then
    PKGCFG_SPECTRA="$(pkg-config --variable=includedir spectra 2>/dev/null || true)"
    if [[ -n "$PKGCFG_SPECTRA" ]]; then
      CANDIDATES=( "$PKGCFG_SPECTRA" "${CANDIDATES[@]}" )
    fi
  fi

  if [[ -d "/opt/homebrew/Cellar/spectra" ]]; then
    for c in /opt/homebrew/Cellar/spectra/*/include; do
      if [[ -d "$c" ]]; then CANDIDATES+=( "$c" ); fi
    done
  fi

  for cand in "${CANDIDATES[@]}"; do
    if valid_spectra "$cand"; then
      SPECTRA_DIR="$cand"
      break
    fi
  done
fi

if [[ "$SPECTRA_DIR" == "fetch" ]] || (! valid_spectra "$SPECTRA_DIR" && [[ $YES -eq 1 ]]); then
  if command -v git >/dev/null 2>&1; then
    EXT="$ROOT/external-libs"
    mkdir -p "$EXT"
    say "Fetching Spectra (header-only) -> external-libs/spectra"
    rm -rf "$EXT/spectra"
    if git clone --depth 1 --branch "v1.0.1" https://github.com/yixuan/spectra "$EXT/spectra" >/dev/null 2>&1; then
      SPECTRA_DIR="$EXT/spectra/include"
    fi
  fi
fi

if valid_spectra "$SPECTRA_DIR"; then
  say "Found Spectra at $SPECTRA_DIR"
else
  die "Spectra eigensolver library was not found!
SAM requires Spectra (https://spectralib.org) for large-scale symmetric matrix diagonalization.
Please install Spectra:
  macOS:  brew install spectra
  Ubuntu: sudo apt-get install libspectra-dev
  Manual: git clone https://github.com/yixuan/spectra external-libs/spectra
Or specify:
  ./configure.sh --spectra=/path/to/include"
fi

# 6. SAM Data Directory
SAM_DATA_DIR="${OPT_data_dir:-$ROOT/data}"
if [[ -d "$SAM_DATA_DIR" ]]; then
  say "Using SAM data directory: $SAM_DATA_DIR"
else
  warn "Data directory $SAM_DATA_DIR does not exist yet."
fi

# 7. Write config/config.generated.mk
# When MPI is on, make the mpicxx wrapper invoke the inherited (GCC) compiler.
MPI_WRAP_BLOCK=""
if [[ -n "$MPI_UNDERLYING_CXX" ]]; then
  MPI_WRAP_BLOCK="OMPI_CXX := $MPI_UNDERLYING_CXX
MPICH_CXX := $MPI_UNDERLYING_CXX"
fi

mkdir -p "$ROOT/config"
cat > "$GEN" <<EOF
# Generated by config/configure.sh on $(date). Re-run \`make config\` to change.
SAM_ROOT := $ROOT
U3LIB_DIR := $U3LIB_DIR
U4LIB_DIR := $U4LIB_DIR
UNTOU3_DIR := $UNTOU3_DIR

CXX := $CXX
${MPI_WRAP_BLOCK}
CXXFLAGS := ${INHERIT_CXXFLAGS:--std=c++20 -O3}
LDFLAGS :=

USE_MPI := $USE_MPI
USE_OPENMP := ${USE_OPENMP:-1}
OMP_CXXFLAGS := ${OMP_CXXFLAGS:--fopenmp}
OMP_LDFLAGS := ${OMP_LDFLAGS:--fopenmp}

BOOST_ROOT := $BOOST_ROOT
EIGEN_DIR := $EIGEN_DIR
FMT_DIR := $FMT_DIR
SPECTRA_DIR := $SPECTRA_DIR

USE_WIGNERNJ := ${USE_WIGNERNJ:-1}
WIGNERNJ_DIR := $WIGNERNJ_DIR

USE_KRANS := ${USE_KRANS:-1}
KRANS_DIR := $KRANS_DIR
NULLSPACE_SOLVER := ${NULLSPACE_SOLVER:-krans}

SAM_DATA_DIR := $SAM_DATA_DIR
EOF

say "Configuration written to $GEN"
say "Run 'make' to build SAM library and executables."
