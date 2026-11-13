#!/usr/bin/env bash
# SAM configuration — resolves prerequisites (u3lib, u4lib, untou3, Spectra, MPI)
# and writes config/config.generated.mk. Run via `make config` (or directly).
#
# Usage:
#   ./configure.sh [--u3lib=PATH] [--u4lib=PATH] [--untou3=PATH]
#                  [--spectra=PATH] [--cxx=PATH] [--mpicxx=PATH]
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
OPT_mpicxx=""

for arg in "$@"; do
  case "$arg" in
    --yes|-y) YES=1 ;;
    --u3lib=*) OPT_u3lib="${arg#*=}" ;;
    --u4lib=*) OPT_u4lib="${arg#*=}" ;;
    --untou3=*) OPT_untou3="${arg#*=}" ;;
    --spectra=*) OPT_spectra="${arg#*=}" ;;
    --cxx=*) OPT_cxx="${arg#*=}" ;;
    --data-dir=*) OPT_data_dir="${arg#*=}" ;;
    --mpicxx=*) OPT_mpicxx="${arg#*=}" ;;
    --help|-h)
      cat << 'HELP'
Usage: ./configure.sh [OPTIONS]

Options:
  --u3lib=PATH      Path to u3lib root [default: ../u3lib]
  --u4lib=PATH      Path to u4lib root [default: ../u4lib]
  --untou3=PATH     Path to untou3 root [default: ../untou3]
  --spectra=PATH    Spectra include directory or source root; "fetch"
                    downloads it [default: found, else ask]
  --cxx=PATH        C++ compiler wrapped by MPI [default: compiler from u3lib/u4lib]
  --mpicxx=PATH     MPI compiler wrapper (MPI is required) [default: mpicxx]
  --data-dir=PATH   Where SAM keeps its data [default: SAM/data]. Another
                    path makes SAM/data a link to it (e.g. on /work).
  --yes, -y         Non-interactive mode (accept all defaults)
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

# 4. MPI & Toolchain (MPI is required)
#
# The underlying compiler always follows the prerequisite libraries (u3lib /
# u4lib) -- never Apple clang. We use the MPI wrapper (mpicxx) as CXX but point
# it at that inherited compiler through OMPI_CXX / MPICH_CXX, so the OpenMP
# flags (-fopenmp) and ABI inherited from u3lib/u4lib stay valid.
USE_MPI=1

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

# Find the MPI compiler wrapper.
MPICXX_BIN="$OPT_mpicxx"
if [[ -z "$MPICXX_BIN" ]]; then
  if command -v mpicxx >/dev/null 2>&1; then
    MPICXX_BIN="$(command -v mpicxx)"
  elif [[ -x "/opt/homebrew/bin/mpicxx" ]]; then
    MPICXX_BIN="/opt/homebrew/bin/mpicxx"
  fi
fi

if [[ -z "$MPICXX_BIN" || ! -x "$MPICXX_BIN" ]]; then
  die "MPI was not found, and SAM requires it (mpicxx is missing).
Please install an MPI library:
  macOS:    brew install open-mpi
  Ubuntu:   sudo apt-get install libopenmpi-dev openmpi-bin
  Cluster:  load the MPI module, e.g. 'module load openmpi' or 'module load mpich'
Or point to the compiler wrapper:
  ./config/configure.sh --mpicxx=/path/to/mpicxx"
fi

# Check that the wrapper can compile and link an MPI + OpenMP program with the
# inherited compiler, so problems show up now and not halfway through `make`.
MPI_TEST_DIR="$(mktemp -d)"
cat > "$MPI_TEST_DIR/mpi_test.cpp" << 'MPITEST'
#include <mpi.h>
int main(int argc, char** argv)
{
  MPI_Init(&argc, &argv);
  int rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Finalize();
  return rank;
}
MPITEST
if ! OMPI_CXX="$UNDERLYING_CXX" MPICH_CXX="$UNDERLYING_CXX" \
     "$MPICXX_BIN" -std=c++20 ${OMP_CXXFLAGS:--fopenmp} \
     "$MPI_TEST_DIR/mpi_test.cpp" -o "$MPI_TEST_DIR/mpi_test" \
     > "$MPI_TEST_DIR/log" 2>&1; then
  sed 's/^/    /' "$MPI_TEST_DIR/log" | tail -15 >&2
  rm -rf "$MPI_TEST_DIR"
  die "$MPICXX_BIN (wrapping $UNDERLYING_CXX) cannot build a test MPI program.
Make sure the MPI library was built for this compiler, or choose another one
with --mpicxx=PATH or --cxx=PATH."
fi
rm -rf "$MPI_TEST_DIR"

CXX="$MPICXX_BIN"
MPI_UNDERLYING_CXX="$UNDERLYING_CXX"   # make mpicxx wrap the inherited GCC
say "Using MPI compiler: $CXX (wrapping $MPI_UNDERLYING_CXX); test program OK"

# 5. Resolve Spectra (required header-only eigensolver)
#
# Order: --spectra=PATH, then known locations, then (interactive) ask for a
# path or download it, (--yes) download it. PATH may be the include directory
# or the Spectra source root; --spectra=fetch downloads it.
SPECTRA_VERSION="v1.0.1"

valid_spectra() {
  local p="$1"
  [[ -n "$p" && -f "$p/Spectra/SymEigsSolver.h" ]]
}

# the include directory for PATH (include dir or source root), or nothing
spectra_include() {
  local p="${1%/}"
  p="${p/#\~/$HOME}"
  if valid_spectra "$p"; then echo "$p"
  elif valid_spectra "$p/include"; then echo "$p/include"
  fi
}

fetch_spectra() {
  command -v git >/dev/null 2>&1 || die "git is needed to download Spectra"
  local ext="$ROOT/external-libs"
  mkdir -p "$ext" || die "cannot create $ext"
  say "Downloading Spectra $SPECTRA_VERSION (header-only) -> external-libs/spectra"
  rm -rf "$ext/spectra"
  git clone --depth 1 --branch "$SPECTRA_VERSION" \
      https://github.com/yixuan/spectra "$ext/spectra" >/dev/null 2>&1 ||
    die "could not download Spectra from https://github.com/yixuan/spectra
(no network here? On a cluster, run make config on a login node, or clone it
elsewhere and pass ./configure.sh --spectra=/path/to/spectra)"
  SPECTRA_DIR="$ext/spectra/include"
}

SPECTRA_DIR=""
if [[ "$OPT_spectra" == "fetch" ]]; then
  fetch_spectra
elif [[ -n "$OPT_spectra" ]]; then
  SPECTRA_DIR="$(spectra_include "$OPT_spectra")"
  [[ -n "$SPECTRA_DIR" ]] ||
    die "--spectra=$OPT_spectra: no Spectra/SymEigsSolver.h there or in its include/"
else
  CANDIDATES=(
    "$ROOT/external-libs/spectra/include"
    "$ROOT/../spectra/include"            # DeepSANE/spectra next to SAM
    "${SPECTRA_ROOT:+$SPECTRA_ROOT/include}"
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

  if [[ -z "$SPECTRA_DIR" ]]; then
    if [[ $YES -eq 0 && -t 0 ]]; then
      say "Spectra (https://spectralib.org, header-only eigensolver) was not found."
      echo "    Give the path to an existing copy (its include directory or source"
      echo "    root), or press Enter to download $SPECTRA_VERSION into SAM/external-libs/spectra."
      while [[ -z "$SPECTRA_DIR" ]]; do
        read -r -p "    Spectra path [download]: " ans || ans=""
        if [[ -z "$ans" ]]; then
          fetch_spectra
        else
          SPECTRA_DIR="$(spectra_include "$ans")"
          [[ -n "$SPECTRA_DIR" ]] ||
            warn "no Spectra/SymEigsSolver.h under $ans or $ans/include; try again"
        fi
      done
    elif [[ $YES -eq 1 ]]; then
      fetch_spectra
    else
      die "Spectra eigensolver library was not found!
SAM requires Spectra (https://spectralib.org) for large-scale symmetric matrix diagonalization.
Run make config in a terminal to be asked for its path or to download it, or:
  ./configure.sh --spectra=/path/to/spectra    an existing copy
  ./configure.sh --spectra=fetch               download it into external-libs/spectra"
    fi
  fi
fi

valid_spectra "$SPECTRA_DIR" || die "Spectra not usable at $SPECTRA_DIR"
say "Found Spectra at $SPECTRA_DIR"

# 6. SAM data directory
#
# Data lives in SAM/data by default. On a cluster the user usually wants it on
# a work file system instead; then the data goes there and SAM/data becomes a
# symbolic link to it, so scripts can keep using SAM/data. The real (resolved)
# path is what gets built into the binaries, so compute nodes never need to go
# through the home directory. SAM_DATA_DIR still overrides it at run time.
LOCAL_DATA="$ROOT/data"

# current choice: the link target if SAM/data is already a link, else SAM/data
if [[ -L "$LOCAL_DATA" ]]; then
  CURRENT_DATA="$(cd "$LOCAL_DATA" 2>/dev/null && pwd -P || readlink "$LOCAL_DATA")"
else
  CURRENT_DATA="$LOCAL_DATA"
fi

DATA_IN="$OPT_data_dir"
if [[ -z "$DATA_IN" && $YES -eq 0 && -t 0 ]]; then
  say "Where should SAM keep its data (operator expansions, rmes, hws, ...)?"
  echo "    Press Enter for the default, or give another path (e.g. on /work)."
  read -r -p "    data directory [$CURRENT_DATA]: " DATA_IN
fi
DATA_IN="${DATA_IN:-$CURRENT_DATA}"
DATA_IN="${DATA_IN/#\~/$HOME}"

[[ "$DATA_IN" == *[[:space:]]* ]] && die "The data directory path contains a space,
which the build cannot pass to the compiler: $DATA_IN
Please choose a path without spaces."

CREATED_DATA=0
[[ -e "$DATA_IN" ]] || CREATED_DATA=1
mkdir -p "$DATA_IN" 2>/dev/null || die "Cannot create the data directory $DATA_IN"
SAM_DATA_DIR="$(cd "$DATA_IN" && pwd -P)"

# is the chosen directory SAM/data itself (a real folder, not a link)?
IS_LOCAL=0
if [[ -d "$LOCAL_DATA" && ! -L "$LOCAL_DATA" ]]; then
  [[ "$(cd "$LOCAL_DATA" && pwd -P)" == "$SAM_DATA_DIR" ]] && IS_LOCAL=1
fi

if [[ $IS_LOCAL -eq 1 ]]; then
  say "Using SAM data directory: $SAM_DATA_DIR"
else
  if [[ -L "$LOCAL_DATA" ]]; then
    OLD_TARGET="$(readlink "$LOCAL_DATA")"
    if [[ "$(cd "$LOCAL_DATA" 2>/dev/null && pwd -P)" != "$SAM_DATA_DIR" ]]; then
      if [[ $YES -eq 0 && -t 0 ]]; then
        read -r -p "    SAM/data points to $OLD_TARGET. Re-point it to $SAM_DATA_DIR? [y/N] " ans
        if [[ "$ans" != [yY]* ]]; then
          [[ $CREATED_DATA -eq 1 ]] && rmdir "$SAM_DATA_DIR" 2>/dev/null
          die "Kept SAM/data -> $OLD_TARGET. Nothing was changed."
        fi
      else
        warn "re-pointing SAM/data from $OLD_TARGET to $SAM_DATA_DIR"
      fi
      rm "$LOCAL_DATA" && ln -s "$SAM_DATA_DIR" "$LOCAL_DATA" \
        || die "Could not re-point the SAM/data link"
    fi
  elif [[ -d "$LOCAL_DATA" ]]; then
    # a real folder: replace it only if it holds nothing but Finder cruft
    if [[ -z "$(ls -A "$LOCAL_DATA" | grep -v '^\.DS_Store$')" ]]; then
      rm -f "$LOCAL_DATA/.DS_Store"
      rmdir "$LOCAL_DATA" || die "Could not remove the empty folder SAM/data"
      ln -s "$SAM_DATA_DIR" "$LOCAL_DATA" || die "Could not create the SAM/data link"
    else
      [[ $CREATED_DATA -eq 1 ]] && rmdir "$SAM_DATA_DIR" 2>/dev/null
      die "SAM/data is a real folder that already holds files, so it cannot
become a link to $SAM_DATA_DIR. Nothing was changed. Either keep the default
(run 'make config' and press Enter), or move the files and try again:
  mv \"$LOCAL_DATA\"/* \"$SAM_DATA_DIR\"/ && rmdir \"$LOCAL_DATA\"
  make config"
    fi
  elif [[ -e "$LOCAL_DATA" ]]; then
    die "SAM/data exists but is not a folder or a link; please remove it first."
  else
    ln -s "$SAM_DATA_DIR" "$LOCAL_DATA" || die "Could not create the SAM/data link"
  fi
  say "Using SAM data directory: $SAM_DATA_DIR (SAM/data -> $SAM_DATA_DIR)"
fi

# 7. Write config/config.generated.mk
# Make the mpicxx wrapper invoke the inherited (GCC) compiler.
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
