#pragma once

// HOutilities was split into three cohesive headers. This umbrella keeps the
// old include path working; new code may include the narrower headers directly.
//
//   HObasics.h  - analytic HO primitives (valence shell, oscillator lengths,
//                 boson/r/p reduced matrix elements)
//   HOorbits.h  - single-particle labeling & orbit indexing (qn_t,
//                 pack_mscheme_state, Table_nljmjmt, Orbit, OrbitsMap)
//   HOmes.h     - matrix-element storage (OBME_scalar, TwBME_scalar,
//                 TwBME_general) and the m-scheme two-body cache
#include <HObasics.h>
#include <HOorbits.h>
#include <HOmes.h>
