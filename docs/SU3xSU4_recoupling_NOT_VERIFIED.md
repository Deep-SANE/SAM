# Two‑Body Interaction Recoupling in the SU(3)×SU(4) Scheme

**Status: DRAFT for verification.** This generalizes §4 / §4.1 and Appendix A.3.2 of
`SU3NCSM.pdf` (the SU(3)×SU(2) notes) to the SU(3)×SU(4) (Wigner supermultiplet) scheme.
Everything that is *unchanged* from the SU(2) notes is stated as such; everything that is
*new* (the SU(4) coefficients/phases) is written symbolically and flagged **[VERIFY]** where
the exact convention must be confirmed against u4lib.

Notation: `[n] ≡ (n,0)` is the U(3) irrep of a creation operator in shell `n`;
`[0n] ≡ (0,n)` for an annihilation. `(1,0,0)` and `(0,0,1)` are the SU(4) fundamental and its
conjugate. `W = (w1,w2,w3)` denotes a coupled SU(4) irrep (it replaces the SU(2) spin `S`).
`Πx ≡ √(dim x)` (so `Π_S = √(2S+1)` becomes `Π_W = √(dim W)`).

---

## 1. Paired‑scheme interaction (analog of Eq. 4.2)

A general two‑body interaction is decomposed into SU(3)×SU(4) tensors in the **paired** form
(two creations coupled together, two annihilations coupled together):

$$
V \;=\; \sum \; \sigma \;
\Big[\,
\big\{ a^{\dagger[n_a](1,0,0)} \times a^{\dagger[n_b](1,0,0)} \big\}^{\rho_f(\lambda_f\mu_f)\,W_f}
\times
\big\{ \tilde a^{[0 n_d](0,0,1)} \times \tilde a^{[0 n_c](0,0,1)} \big\}^{\rho_i(\lambda_i\mu_i)\,W_i}
\Big]^{\rho_0(\lambda_0\mu_0)\,\kappa_0 L_0\,W_0}
$$

where the strength `σ` is the closed‑form coefficient (the paired‑scheme analog of
`app/archived/CoMNumberOp_SAM_Analytical_NoPermutation.cpp`), obtained by contracting the
**coupled** two‑body reduced matrix element `⟨(a b)JT ‖ V ‖ (c d)JT⟩` (a lookup into
`TwBME_scalar::me_map`, i.e. `getME(a,b,c,d,J)`) with the U(3)→SO(3) and U(4)→SU(2)×SU(2)
coupling coefficients and the SU(2) 9‑j (`racah9`), exactly as in the archived file.

Because the two creations sit in one subsystem and the two annihilations in the other, this
form **cannot** be used directly with the SU(3)×SU(4) NCSM many‑body basis, which couples the
single‑shell tensors in **increasing shell order, with all operators of a shell coupled
first**. So each paired tensor must be **recoupled** into shell‑ordered form (and its strength
distributed accordingly) before it is written to the tensor file. That recoupling is this note.

---

## 2. Generalization principle (the one idea)

The SU(3)×SU(2) recoupling factorizes as

$$
\text{(fermion phase } (-1)^\phi) \;\times\; \text{(SU(3) recoupling in } (\lambda\mu)) \;\times\;
\text{(SU(2) recoupling in } S).
$$

The SU(3)×SU(4) recoupling has the **same three factors**, with the SU(2) spin factor replaced
by an SU(4) factor:

| SU(3)×SU(2) object | SU(3)×SU(4) replacement | Provider |
|---|---|---|
| single‑operator spin `1/2` | SU(4) fund. `(1,0,0)` (creation) / `(0,0,1)` (annih.) | — |
| coupled spin `S` | coupled SU(4) irrep `W=(w1,w2,w3)` | — |
| `Π_S = √(2S+1)` | `Π_W = √(dim W)` | u4lib dim |
| SU(2) 6‑j `{· · ·; · · ·}` | SU(4) **U‑coefficient** `U[…]` | `U4_U6Z6Coeffs.h` |
| SU(2) 9‑j (3×3 array) | SU(4) **9‑(w) coefficient** | `U4_U9Coeffs.h` |
| SU(2) exchange phase `(−1)^{S1+S2−S12}` | SU(4) **Z‑coefficient / exchange factor** for `(1,0,0)²` (or `(0,0,1)²`) | `U4_U6Z6Coeffs.h` (Z) |

The **SU(3) part is byte‑for‑byte the same** as the notes (u3lib `U6`/`Z6`/`U9`), and the
**fermion phase `(−1)^φ` is the same** (it is pure anticommutation combinatorics, independent
of the internal groups).

> **Consequence.** The recoupling *decision table* of §4.1 (pages 15–16) — the shell‑ordering
> cases `{n1n2}×{n3n4}`, the sequence of relations to apply (A.49–A.57), the `(−1)^φ` column,
> and the resulting tree type **TTY1–4** — is **identical** for SU(3)×SU(4). Only each
> relation's coefficient acquires an SU(4) partner alongside its SU(3) one. So we do **not**
> re‑derive the table; we only generalize the seven relations it invokes.

Target tree shapes (`U,V` = the two creations `a†_{n1}, a†_{n2}`; `X,Z` = the two annihilations
`ã_{n3}, ã_{n4}`), unchanged:

```
TTY1 = [[U×V]×[X×Z]]          (two same-shell pairs)          ↔ E.g.4
TTY2 = [[[U×V]×X]×Z]          (left-nested / distinct)        ↔ E.g.1
TTY3 = [[U×[V×X]]×Z]          (middle two share a shell)      ↔ E.g.2
TTY4 = [U×[V×[X×Z]]]          (last three share a shell)      ↔ E.g.3
```

---

## 3. Generalized recoupling relations (A.49′–A.57′)

Below, every tensor leg carries **both** a U(3) irrep and a U(4) irrep; I suppress the
`κ L J M` (SO(3)/physical) labels since they ride along unchanged. A coupled pair is written
`{A×B}^{ρ(λμ);W}`. The primed labels A.49′…A.57′ denote the SU(3)×SU(4) versions of A.49…A.57.

### A.49′ — swap of a multiplicity‑free coupling
SU(2) form: `[X×Y]^{(λμ)}_S = (−1)^{(λ1+μ1)+(λ2+μ2)−(λ+μ)+S1+S2−S} [Y×X]^{(λμ)}_S`.
SU(3)×SU(4):
$$
\big[X^{(\lambda_1\mu_1)W_1}\times Y^{(\lambda_2\mu_2)W_2}\big]^{(\lambda\mu)\,W}
=(-1)^{(\lambda_1+\mu_1)+(\lambda_2+\mu_2)-(\lambda+\mu)}\;\sigma^{(4)}_{W}\;
\big[Y^{(\lambda_2\mu_2)W_2}\times X^{(\lambda_1\mu_1)W_1}\big]^{(\lambda\mu)\,W}
$$
where `σ⁴_W` is the SU(4) exchange sign for `W1×W2→W` (for `(1,0,0)×(1,0,0)`: `+1` for the
symmetric `W=(2,0,0)`, `−1` for the antisymmetric `W=(1,1,0)` — the analog of
`(−1)^{1−S}`). **[VERIFY: exact SU(4) exchange sign / whether it's a full Z‑coefficient when
`W1≠W2`.]**

### A.50′ — swap when the coupling is not multiplicity‑free (Z‑coefficient)
$$
\big[X\times Y\big]^{\rho(\lambda\mu)W}
=\sum_{\rho'\,W'} \Phi^{(3)}_{\rho\rho'}\big[(\lambda_1\mu_1),(\lambda_2\mu_2),(\lambda\mu)\big]\,
\Phi^{(4)}_{W W'}\big[W_1,W_2,W\big]\,
\big[Y\times X\big]^{\rho'(\lambda\mu)W'}
$$
`Φ³` is the u3lib SU(3) Z‑coefficient (as in A.50); `Φ⁴` its u4lib SU(4) partner. For the
paired scheme `W1=W2=(1,0,0)` (or `(0,0,1)`), so `W'` is fixed and `Φ⁴` reduces to `σ⁴_W`.

> For A.49′/A.50′ the overall fermion factor `(−)^φ` from reordering operators multiplies the
> right‑hand side, exactly as stated below A.50 in the notes.

### A.51′ — reorder `U↔V` inside the first (creation) pair
SU(2): prefactor `(−1)^{(λ1+μ1)+(λ2+μ2)−(λ_f+μ_f)+1−S12}`. SU(3)×SU(4):
$$
(-1)^{(\lambda_1+\mu_1)+(\lambda_2+\mu_2)-(\lambda_{12}+\mu_{12})}\;\sigma^{(4)}_{W_{12}}\;
\Big[\{V\times U\}^{(\lambda_{12}\mu_{12})W_{12}}\times\{X\times Z\}^{(\lambda_{34}\mu_{34})W_{34}}\Big]^{\rho_0(\lambda_0\mu_0)W_0}
$$
(the `+1−S12` in the SU(2) exponent is the fundamental‑pair exchange sign, i.e. `σ⁴_{W12}`
here). **[VERIFY: SU(4) fundamental‑pair exchange sign.]**

### A.52′ — reorder `X↔Z` inside the second (annihilation) pair
Same as A.51′ with `(1,2)→(3,4)`, `W12→W34`, using the **conjugate** fundamental `(0,0,1)`:
$$
(-1)^{(\lambda_3+\mu_3)+(\lambda_4+\mu_4)-(\lambda_{34}+\mu_{34})}\;\bar\sigma^{(4)}_{W_{34}}\;
\Big[\{U\times V\}^{W_{12}}\times\{Z\times X\}^{W_{34}}\Big]^{\rho_0(\lambda_0\mu_0)W_0}.
$$
**[VERIFY: conjugate‑fundamental exchange sign `σ̄⁴`.]**

### A.53′ — regroup `[U×{V×{X×Z}}]` (U‑coefficient, associativity)
SU(2) form uses a 6‑j `{S12 ½ S123; ½ Ŝ0 S234? …}` and a U(3) `U[…]`. SU(3)×SU(4):
$$
\text{(SU(2)→SU(4): )}\quad
\Pi_{W_{34}}\!\!\sum_{W_{234}} \Pi_{W_{234}}\,
\mathcal{U}^{(4)}\!\big[W_{12}\,W_3\,W_0\,W_4;\,W_{123}\,W_{234}\big]
\;\times\;
\sum_{\rho'\dots} U\big[(\lambda_1\mu_1)(\lambda_2\mu_2)(\lambda_0\mu_0)(\lambda_4\mu_4);\dots\big]\;
[\,U\times\{V\times\{X\times Z\}\}\,]
$$
with `U` the u3lib SU(3) U‑coefficient and `𝒰⁴` the u4lib SU(4) U‑coefficient replacing the
SU(2) 6‑j. **[VERIFY: exact intermediate‑irrep sums and `𝒰⁴` argument order.]**

### A.54′ — regroup `[{{U×V}×X}×Z]` (U‑coefficient)  →  **TTY2**
$$
\Pi_{W_{34}}\!\!\sum_{W_{123}}\Pi_{W_{123}}\,
\mathcal{U}^{(4)}\!\big[W_{12}\,W_3\,W_0\,W_4;\,W_{123}\,W_{34}\big]\;\times\;
U\big[(\lambda_{12}\mu_{12})(\lambda_3\mu_3)(\lambda_0\mu_0)(\lambda_4\mu_4);\,(\lambda_{123}\mu_{123})\dots\big]\;
[\{\{U\times V\}\times X\}\times Z].
$$

### A.55′ — regroup `[{U×{X×Z}}×V]` (Z‑coefficient)  →  **TTY3**
Built (per the notes) from the U(4) Z‑coefficient and 6‑j analog:
$$
\bar\sigma^{(4)}\,\Pi_{W_{12}}\!\!\sum_{W_{134}}\Pi_{W_{134}}\,
\mathcal{Z}^{(4)}\!\big[W_2\,W_1\,W_0\,W_{34};\,W_{12}\,W_{134}\big]\;\times\;
Z\big[(\lambda_2\mu_2)(\lambda_1\mu_1)(\lambda_0\mu_0)(\lambda_{34}\mu_{34});\dots\big]\;
[\{U\times\{X\times Z\}\}\times V].
$$
with `Z` the u3lib SU(3) Z‑coefficient and `𝒵⁴` its u4lib partner.

### A.56′ — regroup `[{U×X}×{V×Z}]` (9‑(λμ) ⊗ 9‑(w))  →  used to reach TTY2 in mixed cases
The SU(2) 9‑j and the SU(3) 9‑(λμ) both appear; add the SU(4) 9‑(w):
$$
\Pi_{W_{12}}\Pi_{W_{34}}\!\!\sum_{\substack{W_{13}\,W_{24}}}\!\!\Pi_{W_{13}}\Pi_{W_{24}}
\begin{Bmatrix} W_1 & W_2 & W_{12}\\ W_3 & W_4 & W_{34}\\ W_{13} & W_{24} & W_0\end{Bmatrix}^{(4)}
\;\times\;
\begin{Bmatrix}(\lambda_1\mu_1)&(\lambda_2\mu_2)&(\lambda_{12}\mu_{12})\\ (\lambda_3\mu_3)&(\lambda_4\mu_4)&(\lambda_{34}\mu_{34})\\ (\lambda_{13}\mu_{13})&(\lambda_{24}\mu_{24})&(\lambda_0\mu_0)\end{Bmatrix}
\;[\{U\times X\}\times\{V\times Z\}]
$$
where the first factor is the u4lib 9‑(w) and the second the u3lib 9‑(λμ) (`U3_U9Coeffs`).

### A.57′ — regroup `[{U×Z}×{V×X}]` (9‑(λμ) ⊗ 9‑(w), with a phase)  →  proton–neutron split
Same as A.56′ with `X↔Z` in the sub‑couplings and the extra SU(2) exchange phase
`(−1)^{(λ3+μ3)+(λ4+μ4)−(λ34+μ34)+1−S34}` promoted to `(−1)^{(λ3+μ3)+(λ4+μ4)−(λ34+μ34)}·σ̄⁴_{W34}`.

---

## 4. The recoupling decision table (unchanged structure)

Identical to pages 15–16 of the notes — reproduced here for completeness. `a,b,c,d` are the
four operators (2 creations, 2 annihilations) in the paired tensor; the `Case` column is their
shell ordering; apply the listed relations (now the primed versions), pick up `(−)^φ`, land in
the shown tree `Type`. (Key such as `aa0aa0` = the `map<CTuple<char,6>, …>` structure lookup.)

| `{n1n2}×{n3n4}` | Case | Recoupling | `(−)^φ` | Result | Type |
|---|---|---|---|---|---|
| `{aa}×{aa}` | | none | | | TTY1 |
| `{ba}×{aa}` | | A.53′ | + | `[b{a{aa}}]` | TTY4 |
| `{aa}×{ba}` | | A.52′→A.54′→A.50′ | + | `[b{{aa}a}]` | |
| `{aa}×{ab}` | | A.54′ | + | `[{{aa}a}b]` | TTY2 |
| `{ab}×{aa}` | | A.55′ | + | `[{a{aa}}b]` | TTY3 |
| `{aa}×{bb}` | a<b | none | | | TTY1 |
| | a>b | A.50′ | + | `[{bb}{aa}]` | TTY1 |
| `{aa}×{bc}` | a<b<c | A.54′ | + | `[{{aa}b}c]` | TTY2 |
| | b<a<c | A.50′→A.55′ | + | `[{b{aa}}c]` | TTY3 |
| | b<c<a | A.50′ | + | `[{bc}{aa}]` | TTY1 |
| `{bc}×{aa}` | b<c<a | none | | | TTY1 |
| | b<a<c | A.55′ | + | `[{b{aa}}c]` | TTY3 |
| | a<b<c | A.50′→A.54′ | + | `[{{aa}b}c]` | TTY2 |
| `{ab}×{ac}` | b≤c | A.56′→A.54′ | − | `[{{aa}b}c]` | TTY2 |
| | b>c | A.56′→A.52′→A.54′ | + | `[{{aa}c}b]` | TTY2 |
| `{ba}×{ca}` | b≤c | A.56′ | − | `[{bc}{aa}]` | TTY1 |
| | b>c | A.56′→A.51′ | + | `[{cb}{aa}]` | TTY1 |
| `{ba}×{ac}` | | A.57′→A.55′ | + | `[{b{aa}}c]` | TTY3 |
| `{ab}×{ca}` | | A.57′→A.51′→A.55′ | − | `[{c{aa}}b]` | TTY3 |
| `{ab}×{cd}` | a<b<c<d | A.54′ | + | `[{{ab}c}d]` | TTY2 |
| | a<c<b<d | A.56′→A.54′ | − | `[{{ac}b}d]` | TTY2 |
| | a<c<d<b | A.56′→A.52′→A.54′ | + | `[{{ac}d}b]` | TTY2 |
| | c<a<b<d | A.56′→A.51′→A.54′ | + | `[{{ca}b}d]` | TTY2 |
| | c<a<d<b | A.56′→A.51′&A.52′→A.54′ | − | `[{{ca}d}b]` | TTY2 |
| | c<d<a<b | A.50′→A.54′ | + | `[{{cd}a}b]` | TTY2 |

(The mixed proton–neutron cases — a creation and an annihilation sharing a shell — are the
`{ab}×{ac}`, `{ba}×{ca}`, `{ba}×{ac}`, `{ab}×{ca}` rows: they invoke A.56′/A.57′ first, which is
where the SU(4) 9‑(w) does the real work.)

---

## 5. Items to verify (the SU(4)-specific pieces)

1. **Exchange signs** `σ⁴_W`, `σ̄⁴_W` (A.49′, A.51′, A.52′, A.57′) for `(1,0,0)×(1,0,0)` and
   `(0,0,1)×(0,0,1)`: is it simply `+1`(symmetric)/`−1`(antisymmetric), or a full Z‑coefficient?
2. **U4 coefficient conventions**: the exact argument order and normalization of the u4lib
   U‑coefficient (`U4_U6Z6Coeffs`), Z‑coefficient, and 9‑(w) (`U4_U9Coeffs`) so A.53′–A.57′ map
   1‑to‑1 onto function calls. (I'll read the `get(...)` signatures when we implement.)
3. **Π factors**: `Π_W = √(dim W)` — confirm `dim W` (and whether any `Π` should instead be a
   different Casimir/normalization, as sometimes differs between SU(3) and SU(4) conventions).
4. **Fermion phase `(−)^φ`**: confirm it is exactly the SU(2)‑notes value (pure operator
   anticommutation), i.e. independent of `W`, so the table's `(−)^φ` column is reused verbatim.
5. **Multiplicity**: in SU(4) the couplings `(1,0,0)×(1,0,0)` and `(0,0,1)×(0,0,1)` are
   multiplicity‑free (like the SU(3) `[n]×[m]`), so the `ρ'/W'` sums in A.50′ collapse — confirm.

Once (1)–(5) are confirmed, §3 becomes a direct set of u3lib×u4lib coefficient products, and
§4 is a literal port of the notes' table. The validation harness (Phase B plan) then checks
`Σ_paired σ · ⟨T_shell | T_paired⟩ == strength_T(current expand_new)` per tree case.
