/-
# Circuit overlap as a predictor of per-instance correctness

The card scores each hidden unit by attribution, takes the top-K units as a
"circuit", and predicts whether the model will get an item right from the
Jaccard overlap between that item's circuit and a reference circuit. The claim
is that the resulting AUC clears a threshold.

The soundness question underneath that claim is whether the overlap says
anything about the FUNCTION the network computes, or only about the arbitrary
order its hidden units happen to be stored in. Hidden units carry a permutation
symmetry: relabelling them yields a network computing the same function, and a
per-unit attribution moves with the relabelling. If the overlap moved too, the
number would be an artifact of the checkpoint's memory layout.

Within one model it does not move, and this file proves it. `topK` is defined
concretely -- the units that fewer than `K` other units outscore -- and:

* `topK_image` says the selection is equivariant: relabelling the units and the
  scores together relabels the selected circuit and nothing else.
* `jaccard_image` says Jaccard is invariant when both sets move under one
  permutation, because the intersection and the union are images of that same
  permutation and `Finset.card_image_of_injective` carries both cardinalities
  through.
* `jaccard_topK_relabel` composes them into the statement the deployed use rests
  on: relabelling the hidden units leaves the reference/item overlap exactly
  unchanged, for every reference vector, item vector and K.

The converse boundary is proved too. `jaccard_not_invariant_of_one_sided_image`
exhibits two sets and a permutation applied to only ONE of them that changes the
overlap from 1 to 0. That is the situation of two independently trained models,
whose unit labellings are unrelated, and it is why the cross-model comparison is
NOT licensed by anything here. It remains an open question in
`AIQ.Conjectures.Circuits`.

`topK` is only "the top K" when the scores have no ties: with all scores equal
every unit is outscored by none of them and `topK` is the whole universe.
`topK_card_le` therefore carries an injectivity hypothesis on the scores, and
that hypothesis is the honest form of a tie-breaking rule the card leaves
implicit. Ties in a float attribution vector are unlikely but the card does not
say what it does with them, and the answer changes the circuit's size.

## What this does not formalize

* **No attribution and no network.** `w : α → ℝ` is an arbitrary score vector;
  nothing says it is an attribution, and nothing says a relabelling of units is
  the only symmetry the attribution respects. The equivariance proved here is a
  fact about top-K selection, and it transfers to the card exactly insofar as
  the attribution is computed per unit from unit-local quantities.
* **No AUC.** The card's actual claim is about ranking quality across items, and
  no notion of ranking, label, or threshold appears below. This is now the
  largest gap: `jaccard_topK_relabel` says the predictor is well defined, not
  that it predicts anything.
* **No tie-breaking.** See above: `topK_card_le` assumes distinct scores rather
  than specifying what happens without them.
* **Cross-model comparison is still unjustified**, and
  `jaccard_not_invariant_of_one_sided_image` shows the naive hope is false, not
  merely unproved.

## Where a real formalization would go

The equivariance gap listed here previously is closed. The next one is the AUC
claim itself: define the item labels and the induced ranking, and state what
would have to be true of the score distribution for the overlap to separate
correct from incorrect items above chance. Even the definition would be worth
having, because "AUC ≥ θ" as computed is a statement about one finite sample and
the card reports it as a property of the model.

Making the cross-model claim true, rather than merely stating it, needs a
canonical alignment between two models' units -- a matching, not a permutation of
one index set -- and then the invariance argument here does not apply at all.

## Prior art worth using

Everything used here is Mathlib: `Equiv.Perm` for the relabelling,
`Finset.filter_image` for equivariance of the selection, and
`Finset.card_image_of_injective` to carry cardinalities through it.

For the AUC step, a formal treatment of ranking statistics would be needed;
**StatsMLlib** (github.com/Lean-MoDS/StatsMLlib) carries the concentration
inequalities that would bound the sampling error of a measured AUC and pins
`v4.33.0` against this development's `v4.33.0-rc2`.
-/
import Mathlib.Data.Finset.Card
import Mathlib.Data.Finset.Image
import Mathlib.Data.Finset.Max
import Mathlib.Data.Real.Basic
import Mathlib.Logic.Equiv.Basic
import Mathlib.Tactic.Linarith

namespace AIQ.Teams.CircuitOverlap

variable {α : Type*} [DecidableEq α]

/-! ## Overlap -/

/-- Jaccard overlap of two circuits. Zero on two empty circuits, which the card
never produces: `topK` with `K > 0` is nonempty on a nonempty set of units, by
`topK_nonempty`. -/
noncomputable def jaccard (A B : Finset α) : ℝ :=
  ((A ∩ B).card : ℝ) / ((A ∪ B).card : ℝ)

/-- Overlap does not depend on the order of its arguments. -/
theorem jaccard_comm (A B : Finset α) : jaccard A B = jaccard B A := by
  unfold jaccard
  rw [Finset.inter_comm, Finset.union_comm]

/-- A circuit overlaps itself completely. -/
theorem jaccard_self {A : Finset α} (hA : A.Nonempty) : jaccard A A = 1 := by
  unfold jaccard
  rw [Finset.inter_self, Finset.union_self]
  exact div_self (Nat.cast_ne_zero.mpr (Finset.card_ne_zero_of_mem hA.choose_spec))

/-- Overlap is bounded by one: the intersection is contained in the union. -/
theorem jaccard_le_one (A B : Finset α) : jaccard A B ≤ 1 := by
  unfold jaccard
  rcases Finset.eq_empty_or_nonempty (A ∪ B) with h | h
  · simp [Finset.union_eq_empty.mp h]
  · rw [div_le_one (by exact_mod_cast Finset.card_pos.mpr h)]
    exact_mod_cast Finset.card_le_card (Finset.inter_subset_union)

/-! ## Relabelling hidden units

A relabelling of the hidden units is an `Equiv.Perm` on the unit index type. It
acts on a set of units by image and on a score vector by precomposition with the
inverse, so that the score of a unit follows the unit. -/

/-- **Overlap is invariant when both circuits move under one relabelling.**
Both the intersection and the union of the images are images of the
intersection and the union, and a permutation is injective, so both
cardinalities are unchanged and so is their ratio. -/
theorem jaccard_image (σ : Equiv.Perm α) (A B : Finset α) :
    jaccard (A.image σ) (B.image σ) = jaccard A B := by
  unfold jaccard
  rw [← Finset.image_inter _ _ σ.injective, ← Finset.image_union,
    Finset.card_image_of_injective _ σ.injective,
    Finset.card_image_of_injective _ σ.injective]

/-- **Moving only one circuit is not harmless.** Two identical single-unit
circuits overlap completely; relabel one side alone and they overlap not at all.
This is the situation of two independently trained models, whose unit
labellings have no reason to agree, and it is why nothing here licenses
comparing circuits across models. -/
theorem jaccard_not_invariant_of_one_sided_image :
    ∃ (A B : Finset ℕ) (σ : Equiv.Perm ℕ), jaccard (A.image σ) B ≠ jaccard A B := by
  refine ⟨{0}, {0}, Equiv.swap 0 1, ?_⟩
  have h : ({0} : Finset ℕ).image (Equiv.swap 0 1) = {1} := by
    simp [Equiv.swap_apply_left]
  rw [h, jaccard_self ⟨0, Finset.mem_singleton_self 0⟩]
  unfold jaccard
  norm_num

/-! ## Top-K selection -/

/-- The units that fewer than `K` other units outscore. With distinct scores
this is exactly the top `K` units by score; see `topK_card_le`.

Stating the selection this way rather than by sorting is what makes
equivariance provable without choosing a tie-break: the definition mentions only
comparisons between scores, and a relabelling preserves those. -/
noncomputable def topK (U : Finset α) (w : α → ℝ) (K : ℕ) : Finset α :=
  U.filter fun a => (U.filter fun b => w a < w b).card < K

omit [DecidableEq α] in
theorem mem_topK {U : Finset α} {w : α → ℝ} {K : ℕ} {a : α} :
    a ∈ topK U w K ↔ a ∈ U ∧ (U.filter fun b => w a < w b).card < K := by
  simp [topK]

omit [DecidableEq α] in
/-- A circuit is a set of the model's own units. -/
theorem topK_subset (U : Finset α) (w : α → ℝ) (K : ℕ) : topK U w K ⊆ U :=
  Finset.filter_subset _ _

omit [DecidableEq α] in
/-- A higher score is outscored by strictly fewer units: everything above `b` is
above `a`, and `b` itself is above `a` and not above `b`. This is the rank
comparison that makes the selection deserve the name "top K". -/
theorem card_outranking_lt {U : Finset α} {w : α → ℝ} {a b : α}
    (hb : b ∈ U) (hab : w a < w b) :
    (U.filter fun c => w b < w c).card < (U.filter fun c => w a < w c).card := by
  apply Finset.card_lt_card
  rw [Finset.ssubset_iff_of_subset]
  · exact ⟨b, Finset.mem_filter.mpr ⟨hb, hab⟩, by simp⟩
  · intro c hc
    rw [Finset.mem_filter] at hc ⊢
    exact ⟨hc.1, hab.trans hc.2⟩

omit [DecidableEq α] in
/-- **The selection really selects at most `K` units, provided no two units tie.**
The rank map `a ↦ #{b ∈ U | w a < w b}` is injective on `U` when the scores are,
and `topK` is exactly its preimage of `{0, …, K-1}`.

The injectivity hypothesis is not removable: with all scores equal, no unit is
outscored by any other, and `topK U w 1` is all of `U`. The card does not say
what it does with tied attributions, and this is where that omission would be
felt. -/
theorem topK_card_le {U : Finset α} {w : α → ℝ} {K : ℕ} (hw : Set.InjOn w U) :
    (topK U w K).card ≤ K := by
  have hmaps : Set.MapsTo (fun a => (U.filter fun b => w a < w b).card)
      (topK U w K) (Finset.range K) := by
    intro a ha
    simp only [Finset.coe_range, Set.mem_Iio]
    exact (mem_topK.mp (Finset.mem_coe.mp ha)).2
  have hinj : Set.InjOn (fun a => (U.filter fun b => w a < w b).card) (topK U w K) := by
    intro a ha b hb hab
    have ha' := (mem_topK.mp (Finset.mem_coe.mp ha)).1
    have hb' := (mem_topK.mp (Finset.mem_coe.mp hb)).1
    dsimp only at hab
    rcases lt_trichotomy (w a) (w b) with h | h | h
    · exact absurd hab.symm (Nat.ne_of_lt (card_outranking_lt hb' h))
    · exact hw (Finset.mem_coe.mpr ha') (Finset.mem_coe.mpr hb') h
    · exact absurd hab (Nat.ne_of_lt (card_outranking_lt ha' h))
  calc (topK U w K).card
      ≤ (Finset.range K).card :=
        Finset.card_le_card_of_injOn _ hmaps hinj
    _ = K := Finset.card_range K

omit [DecidableEq α] in
/-- A nonempty model has a nonempty circuit whenever `K > 0`: the highest-scoring
unit is outscored by nobody. This is what the docstring on `jaccard` relies on
when it says the card never divides by zero. -/
theorem topK_nonempty {U : Finset α} {w : α → ℝ} {K : ℕ} (hU : U.Nonempty) (hK : 0 < K) :
    (topK U w K).Nonempty := by
  obtain ⟨a, haU, hmax⟩ := U.exists_max_image w hU
  refine ⟨a, mem_topK.mpr ⟨haU, ?_⟩⟩
  have : (U.filter fun b => w a < w b) = ∅ := by
    rw [Finset.filter_eq_empty_iff]
    exact fun b hb => not_lt.mpr (hmax b hb)
  rw [this]
  simpa using hK

/-- **Top-K selection is equivariant.** Relabelling the units by `σ` -- which
carries the score of a unit along with the unit, hence the `σ.symm` on the
scores -- relabels the selected circuit and does nothing else to it.

This is the step that turns the permutation symmetry of hidden units from a
worry into a bookkeeping fact: the circuit is not a set of indices that happens
to be stable, it is a set determined by score comparisons, and comparisons do
not know the indices. -/
theorem topK_image (σ : Equiv.Perm α) (U : Finset α) (w : α → ℝ) (K : ℕ) :
    topK (U.image σ) (fun a => w (σ.symm a)) K = (topK U w K).image σ := by
  have hcard : ∀ a : α,
      ((U.image σ).filter fun b => w (σ.symm (σ a)) < w (σ.symm b)).card
        = (U.filter fun b => w a < w b).card := by
    intro a
    rw [Finset.filter_image, Finset.card_image_of_injective _ σ.injective]
    simp
  unfold topK
  rw [Finset.filter_image]
  refine congrArg (Finset.image σ) (Finset.filter_congr fun a _ => ?_)
  rw [hcard a]

/-- **The claim the deployed use rests on.** Relabelling the hidden units of a
model changes neither the reference circuit nor the item circuit in any way that
the overlap can see: the measured overlap is identical, for every pair of score
vectors and every `K`.

So comparing circuits within one model measures the model and not the order its
units were stored in. Nothing here extends to two models: see
`jaccard_not_invariant_of_one_sided_image`. -/
theorem jaccard_topK_relabel (σ : Equiv.Perm α) (U : Finset α)
    (wRef wItem : α → ℝ) (K : ℕ) :
    jaccard (topK (U.image σ) (fun a => wRef (σ.symm a)) K)
        (topK (U.image σ) (fun a => wItem (σ.symm a)) K)
      = jaccard (topK U wRef K) (topK U wItem K) := by
  rw [topK_image, topK_image, jaccard_image]

end AIQ.Teams.CircuitOverlap
