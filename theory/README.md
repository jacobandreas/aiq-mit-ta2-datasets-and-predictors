# Theory indexes

Vendored, machine-readable descriptions of the theoretical statements this
repository's evaluation cards refer to. Each entry names a statement, where it
is formalized (when it is), and its premises by binder name.

These are **data, not code**, kept here so the cards resolve inside this
repository alone — no external checkout is needed to load a card or to audit
it. Refresh them by copying the upstream file over the local one.

The annotation vocabulary itself is *not* vendored: the source files
`import magnet.theory as theory` and use the decorators from MAGNET directly,
so there is no second copy to drift.

Nothing here is imported or executed. MAGNET reads the cards' `theory:` blocks
and the annotations in the source with `ast`, so auditing a card never requires
installing this repository's runtime dependencies.

A statement with no `declaration:` and no `formalization:` is **stated, not
formalized** — that is deliberate and is recorded as such. Naming a formal
declaration that does not exist is the failure this machinery exists to prevent.

An entry of `kind: conjecture` is an open question, carrying `sorry` in its
formalization. Cards link to those with `motivates`, which creates no
premise-coverage obligation.

## Building the Lean

`lean/` is a real Lean 4 project, not a document, not a document. It is a
self-contained Lake project so you can check it yourself:

```bash
cd theory/lean
lake exe cache get     # downloads prebuilt Mathlib oleans, a few minutes
lake build
```

A clean build prints nothing. There is no `sorry` in it.

**Do not run `lake update`.** The committed `lake-manifest.json` pins the exact
Mathlib revision these proofs were checked against; resolving fresh picks up a
newer Mathlib, and a proof that holds against one revision is not guaranteed to
hold against another. `lean/formalization.yaml` records the toolchain, the card this
is about, and — in `fidelity.known_limitations` — what it deliberately does not
formalize.

The file's own header is the thing worth reading first: it says what is proved,
what is assumed, and what to do next.
