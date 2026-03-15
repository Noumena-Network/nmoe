# Falsification Ledger Index (Posts 0000-0011)

This directory is the attack-surface companion to `repro/claim_ledger/`.

- claim ledgers enumerate the live claims, evidence pointers, and current verification status
- falsification ledgers enumerate the hostile tests, surviving claim boundary, and remaining attack surface

The intent is simple: do not weaken the paper to fit current receipts. Strengthen the attack until the surviving claim is exactly what the math and evidence support.

## Status Legend

- `attack_surface_defined`: the live claim and kill surface are written down, but the main hostile tests are not yet closed.
- `partial_attack`: the claim already has at least one real attack, but stronger or broader falsification is still missing.
- `strong_attack`: the claim has already survived a serious adverse test matrix inside the post's stated scope.
- `theory_program`: the ledger is mainly about theorem / conjecture boundaries and the measurements that would count as empirical failure.

## Inventory

| Post | Status | Companion claim ledger |
|------|--------|------------------------|
| 0000 | `attack_surface_defined` | `repro/claim_ledger/0000.md` |
| 0001 | `attack_surface_defined` | `repro/claim_ledger/0001.md` |
| 0002 | `partial_attack` | `repro/claim_ledger/0002.md` |
| 0003 | `partial_attack` | `repro/claim_ledger/0003.md` |
| 0004 | `partial_attack` | `repro/claim_ledger/0004.md` |
| 0005 | `strong_attack` | `repro/claim_ledger/0005.md` |
| 0006 | `partial_attack` | `repro/claim_ledger/0006.md` |
| 0007 | `theory_program` | `repro/claim_ledger/0007.md` |
| 0008 | `partial_attack` | `repro/claim_ledger/0008.md` |
| 0009 | `partial_attack` | `repro/claim_ledger/0009.md` |
| 0010 | `strong_attack` | `repro/claim_ledger/0010.md` |
| 0011 | `strong_attack` | `repro/claim_ledger/0011.md` |
