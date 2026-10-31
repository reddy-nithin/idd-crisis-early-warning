# Data governance

## What is and isn't in this repo

| In the repo | Not in the repo |
|---|---|
| Code, tests (on made-up timelines), docs | The incident data (`data/`) |
| Aggregate result tables (`results/`) | Per-person predictions and model outputs (`outputs/`) |
| Figures built from aggregates | Fine-tuned model weights |
| | API keys (`.env`) |

## How the data is handled

- **Source.** Pseudonymised incident reports from the clinical lead's own program, shared for this feasibility study. People are identified only by study codes. There are no names, DMH IDs, addresses or organisation names in the structured fields.
- **Identifier check.** Before any modelling we checked every column, including free text. Two list columns contained typed staff notes, one of which included a first name. Those notes are removed from the working copy by `scripts/strip_free_text.py`, which fails loudly if any are left. The original file is kept separately and untouched.
- **One machine.** The data lives on one encrypted laptop. It is not on cloud storage, shared servers or university GPU machines, and it is never sent to a cloud API. TabPFN runs locally with the `tabpfn` package, never `tabpfn-client`.
- **Reduced detail.** Program names are reduced to a setting type (group home, supported living, inpatient, other). Ages and gender are kept, as agreed with the clinical lead.
- **Sharing rule.** Only aggregates leave the machine: no rows, no individual timelines, no exact event dates, and nothing reported for groups under 11 people. `scripts/make_figures.py` enforces the 11-person rule for the fairness table.
- **End of project.** The raw data is deleted when the project ends unless a formal agreement says otherwise.

## Open items

- A formal data use agreement and an IRB determination are needed before this goes beyond the hackathon feasibility study (Phase 2 of the [roadmap](04-roadmap.md)).
- The exact dates in the data make it a limited data set, not fully de-identified. Any future sharing should shift or coarsen dates.
