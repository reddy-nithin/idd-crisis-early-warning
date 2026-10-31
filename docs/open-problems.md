# Open problems

Ideas we haven't tried yet. If you're new to the project, pick one that interests you, or add your own. Each one names the files you'd touch.

## Modelling

1. **Event order.** The pitch asked whether particular sequences of incidents (for example aggression, then elopement, then police) come before a crisis. Our features count events in windows and lose the order. Options: n-gram counts of incident categories, a small recurrent model or transformer on each person's event sequence, or sequential pattern mining. Files: `features.py`, `fimpp_export.py` already builds per-person sequences.
2. **Personal baselines that adapt.** The "usual rate" is the person's average over all of their history. A rolling or exponentially weighted baseline, or a Bayesian per-person rate that shrinks toward the group when history is short, might make the self-relative features much stronger. Files: `features.py`.
3. **Change-point detection.** Treat each person's incident rate as a signal and detect when it shifts (for example Bayesian online change-point detection or CUSUM). Use "time since the last change point" as a feature or as its own alarm.
4. **Time-to-event model.** Predict how many days until the next crisis instead of yes/no in 28 days (Cox model, recurrent-event Hawkes process). This also handles people with many crises more naturally.
5. **Group fairness fix.** The model flags women more and misses more crises in men. Try gender-specific thresholds or interaction terms, and check whether the gap is real or comes from women in this cohort having more crises per person. Files: `evaluate.py`.
6. **Calibration.** Probabilities aren't calibrated yet. Isotonic or Platt scaling on a validation slice of the training period.

## Data and labels

7. **Which crises are invisible, and why?** Profile the 41% of crises with no incident in the prior 30 days: are they new people, people coming out of inpatient stays, particular settings? That tells us where to look for extra data.
8. **Labels B and C.** Run the full comparison with police and ER/hospitalisation added to the crisis definition (`--label B`, `--label C`).
9. **Victim-only rows.** Some rows have the person only as a victim or witness. Should those count as their own behaviour? Needs clinical input. Files: `events.py`.

## Product

10. **Synthetic data generator.** Same columns and rough statistics as the real export, so anyone can run the pipeline and the demo. Must not copy real timelines.
11. **Dashboard.** Streamlit or Next.js page: timeline, baseline band, risk flag, top reasons. The design should come from the clinical team's workflow, not ours.
12. **Explanations.** Per-person "why this flag" from logistic contributions or SHAP values, in plain words a direct support professional would understand.

## Rules for new experiments

- Never look at test-set results while choosing features or settings. Use the training period only, then run the test once.
- Keep the patient data on the approved machine. Share only aggregates, and nothing about groups smaller than 11 people. See [data-governance.md](data-governance.md).
