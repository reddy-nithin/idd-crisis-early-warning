# The problem

## What happens today

Adults with intellectual and developmental disabilities (IDD) who live in Missouri Department of Mental Health (DMH) funded group homes and individualized supported living (ISL) often go through a repeating cycle: a stable stretch, then a psychiatric or behavioural crisis, an ER visit, a same-day discharge, and another crisis. Over time this can cost them their home placement. Some end up homeless, arrested, or in hospital for months.

Staff write an incident report for every serious event: aggression, self-injury, property destruction, elopement (leaving without permission), psychiatric emergencies, police involvement, restraint, injuries. For some people there are years of these reports. They are read one at a time, when they are filed. An aggression incident, an elopement a few days later, another incident the next week: none of them looks unusual alone. The change in the overall pattern is what's hard to see.

The clinical lead who pitched this project put it this way: people look stable, then crash, and in hindsight the signs were in the incident reports.

## Why it matters

- Mental ill-health is common in adults with intellectual disabilities: 40.9% met clinical criteria in a population study of over 1,000 adults ([Cooper et al., 2007](https://doi.org/10.1192/bjp.bp.106.022483)). The 12-month prevalence of aggressive behaviour was 51.8% in over 3,000 adults receiving services ([Crocker et al., 2006](https://doi.org/10.1111/j.1365-2788.2006.00815.x)).
- Medicaid-insured adults with IDD have 3.2 times the inpatient admission rate and 2.6 times the ER visit rate of other adults. For psychiatric conditions the gap is 5.8 to 15 times ([Lauer et al., 2021](https://doi.org/10.1097/JAC.0000000000000373)).
- Poor mental health is a consistent predictor of high healthcare spending in people with IDD (Fujiura et al., 2018). A single ER visit with an ambulance costs around $1,168, and the hardest cases cost the state close to $1M a year (figures from the clinical team).
- Many people with IDD can't easily tell staff that their mood, anxiety, sleep or pain has changed. A change in behaviour is often the first thing anyone can see.

## Who it affects

The people with IDD first, then their families, direct support staff, clinicians, residential providers, day programs, police, ERs and the state Medicaid waiver programs that pay for care.

## The question

Can the data we already collect recognise when someone is moving toward a crisis, early enough for the team to check in, talk to caregivers, look for medical or environmental causes, or add support?

The pitch asked us to focus on change from each person's own baseline, because people with IDD vary a lot. Four incidents in a week is alarming for someone who usually has one a month, and an ordinary week for someone else.

## What this is not

It does not diagnose, recommend treatment, or replace clinical judgement. It is decision support: a flag that says "this person's pattern has changed, take a closer look". A clean negative result would also have been useful, because it would show what else needs to be collected.

## What others have done

| Study | Setting | Result | What we reuse |
|---|---|---|---|
| [Garriga et al., 2022, Nature Medicine](https://doi.org/10.1038/s41591-022-01811-5) | 17,122 general psychiatry patients, NHS electronic records | AUROC 0.80 for crisis onset in the next 28 days (XGBoost); 60 clinicians found it useful in 64% of cases in a 6-month pilot | Crisis onset definition, 28-day horizon, time split, "time since" features |
| [Wang et al., 2026, npj Mental Health Research](https://doi.org/10.1038/s44184-026-00194-6) | 17,703 inpatients, staff-reported aggression | AUROC 0.81; false alarms higher for police admissions, unhoused patients and men | Fairness audit by group |
| [Lauer et al., 2021](https://doi.org/10.1097/JAC.0000000000000373) | 72,595 Medicaid adults with IDD | Much higher acute care use | Motivation |
| [Blankers et al., 2020](https://doi.org/10.1186/s12911-020-01361-1) | Psychiatric crisis care | ML predicts hospitalisation after crisis care | Background |

All earlier crisis-prediction work we found uses large general psychiatry hospital records. None uses people with IDD, residential incident logs alone, or a person's own baseline. That is the gap this project tests.

## References

- Blankers, M., van der Post, L. F. M., & Dekker, J. J. M. (2020). Predicting hospitalization following psychiatric crisis care using machine learning. BMC Medical Informatics and Decision Making, 20, 332.
- Cooper, S.-A., Smiley, E., Morrison, J., Williamson, A., & Allan, L. (2007). Mental ill-health in adults with intellectual disabilities. The British Journal of Psychiatry, 190, 27 to 35.
- Crocker, A. G., et al. (2006). Prevalence and types of aggressive behaviour among adults with intellectual disabilities. Journal of Intellectual Disability Research, 50(9), 652 to 661.
- Fujiura, G. T., et al. (2018). Predictors of healthcare expenditures among people with IDD. [CITE: full reference]
- Garriga, R., et al. (2022). Machine learning model to predict mental health crises from electronic health records. Nature Medicine, 28, 1240 to 1248.
- Lauer, E., et al. (2021). Health service utilization patterns among Medicaid-insured adults with intellectual and developmental disabilities. Journal of Ambulatory Care Management, 44(2), 138 to 147.
- Wang, et al. (2026). Fairness audit of an aggression prediction model. npj Mental Health Research, 5, 16.
