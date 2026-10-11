# Submission checklist

The instructor requires a GitHub repository URL containing the source code and a PDF report. Initializing the local Git repository does not create a hosted repository or submit a URL. Creating a commit is a separate step.

## Before submission

- [x] The cover names **Kiseong Kim** as team leader and **Jisung Park** as team member, matching the team's intended roles.
- [ ] Both students read the PDF, verify its claims against saved results, and can explain the required CNN/RNN experiments.
- [x] The PDF opens correctly and the report tables match the saved result files.
- [x] Run `python -m pytest -q` and `python scripts/verify_submission.py` in the installed project environment; check `artifacts/verification.json`.
- [x] Dataset provenance, filtering, licenses, model versions and limitations remain included.
- [x] Koala and OASST are excluded from all splits; the final 38 experiments are retrained on the revised subset. Source exclusions are enforced by `verify_submission.py`.
- [x] RQ1 and all three RQ2 input conditions are present for both CNN and LSTM.
- [x] Extra-credit RQ3 and RQ4 conclusions match the actual experiments and do not claim universal attribution.
- [x] The source and reproduction commands are included; credentials, virtual environments and raw private files are excluded.
- [x] Final files are published at https://github.com/kjk6172/cmpsc448_Midterm1 . Anonymous source/PDF access and a fresh-clone test/artifact verification passed.
- [ ] Submit that repository URL through the course's required channel before **October 11, 2026, 11:59 p.m. America/New_York**.

## Optional presentation

The assignment offers a default 17/20 presentation score when the team chooses not to present. If presenting, use the actual results and rehearse the questions in `DEFENSE_GUIDE_KO.md`. Extra-credit grading and the final grade remain the instructor's decision.

## Local artifacts

- `report/project_report.pdf`: final English report.
- `report/project_report.md`: matching text for easy review.
- `report/presentation.pdf`: optional eight-slide presentation.
- `docs/PRESENTATION_NOTES_KO.md`: Korean speaker notes for the optional presentation.
- `report/figures/`: charts generated from actual results.
- `artifacts/`: measured outcomes, predictions, checkpoints and execution metadata.
- `data/`: curated licensed public examples and provenance, as described in the data documentation.
- `docs/DEFENSE_GUIDE_KO.md`: Korean explanation and presentation guide.

## Repository and later updates

The published repository is **https://github.com/kjk6172/cmpsc448_Midterm1**. The final scientific artifacts were checked against the local Git tree, and a fresh public clone passed all 12 unit tests and full artifact verification. Both students' reading/understanding and submission through the course channel remain separate manual responsibilities.

For later authorized changes, commit the updated files and push to the configured `origin`. Regenerate affected results and the PDF before uploading changes to data or model code. The local `.tools` helper is ignored and is not included in the submission package.

```powershell
git remote -v
git add .
git commit -m "Update CMPSC 448 submission"
git push origin main
```

Submit the repository URL itself through the course's required channel; the ZIP is only a portable backup.
