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
- [ ] Create or choose a GitHub repository, upload/push the committed files, and verify code and PDF are visible using the submitted URL.
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

## Commit and host the project when ready

The selected submission repository is https://github.com/kjk6172/cmpsc448_Midterm1 . The packaging step stages the final files; committing, uploading and course-channel URL submission are separate actions. From this workspace, the downloaded local MinGit executable can be used if `git` is not on `PATH`:

```powershell
.\.tools\git\cmd\git.exe status
.\.tools\git\cmd\git.exe commit -m "Complete CMPSC 448 midterm project"
```

If Git requests an identity, configure your chosen name and email before committing. The local `.tools` helper is ignored and is not included in the submission package; an extracted package can use a regular Git installation.

After creating your own GitHub repository, use its actual URL in these optional commands:

```powershell
git remote add origin https://github.com/kjk6172/cmpsc448_Midterm1.git
git branch -M main
git push -u origin main
```

The URL above is the team's selected repository. Verify that the hosted source and `report/project_report.pdf` can be accessed by the instructor, then submit the repository URL through the course channel.
