## Pull Request Checklist
- [ ] The title of the PR is clear and concise.
- [ ] I have provided a detailed description of the changes and the reason for the changes.
- [ ] I have linked the related issue(s) in the description (if applicable).
- [ ] I have followed the coding guidelines of this project.
- [ ] I have ensured there are no breaking changes.
- [ ] I have updated related documentation (if applicable).
- [ ] I have added reviewers, including at least one member of the MLOps team (@hrosegalb, @PaulNdrei, @ankush13r, @igorktech)

## New Task Checklist
- [ ] I have ensured that the dataset used in the task has been human annotated or translated, or that, if synthetic, a strong and reliable human revision has taken place.
- [ ]  I have explored existing Harness tasks to check the types of prompts, metrics and setups used in similar tasks, and either adopted the same for my task or have a strong reason(s) not to do it (please provide justification in description).
- [ ]  I have reviewed the dataset structure and contents to ensure that the prompt used does not impact the quality of the input given to the models.
- [ ]  If errors were detected, I have added suitable pre- or post-processing to the task.
- [ ]  I have tested that my task runs from beginning to end.
- [ ]  I have reviewed the inputs given to the models to ensure that instructions are natural, grammatical, and as I expected overall (e.g., punctuation and capitalization are used correctly, the few-shot context is not truncated, etc.).
- [ ]  I have reviewed the outputs of the evaluations of the models, making sure that metrics and other aspects of the setup work as I expected.
- [ ]  If my task is in Spanish, Catalan, Galician, Basque, Portuguese or English, I have talked to Javier Aula-Blasco regarding the possibility of adding it to one of the Benches.

## Description
Please provide a short summary of the changes in this PR and why they are needed.

## Issue Link(s)
Closes # (replace with the issue number or description)

## Testing
Please describe how you tested the changes and any relevant details for reviewers.

