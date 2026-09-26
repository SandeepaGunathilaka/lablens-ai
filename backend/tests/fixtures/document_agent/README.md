# Document Agent fixtures

Keep only anonymized CBC and Lipid Profile samples in this directory. Do not commit
patient names, dates of birth, report identifiers, contact details, or other health
information that can identify a person.

Week 9 evaluation should add representative samples for each category below, with an
expected-results JSON file beside each report:

- selectable-text PDF;
- clear scanned PDF or PNG/JPEG;
- blurry or low-contrast scan;
- at least two distinct laboratory layouts for CBC and Lipid Profile reports.

The current unit tests create synthetic documents in memory so the repository remains
free of patient data until approved anonymized fixtures are available.
