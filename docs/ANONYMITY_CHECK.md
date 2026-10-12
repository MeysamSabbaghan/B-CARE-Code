# Anonymity checklist (public)

- No author names / emails / institutional affiliations in packaged files
- No personal home-directory absolute paths in packaged files
- No API keys (`.env.example` placeholders only: `REPLACE_ME`)
- No `CITATION.cff` with real identities
- Branding uses **B-CARE** only
- No internal cluster/repository tree names in documentation or source comments
- Create a fresh git history before any public push (this package ships without `.git`)
