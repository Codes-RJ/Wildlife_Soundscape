# Known-Good Patch Notes

The two classification files under `known_good/classification/` are the latest
corrected copies available during this handoff.

Important invariant already corrected:

If:
`second_label is None`

then:
`second_confidence is None`

The ensemble file also handles the ordinary no-second-class fusion case by
keeping the second confidence as None and calculating margin against zero
implicitly/top-score-only.

Gemini must not regress this contract while adding taxonomy/context support.
