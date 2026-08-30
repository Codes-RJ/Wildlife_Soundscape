# Spec — Database + Dashboard Integration

## Database

Add `soundscape_indices` using backward-compatible initialization/migration.

Keep index parameter metadata so future analysis can distinguish results
computed with different bands/FFT settings.

## Dashboard

Add:
- index time-series view;
- node selector or aggregate view;
- parameter/config note;
- empty-state message.

Do not imply:
- high ACI = proven high species richness;
- NDSI = direct animal count.

For BirdNET, display acoustic prediction separately from geo-context evidence.

## Performance

Do not repeatedly recompute historical indices every Streamlit rerun.
Read persisted results.

Live calculation should happen in the analysis/service layer, not inside plotting
functions.
