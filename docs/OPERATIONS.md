# Operations guide

This is the operational entry point for the Wildlife Soundscape prototype:
startup, live hardware, dataset validation, and troubleshooting.

## Software demo

From the repository root on Windows:

```powershell
python -m venv .venv
.venv\Scripts\activate
python -m pip install -e ".[dev]"
scripts\run_demo.bat
```

Open `http://localhost:8501`. The launcher starts the receiver on TCP `5001`,
the three-node simulator, and Streamlit. The services continue until stopped
with `Ctrl+C`. For separate terminals:

```powershell
.venv\Scripts\python.exe -m wildlife_soundscape --auto-start --session-label demo
.venv\Scripts\python.exe -m wildlife_soundscape.runtime.simulator --host 127.0.0.1 --port 5001
.venv\Scripts\python.exe -m streamlit run src/wildlife_soundscape/dashboard/app.py
```

Run only the dashboard to inspect saved sessions. The generated demonstration
session contains 144 synthetic points, split 72 inside and 72 outside the
configured microphone triangle. Generate another additive session with:

```powershell
.venv\Scripts\python.exe -m wildlife_soundscape.tools.populate_demo_session --events 144
```

## Live hardware

Copy `config.example.json` to ignored `config.local.json`, set measured node
positions and the receiver address, then configure each firmware sketch with
the local Wi-Fi credentials. Flash Node 1 as master and Nodes 2 and 3 as
slaves. Follow `firmware/README.md` for the board/library baseline and
`docs/TECHNICAL_REFERENCE.md` for the protocol and wiring model. Start:

```powershell
scripts\run_live.bat
```

Confirm all three nodes connect and report healthy clocks before interpreting
events. Do not run the simulator alongside physical nodes.

## Data and recovery

Runtime outputs are in `data/database`, `data/events`, `data/recordings` and
`data/exports`. Validate a manifest and its audio with:

```powershell
wildlife-validate-dataset path\to\dataset-manifest.json --verify-checksums
```

The recovery utility targets only old heuristic `unknown` classifications. It
backs up SQLite before applying derived-feature updates:

```powershell
.venv\Scripts\python.exe -m wildlife_soundscape.tools.reprocess_unknown --limit 10
.venv\Scripts\python.exe -m wildlife_soundscape.tools.reprocess_unknown --apply
```

## Field validation checklist

Measure and record microphone coordinates, sensor identifiers, firmware build,
software commit, model identity, sample rate, calibration and environmental
conditions. Use at least nine known source positions, repeated impulses, held
out calibration trials, and inside/outside-array positions. Report success
rate, median/RMSE/95th-percentile localization error, signed bias, SNR and
rejected trials. Test reconnects, dropped packets, overload, power loss and
missing audio. Keep failed trials in the denominator.

Public datasets can benchmark components, but only a synchronized three-node
field dataset validates the complete system. Keep raw recordings, manifests,
calibration results and failed trials together.

## Troubleshooting

- Port `5001` or `8501` occupied: close the old service or change the local
  configuration and launcher readiness check together.
- No events: confirm all three nodes are connected, clocks are healthy, and
  the detection thresholds suit the background noise.
- Many `unknown` results: inspect the event's feature quality, SNR and reason;
  `unknown` is retained when evidence is invalid or ambiguous.
- Missing dashboard data: run from the repository root and check the SQLite
  path in the effective configuration.
