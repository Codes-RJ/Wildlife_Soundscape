"""Monochrome dashboard styling and scientifically cautious learning guides."""

from __future__ import annotations

from html import escape
from typing import Literal

import streamlit as st


ThemeName = Literal["Black & White"]


def get_theme_css(theme: str = "Black & White") -> str:
    """Return monochrome application chrome; charts retain semantic colors."""
    del theme
    return """
    <style>
        .stApp, [data-testid="stAppViewContainer"], [data-testid="stHeader"] {
            background: #ffffff !important;
            color: #111111 !important;
            font-family: 'Segoe UI', sans-serif;
        }
        section[data-testid="stSidebar"] {
            background: #f5f5f5 !important;
            border-right: 1px solid #d4d4d4;
        }
        h1, h2, h3, h4, h5 {
            color: #111111 !important;
            letter-spacing: -0.02em;
        }
        h1 { border-bottom: 2px solid #111111; padding-bottom: 12px; }
        a { color: #111111 !important; text-decoration: underline; }
        [data-testid="stMetric"], [data-testid="stExpander"],
        .student-card, .wildlife-banner {
            background: #ffffff !important;
            color: #111111;
            border: 1px solid #d4d4d4 !important;
            border-radius: 8px;
            box-shadow: none !important;
        }
        [data-testid="stMetric"] { padding: 16px 18px; }
        [data-testid="stMetricValue"] {
            color: #111111 !important;
            font-weight: 700;
        }
        [data-testid="stMetricDelta"], [data-testid="stAlert"] {
            color: #333333 !important;
            background: #f5f5f5 !important;
        }
        [data-testid="stAlert"] svg { fill: #333333 !important; }
        [data-testid="stAlert"] p { color: #333333 !important; }
        button[kind="primary"], [data-testid="stBaseButton-primary"] {
            color: #ffffff !important;
            background: #111111 !important;
            border: 1px solid #111111 !important;
        }
        button[kind="secondary"], [data-testid="stBaseButton-secondary"] {
            color: #111111 !important;
            background: #ffffff !important;
            border: 1px solid #777777 !important;
        }
        .wildlife-banner { padding: 18px 20px; margin-bottom: 20px; }
        .animal-chip {
            display: inline-block;
            padding: 5px 9px;
            border: 1px solid #777777;
            border-radius: 999px;
            color: #222222;
            background: #fafafa;
            font-size: 0.82rem;
        }
        .student-card { padding: 14px 16px; margin: 12px 0; }
        .student-badge {
            display: inline-block;
            padding: 2px 7px;
            border: 1px solid #777777;
            border-radius: 999px;
            color: #333333;
            font-size: 0.72rem;
            text-transform: uppercase;
            letter-spacing: 0.04em;
        }
    </style>
    """


def render_indian_wildlife_banner() -> None:
    """Render the project banner without implying validated species coverage."""
    st.markdown(
        """
        <div class="wildlife-banner">
            <div style="display:flex;align-items:center;justify-content:space-between;flex-wrap:wrap;gap:14px;">
                <div>
                    <h3 style="margin:0;color:#111111;font-size:1.45rem;">
                        Indian Wildlife Bioacoustics Observatory
                    </h3>
                    <p style="margin:4px 0 0 0;font-size:0.93rem;color:#555555;">
                        Research prototype for synchronized acoustic monitoring,
                        localization, evidence review, and soundscape analysis
                    </p>
                </div>
                <div style="display:flex;gap:6px;flex-wrap:wrap;">
                    <span class="animal-chip">Bengal tiger</span>
                    <span class="animal-chip">Asian elephant</span>
                    <span class="animal-chip">Indian peafowl</span>
                    <span class="animal-chip">One-horned rhinoceros</span>
                    <span class="animal-chip">Great hornbill</span>
                </div>
            </div>
            <p style="margin:12px 0 0 0;font-size:0.82rem;color:#666666;">
                Species names are study-context examples, not claims that the active
                classifier can identify them. Verify every species observation against
                reviewed audio and a validated model or expert annotation.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_student_explainer(
    *,
    topic_title: str,
    what_is_it: str,
    why_it_matters: str,
    how_to_read: str,
    real_world_example: str,
) -> None:
    """Render an escaped, student-friendly explanation card."""
    values = {
        "title": escape(topic_title),
        "what": escape(what_is_it),
        "why": escape(why_it_matters),
        "how": escape(how_to_read),
        "example": escape(real_world_example),
    }
    st.markdown(
        f"""
        <div class="student-card">
            <span class="student-badge">Student quick guide</span>
            <h4 style="margin:4px 0 10px 0;color:#111111;font-size:1.15rem;">
                {values['title']}
            </h4>
            <div style="font-size:0.93rem;line-height:1.55;color:#333333;">
                <p><b>What is it?</b> {values['what']}</p>
                <p><b>Why does it matter?</b> {values['why']}</p>
                <p><b>How should I read it?</b> {values['how']}</p>
                <p style="margin:0;color:#666666;font-style:italic;">
                    <b>Example:</b> {values['example']}
                </p>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_student_sidebar_guide() -> None:
    """Render a concise guide that separates measurements from inference."""
    with st.sidebar.expander("Bioacoustics quick guide", expanded=False):
        st.markdown(
            """
            - **TDOA:** Differences in sound-arrival time between microphones.
              With known geometry and calibration, these differences support an
              estimated source position; they do not guarantee an exact location.
            - **GCC-PHAT:** A cross-correlation method used to estimate delay.
              Reverberation and noise can still produce incorrect peaks.
            - **ACI:** Variation in sound intensity across time and frequency.
              It can describe a soundscape but is not a direct biodiversity count.
            - **NDSI:** A normalized comparison of energy in configured frequency
              bands. Its ecological meaning depends on site, species, and noise.
            - **Spectral centroid:** The frequency-weighted center of signal energy.
              It describes a recording but does not identify a species by itself.
            """
        )


def render_complete_guide_view() -> None:
    """Render a cautious walkthrough for exploring and interpreting the dashboard."""
    st.title("Student Learning & Human Analysis Guide")
    st.caption(
        "Use the dashboard to inspect acoustic evidence. Keep direct measurements, "
        "model outputs, and ecological interpretations separate."
    )

    with st.expander("1. Explore a live or recorded session", expanded=True):
        st.markdown(
            """
            1. In **Live Monitor**, confirm that all expected nodes are connected and
               that clock, packet, and processing health are acceptable.
            2. Treat each displayed location as an **estimate**. Review residuals,
               correlation quality, calibration status, and whether the estimate lies
               inside or outside the microphone array.
            3. In **Research Analysis**, choose one session and inspect activity,
               environmental, soundscape, and spatial summaries together.
            4. In **Review recordings**, listen to synchronized microphone evidence,
               inspect processing-stage status, and download the session manifest.
            """
        )

    with st.expander("2. Interpret plots without overclaiming", expanded=True):
        st.markdown(
            """
            - A detection establishes that the detector found an acoustic event. It
              does not by itself establish an animal or species observation.
            - A classifier label is a model result. Species-level reporting requires
              a validated model, preserved confidence information, and preferably
              expert review of held-out recordings.
            - Repeated localization points show repeated estimated sound origins.
              They do not prove a nest, territory, individual identity, or movement
              path without independent evidence.
            - Environmental associations are correlations within the sampled data.
              Weather, time, recorder behavior, habitat, and detection bias may all
              contribute; correlation is not causation.
            - ACI, NDSI, entropy, and related indices summarize recordings under
              chosen parameters. Compare like-for-like recording schedules and
              settings, and do not translate one index directly into biodiversity.
            """
        )

    with st.expander("3. Understand the main measurements", expanded=False):
        st.markdown(
            """
            | Measurement | What it supports | Important limitation |
            |---|---|---|
            | TDOA | Relative arrival delays and source-position estimation | Sensitive to geometry, synchronization, reverberation, and calibration |
            | GCC-PHAT peak | Candidate delay and correlation evidence | A strong peak can still be an echo or interference |
            | Spectrogram | Energy distribution across time and frequency | Color depends on scaling and is not a species label |
            | MFCCs | Compact spectral-shape features for models | Not an individual or species fingerprint on their own |
            | ACI | Short-term spectro-temporal variation | Influenced by weather, insects, noise, gain, and analysis settings |
            | NDSI | Relative energy in selected acoustic bands | Band definitions do not cleanly separate nature from machines everywhere |
            """
        )

    with st.expander("4. Make a defensible research claim", expanded=False):
        st.markdown(
            """
            Before reporting results, preserve raw audio and manifests, document
            microphone geometry and calibration, retain failed trials, lock the test
            split by site/session/day groups, and report uncertainty alongside summary
            metrics. Synthetic demonstrations validate software behavior only; final
            localization and classification claims require synchronized field data and
            independently reviewed ground truth.
            """
        )
