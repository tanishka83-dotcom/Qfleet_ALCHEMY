"""Shared dashboard background video and chart styling."""

from __future__ import annotations

from html import escape

import plotly.io as pio
import streamlit as st


VIDEO_MARKUP = """
<div class="bg-shade" aria-hidden="true"></div>
<video id="bgvid" autoplay muted loop playsinline preload="auto"
       src="/app/static/bg.mp4" aria-hidden="true">
</video>
"""

VIDEO_CONTROL = """
<script>
(() => {
  const host = window.parent;
  const media = host.matchMedia("(max-width: 767px)");
  const sync = () => {
      const doc = host.document;
      const retainInBody = (selector) => {
        const matches = Array.from(doc.querySelectorAll(selector));
        const retained = matches.find(element => element.parentElement === doc.body);
        const element = retained || matches[0];
        if (!element) return null;
        matches.filter(match => match !== element).forEach(match => match.remove());
        if (element.parentElement !== doc.body) doc.body.appendChild(element);
        return element;
      };
      const video = retainInBody("#bgvid");
      retainInBody(".bg-shade");
      if (!video) return;
    video.onerror = () => {
      doc.documentElement.classList.add("qfleet-video-fallback");
      doc.body.classList.add("qfleet-video-fallback");
    };
    video.onloadeddata = () => {
      doc.documentElement.classList.remove("qfleet-video-fallback");
      doc.body.classList.remove("qfleet-video-fallback");
    };
    if (media.matches) {
      video.pause();
      video.removeAttribute("autoplay");
    } else {
      video.setAttribute("autoplay", "");
        if (video.paused) video.play().catch(() => {});
    }
  };
  if (!host.__qfleetBackgroundSync) {
    host.__qfleetBackgroundSync = sync;
    host.addEventListener("resize", sync);
    media.addEventListener("change", sync);
    new host.MutationObserver(sync).observe(host.document.documentElement, {
      childList: true,
      subtree: true
    });
  }
  host.__qfleetBackgroundSync();
})();
</script>
"""


def configure_plotly_theme() -> None:
    colorway = ["#22d3ee", "#34d399", "#fbbf24", "#f87171", "#a78bfa"]
    for name in ("plotly", "plotly_dark"):
        template = pio.templates[name]
        template.layout.update(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font={"color": "#f1f5f9"},
            colorway=colorway,
            xaxis={"gridcolor": "rgba(255,255,255,.12)"},
            yaxis={"gridcolor": "rgba(255,255,255,.12)"},
        )
        pio.templates[name] = template


def inject_background() -> None:
    """Inject the shared video once at the app root and configure chart defaults."""
    configure_plotly_theme()
    st.markdown(VIDEO_MARKUP, unsafe_allow_html=True)
    st.components.v1.html(VIDEO_CONTROL, height=0, scrolling=False)


def dashboard_header_markup(page_title: str) -> str:
  """Build escaped shared header markup for the active dashboard page."""
  return (
    '<div class="qfleet-header">'
    '<div class="qfleet-brand"><span aria-hidden="true">⚓</span> QFleet</div>'
    f'<div class="qfleet-page-title">{escape(page_title)}</div>'
    '<div class="qfleet-badge">Synthetic data / SIH 2026</div>'
    '</div>'
  )


def render_dashboard_header(page_title: str) -> None:
  """Render the shared header below the persistent background layer."""
  st.markdown(dashboard_header_markup(page_title), unsafe_allow_html=True)