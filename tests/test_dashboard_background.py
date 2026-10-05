from pathlib import Path

from dashboard.background import (
    VIDEO_CONTROL,
    VIDEO_MARKUP,
    configure_plotly_theme,
    dashboard_header_markup,
)
import plotly.io as pio


ROOT = Path(__file__).resolve().parents[1]


def test_background_video_asset_and_markup_are_available():
    video = ROOT / "dashboard" / "static" / "bg.mp4"

    assert video.is_file()
    assert video.stat().st_size > 0
    assert '<video id="bgvid" autoplay muted loop playsinline preload="auto"' in VIDEO_MARKUP
    assert 'src="/app/static/bg.mp4"' in VIDEO_MARKUP


def test_background_controller_disables_mobile_autoplay():
    assert '(max-width: 767px)' in VIDEO_CONTROL
    assert 'video.pause()' in VIDEO_CONTROL
    assert 'video.removeAttribute("autoplay")' in VIDEO_CONTROL
    assert 'doc.body.appendChild(element)' in VIDEO_CONTROL
    assert 'doc.querySelectorAll(selector)' in VIDEO_CONTROL
    assert 'host.addEventListener("resize", sync)' in VIDEO_CONTROL
    assert 'video.onerror = () =>' in VIDEO_CONTROL
    assert 'qfleet-video-fallback' in VIDEO_CONTROL


def test_dashboard_header_contains_page_identity_and_badge():
    markup = dashboard_header_markup("Fuels & Carbon <Audit>")

    assert "QFleet" in markup
    assert "Synthetic data / SIH 2026" in markup
    assert "Fuels &amp; Carbon &lt;Audit&gt;" in markup


def test_plotly_theme_is_transparent_and_uses_requested_palette():
    configure_plotly_theme()

    for template_name in ("plotly", "plotly_dark"):
        layout = pio.templates[template_name].layout
        assert layout.paper_bgcolor == "rgba(0,0,0,0)"
        assert layout.plot_bgcolor == "rgba(0,0,0,0)"
        assert layout.font.color == "#f1f5f9"
        assert layout.xaxis.gridcolor == "rgba(255,255,255,.12)"
        assert layout.yaxis.gridcolor == "rgba(255,255,255,.12)"
        assert list(layout.colorway) == [
            "#22d3ee", "#34d399", "#fbbf24", "#f87171", "#a78bfa"
        ]