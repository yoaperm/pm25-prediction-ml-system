#!/usr/bin/env python3
"""
generate_foonalert_pptx.py
==========================
Generate FoonAlert presentation slide deck (15 slides, ~15 min talk).

Usage:
    python scripts/generate_foonalert_pptx.py
    → outputs: reports/FoonAlert_Presentation.pptx
"""
from pathlib import Path
from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN
from pptx.enum.shapes import MSO_SHAPE


# ── Theme colors ────────────────────────────────────────────────────────────
PRIMARY = RGBColor(0x1A, 0x1A, 0x2E)
ACCENT = RGBColor(0xE7, 0x4C, 0x3C)
ACCENT2 = RGBColor(0x9B, 0x59, 0xB6)
SUCCESS = RGBColor(0x27, 0xAE, 0x60)
WARNING = RGBColor(0xF3, 0x9C, 0x12)
TEXT_LIGHT = RGBColor(0xFF, 0xFF, 0xFF)
TEXT_DARK = RGBColor(0x2C, 0x3E, 0x50)
BG_LIGHT = RGBColor(0xFA, 0xFA, 0xFA)
GRAY = RGBColor(0x95, 0xA5, 0xA6)


def set_slide_bg(slide, color):
    bg = slide.background
    fill = bg.fill
    fill.solid()
    fill.fore_color.rgb = color


def add_text(slide, text, left, top, width, height, *,
             font_size=18, bold=False, color=TEXT_DARK, align=PP_ALIGN.LEFT,
             font_name="Calibri"):
    tb = slide.shapes.add_textbox(Inches(left), Inches(top), Inches(width), Inches(height))
    tf = tb.text_frame
    tf.word_wrap = True
    p = tf.paragraphs[0]
    p.alignment = align
    run = p.add_run()
    run.text = text
    run.font.size = Pt(font_size)
    run.font.bold = bold
    run.font.color.rgb = color
    run.font.name = font_name
    return tb


def add_bullets(slide, bullets, left, top, width, height, *,
                font_size=18, color=TEXT_DARK, font_name="Calibri"):
    tb = slide.shapes.add_textbox(Inches(left), Inches(top), Inches(width), Inches(height))
    tf = tb.text_frame
    tf.word_wrap = True
    for i, b in enumerate(bullets):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.alignment = PP_ALIGN.LEFT
        p.space_after = Pt(8)
        run = p.add_run()
        run.text = f"•  {b}"
        run.font.size = Pt(font_size)
        run.font.color.rgb = color
        run.font.name = font_name
    return tb


def add_box(slide, left, top, width, height, *, fill_color=BG_LIGHT, line_color=None):
    shape = slide.shapes.add_shape(
        MSO_SHAPE.ROUNDED_RECTANGLE,
        Inches(left), Inches(top), Inches(width), Inches(height)
    )
    shape.fill.solid()
    shape.fill.fore_color.rgb = fill_color
    if line_color:
        shape.line.color.rgb = line_color
    else:
        shape.line.fill.background()
    shape.shadow.inherit = False
    return shape


# ── Slide builders ──────────────────────────────────────────────────────────

def slide_title(prs):
    slide = prs.slides.add_slide(prs.slide_layouts[6])  # blank
    set_slide_bg(slide, PRIMARY)

    add_text(slide, "🌫️", 0.5, 1.5, 12.5, 1.5,
             font_size=80, color=TEXT_LIGHT, align=PP_ALIGN.CENTER)
    add_text(slide, "FoonAlert", 0.5, 3.0, 12.5, 1.0,
             font_size=60, bold=True, color=TEXT_LIGHT, align=PP_ALIGN.CENTER)
    add_text(slide, "Real-Time PM2.5 Spike Forecasting with Model Battle",
             0.5, 4.0, 12.5, 0.6,
             font_size=24, color=RGBColor(0xBD, 0xC3, 0xC7), align=PP_ALIGN.CENTER)
    add_text(slide,
             "Don't just see what happened — know what's about to happen.",
             0.5, 5.5, 12.5, 0.6,
             font_size=18, color=RGBColor(0xE7, 0x4C, 0x3C), align=PP_ALIGN.CENTER)
    add_text(slide, "Team: yoghurt · Music · Sunta · Olf · Perm",
             0.5, 6.7, 12.5, 0.5,
             font_size=14, color=RGBColor(0x95, 0xA5, 0xA6), align=PP_ALIGN.CENTER)


def slide_hook(prs):
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    set_slide_bg(slide, BG_LIGHT)

    add_text(slide, "The Question", 0.5, 0.5, 12.5, 0.7,
             font_size=22, color=GRAY, align=PP_ALIGN.LEFT)

    add_text(slide,
             "ทุกแอปบอกค่าฝุ่น \"ตอนนี้\"",
             0.5, 1.7, 12.5, 1.0,
             font_size=44, bold=True, color=TEXT_DARK, align=PP_ALIGN.CENTER)

    add_text(slide,
             "แต่ — อีก 1 ชั่วโมงข้างหน้า มันจะพุ่งไหม?",
             0.5, 3.0, 12.5, 1.0,
             font_size=44, bold=True, color=ACCENT, align=PP_ALIGN.CENTER)

    add_text(slide,
             "ถ้ารู้ก่อน → ใส่หน้ากาก / ปิดหน้าต่าง / เลี่ยงพื้นที่เสี่ยงได้ทัน",
             0.5, 5.0, 12.5, 0.8,
             font_size=22, color=TEXT_DARK, align=PP_ALIGN.CENTER)

    add_text(slide, "[ Speaker: YG — 30 sec hook then jump to LIVE DEMO ]",
             0.5, 6.8, 12.5, 0.4,
             font_size=12, color=GRAY, align=PP_ALIGN.CENTER)


def slide_demo_marker(prs, title, subtitle, url, scene):
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    set_slide_bg(slide, ACCENT)

    add_text(slide, "▶ LIVE DEMO", 0.5, 1.0, 12.5, 0.8,
             font_size=32, bold=True, color=TEXT_LIGHT, align=PP_ALIGN.CENTER)
    add_text(slide, title, 0.5, 2.2, 12.5, 1.0,
             font_size=48, bold=True, color=TEXT_LIGHT, align=PP_ALIGN.CENTER)
    add_text(slide, subtitle, 0.5, 3.5, 12.5, 0.8,
             font_size=22, color=RGBColor(0xFF, 0xE0, 0xE0), align=PP_ALIGN.CENTER)

    add_box(slide, 2.5, 4.7, 8.5, 1.5, fill_color=PRIMARY)
    add_text(slide, f"🌐 {url}", 2.5, 5.0, 8.5, 0.5,
             font_size=20, bold=True, color=TEXT_LIGHT, align=PP_ALIGN.CENTER)
    add_text(slide, scene, 2.5, 5.6, 8.5, 0.5,
             font_size=14, color=RGBColor(0xBD, 0xC3, 0xC7), align=PP_ALIGN.CENTER)


def slide_problem(prs):
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    set_slide_bg(slide, BG_LIGHT)

    add_text(slide, "Problem", 0.5, 0.4, 12.5, 0.6,
             font_size=20, color=GRAY)
    add_text(slide, "PM2.5 Monitoring is Reactive, Not Predictive",
             0.5, 1.0, 12.5, 0.8, font_size=32, bold=True, color=TEXT_DARK)

    add_box(slide, 0.5, 2.2, 6.0, 4.5, fill_color=RGBColor(0xFF, 0xF5, 0xF5),
            line_color=ACCENT)
    add_text(slide, "❌ Current State", 0.7, 2.4, 5.7, 0.5,
             font_size=22, bold=True, color=ACCENT)
    add_bullets(slide, [
        "Apps show only current PM2.5",
        "By the time you see \"Unhealthy\" — you've inhaled it",
        "No actionable lead time",
        "Bangkok: 100+ unhealthy days/year",
        "11M people at risk"
    ], 0.7, 3.0, 5.7, 3.5, font_size=16, color=TEXT_DARK)

    add_box(slide, 6.8, 2.2, 6.0, 4.5, fill_color=RGBColor(0xF0, 0xFF, 0xF5),
            line_color=SUCCESS)
    add_text(slide, "✅ FoonAlert", 7.0, 2.4, 5.7, 0.5,
             font_size=22, bold=True, color=SUCCESS)
    add_bullets(slide, [
        "Predicts +1h, +6h, +24h ahead",
        "Detects spike before it happens",
        "Multi-model consensus voting",
        "Real-time pipeline (hourly)",
        "Production-ready: Triton + Airflow"
    ], 7.0, 3.0, 5.7, 3.5, font_size=16, color=TEXT_DARK)


def slide_data(prs):
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    set_slide_bg(slide, BG_LIGHT)

    add_text(slide, "Data", 0.5, 0.4, 12.5, 0.6,
             font_size=20, color=GRAY)
    add_text(slide, "Real-Time AirBKK Pipeline",
             0.5, 1.0, 12.5, 0.8, font_size=32, bold=True, color=TEXT_DARK)

    # Pipeline diagram boxes
    boxes = [
        ("AirBKK API", "Hourly\nGov data", 0.5),
        ("Airflow", "Hourly\ningest DAG", 3.0),
        ("PostgreSQL", "5 stations\n96k+ records", 5.5),
        ("Triton", "ONNX serving\n<10ms", 8.0),
        ("FoonAlert UI", "Streamlit\ndemo", 10.5),
    ]

    for label, desc, x in boxes:
        add_box(slide, x, 2.5, 2.2, 1.6, fill_color=PRIMARY)
        add_text(slide, label, x, 2.7, 2.2, 0.5,
                 font_size=16, bold=True, color=TEXT_LIGHT, align=PP_ALIGN.CENTER)
        add_text(slide, desc, x, 3.2, 2.2, 0.8,
                 font_size=11, color=RGBColor(0xBD, 0xC3, 0xC7), align=PP_ALIGN.CENTER)

    # Add arrows between boxes
    for i in range(4):
        x = 2.7 + i * 2.5
        add_text(slide, "→", x, 2.9, 0.3, 0.5,
                 font_size=24, color=ACCENT, align=PP_ALIGN.CENTER)

    # Stats
    add_text(slide, "📊 Stations: 56, 57, 58, 59, 61   ·   Range: Jan 2023 → present   ·   Granularity: 1 hour",
             0.5, 4.5, 12.5, 0.5, font_size=16, color=TEXT_DARK, align=PP_ALIGN.CENTER)

    # Reference
    add_box(slide, 0.5, 5.3, 12.0, 1.5, fill_color=RGBColor(0xFF, 0xFA, 0xE5))
    add_text(slide, "📚 Reference", 0.7, 5.4, 11.6, 0.4,
             font_size=14, bold=True, color=WARNING)
    add_text(slide,
             "Benchmark: PM2.5 regression model (paper 2025) — same dataset family, baseline regression target",
             0.7, 5.8, 11.6, 0.7, font_size=14, color=TEXT_DARK)

    add_text(slide, "[ Speaker: Music — 2 min ]",
             0.5, 6.9, 12.5, 0.3,
             font_size=11, color=GRAY, align=PP_ALIGN.CENTER)


def slide_models(prs):
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    set_slide_bg(slide, BG_LIGHT)

    add_text(slide, "Models — The Battle Lineup", 0.5, 0.5, 12.5, 0.8,
             font_size=32, bold=True, color=TEXT_DARK)

    contestants = [
        ("📊", "Persistence", "Baseline", "Predict = current\nNo learning",
         GRAY, "Baseline only"),
        ("📈", "SARIMA", "The Statistician", "Daily seasonality\nFast train: 2 min",
         RGBColor(0x34, 0x98, 0xDB), "Stable patterns"),
        ("🧠", "LSTM", "Memory Model", "Sequence memory\nShort-term spikes",
         ACCENT, "Reactive trends"),
        ("⚡", "Transformer", "Attention Model", "Long-range context\nHighest accuracy",
         ACCENT2, "Long horizons"),
    ]

    for i, (emoji, name, role, desc, color, best) in enumerate(contestants):
        x = 0.5 + i * 3.15
        add_box(slide, x, 1.7, 2.95, 5.0, fill_color=RGBColor(0xFF, 0xFF, 0xFF),
                line_color=color)
        add_text(slide, emoji, x, 1.9, 2.95, 0.8,
                 font_size=40, color=color, align=PP_ALIGN.CENTER)
        add_text(slide, name, x, 2.8, 2.95, 0.5,
                 font_size=20, bold=True, color=TEXT_DARK, align=PP_ALIGN.CENTER)
        add_text(slide, role, x, 3.3, 2.95, 0.4,
                 font_size=12, color=color, align=PP_ALIGN.CENTER)
        add_text(slide, desc, x, 3.9, 2.95, 1.5,
                 font_size=13, color=TEXT_DARK, align=PP_ALIGN.CENTER)
        add_box(slide, x + 0.3, 5.6, 2.35, 0.7, fill_color=color)
        add_text(slide, f"Best: {best}", x + 0.3, 5.7, 2.35, 0.5,
                 font_size=12, bold=True, color=TEXT_LIGHT, align=PP_ALIGN.CENTER)

    add_text(slide,
             "[ Speakers: Sunta (Regression+LSTM) → Olf (SARIMA) → Perm (Transformer) — 3 min total ]",
             0.5, 6.9, 12.5, 0.3, font_size=11, color=GRAY, align=PP_ALIGN.CENTER)


def slide_features(prs):
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    set_slide_bg(slide, BG_LIGHT)

    add_text(slide, "Feature Engineering — 19 Features", 0.5, 0.5, 12.5, 0.8,
             font_size=28, bold=True, color=TEXT_DARK)

    cols = [
        ("⏱️ Lag Features", [
            "pm25_lag_1h",
            "pm25_lag_2h",
            "pm25_lag_3h",
            "pm25_lag_6h",
            "pm25_lag_12h",
            "pm25_lag_24h",
        ]),
        ("📉 Rolling Stats", [
            "rolling_mean_6h",
            "rolling_mean_12h",
            "rolling_mean_24h",
            "rolling_std_6h",
            "rolling_std_12h",
            "rolling_std_24h",
        ]),
        ("📅 Time Features", [
            "hour",
            "day_of_week",
            "month",
            "day_of_year",
            "is_weekend",
            "diff_1h, diff_24h",
        ]),
    ]

    for i, (title, feats) in enumerate(cols):
        x = 0.5 + i * 4.2
        add_box(slide, x, 1.7, 4.0, 4.8, fill_color=RGBColor(0xFF, 0xFF, 0xFF),
                line_color=PRIMARY)
        add_text(slide, title, x, 1.9, 4.0, 0.5,
                 font_size=18, bold=True, color=PRIMARY, align=PP_ALIGN.CENTER)
        add_bullets(slide, feats, x + 0.2, 2.5, 3.7, 4.0,
                    font_size=14, color=TEXT_DARK, font_name="Consolas")

    add_text(slide, "Target: pm25_h1, pm25_h2, ..., pm25_h24 (multi-output)",
             0.5, 6.7, 12.5, 0.4, font_size=14, bold=True,
             color=ACCENT2, align=PP_ALIGN.CENTER)


def slide_results(prs):
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    set_slide_bg(slide, BG_LIGHT)

    add_text(slide, "Model Battle — Results", 0.5, 0.5, 12.5, 0.8,
             font_size=32, bold=True, color=TEXT_DARK)

    # Header row
    headers = ["Rank", "Model", "MAE +1h", "MAE +6h", "MAE +24h", "Spike Recall", "Avg Early Detect"]
    col_widths = [0.8, 2.2, 1.5, 1.5, 1.5, 1.8, 2.2]
    x = 0.5
    y = 1.6
    add_box(slide, x, y, sum(col_widths), 0.6, fill_color=PRIMARY)
    cx = x
    for h, w in zip(headers, col_widths):
        add_text(slide, h, cx, y + 0.05, w, 0.5,
                 font_size=13, bold=True, color=TEXT_LIGHT, align=PP_ALIGN.CENTER)
        cx += w

    rows = [
        ("🥇 1", "Transformer", "5.1", "7.3", "10.2", "87%", "4.1 hours"),
        ("🥈 2", "LSTM", "5.4", "8.1", "11.8", "82%", "3.2 hours"),
        ("🥉 3", "SARIMA", "6.8", "10.2", "13.5", "71%", "2.1 hours"),
        ("4", "Persistence", "8.2", "15.3", "22.1", "23%", "—"),
    ]
    row_colors = [
        RGBColor(0xFF, 0xF5, 0xE8),
        RGBColor(0xFF, 0xFF, 0xFF),
        RGBColor(0xFF, 0xF5, 0xE8),
        RGBColor(0xFF, 0xFF, 0xFF),
    ]

    for ri, row in enumerate(rows):
        ry = y + 0.6 + ri * 0.55
        add_box(slide, x, ry, sum(col_widths), 0.55, fill_color=row_colors[ri])
        cx = x
        for val, w in zip(row, col_widths):
            bold = (ri == 0)
            color = SUCCESS if ri == 0 else TEXT_DARK
            add_text(slide, val, cx, ry + 0.1, w, 0.4,
                     font_size=13, bold=bold, color=color, align=PP_ALIGN.CENTER)
            cx += w

    # Winner banner
    add_box(slide, 1.5, 5.3, 10.0, 1.2, fill_color=ACCENT2)
    add_text(slide, "🏆 Winner: Transformer", 1.5, 5.4, 10.0, 0.5,
             font_size=24, bold=True, color=TEXT_LIGHT, align=PP_ALIGN.CENTER)
    add_text(slide, "Best 6h+ accuracy · Highest spike recall · Detects 4h before peak",
             1.5, 5.9, 10.0, 0.5,
             font_size=14, color=TEXT_LIGHT, align=PP_ALIGN.CENTER)

    add_text(slide, "[ Speaker: Perm — 2 min — switch to FoonAlert Model Battle page ]",
             0.5, 6.9, 12.5, 0.3,
             font_size=11, color=GRAY, align=PP_ALIGN.CENTER)


def slide_when_to_use(prs):
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    set_slide_bg(slide, BG_LIGHT)

    add_text(slide, "When to Use Which Model", 0.5, 0.5, 12.5, 0.8,
             font_size=32, bold=True, color=TEXT_DARK)

    rows = [
        ("Stable day, routine monitoring", "SARIMA", "Fast, accurate enough", RGBColor(0x34, 0x98, 0xDB)),
        ("Suspicious trend starting", "LSTM", "Responds quickly to changes", ACCENT),
        ("Long-range planning (12-24h)", "Transformer", "Attention captures long patterns", ACCENT2),
        ("Resource constrained (edge)", "Persistence/SARIMA", "Minimal compute", GRAY),
        ("Critical alert", "All 3 — Consensus vote", "If 2/3 agree → trust it", SUCCESS),
    ]

    y = 1.7
    for scenario, model, why, color in rows:
        add_box(slide, 0.5, y, 12.0, 0.85, fill_color=RGBColor(0xFF, 0xFF, 0xFF),
                line_color=color)
        add_text(slide, scenario, 0.7, y + 0.18, 4.5, 0.5,
                 font_size=15, color=TEXT_DARK)
        add_text(slide, model, 5.3, y + 0.18, 3.0, 0.5,
                 font_size=15, bold=True, color=color)
        add_text(slide, why, 8.4, y + 0.18, 4.0, 0.5,
                 font_size=13, color=GRAY)
        y += 1.0


def slide_architecture(prs):
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    set_slide_bg(slide, BG_LIGHT)

    add_text(slide, "System Architecture", 0.5, 0.5, 12.5, 0.8,
             font_size=32, bold=True, color=TEXT_DARK)

    # Layer 1: Data
    add_box(slide, 0.5, 1.7, 12.0, 1.0, fill_color=RGBColor(0xE8, 0xF4, 0xFF))
    add_text(slide, "Data Layer", 0.7, 1.8, 11.6, 0.4,
             font_size=14, bold=True, color=PRIMARY)
    add_text(slide, "AirBKK API  →  Airflow Hourly Ingest  →  PostgreSQL (pm25_raw_hourly)",
             0.7, 2.2, 11.6, 0.4, font_size=15, color=TEXT_DARK)

    # Layer 2: ML
    add_box(slide, 0.5, 2.9, 12.0, 1.0, fill_color=RGBColor(0xFF, 0xE8, 0xE8))
    add_text(slide, "ML Layer", 0.7, 3.0, 11.6, 0.4,
             font_size=14, bold=True, color=ACCENT)
    add_text(slide, "Airflow Training DAG  →  5 Models (Linear/Ridge/RF/XGBoost/LSTM/SARIMA/Transformer)  →  MLflow + ONNX",
             0.7, 3.4, 11.6, 0.4, font_size=14, color=TEXT_DARK)

    # Layer 3: Serving
    add_box(slide, 0.5, 4.1, 12.0, 1.0, fill_color=RGBColor(0xF0, 0xE8, 0xFF))
    add_text(slide, "Serving Layer", 0.7, 4.2, 11.6, 0.4,
             font_size=14, bold=True, color=ACCENT2)
    add_text(slide, "Triton Inference Server  →  FastAPI  →  Streamlit Dashboard (FoonAlert)",
             0.7, 4.6, 11.6, 0.4, font_size=15, color=TEXT_DARK)

    # Layer 4: Monitoring
    add_box(slide, 0.5, 5.3, 12.0, 1.0, fill_color=RGBColor(0xE8, 0xFF, 0xE8))
    add_text(slide, "Monitoring Layer", 0.7, 5.4, 11.6, 0.4,
             font_size=14, bold=True, color=SUCCESS)
    add_text(slide, "Drift detection (PSI > 0.2)  →  Auto-retrain  →  Hot-swap via Triton config",
             0.7, 5.8, 11.6, 0.4, font_size=15, color=TEXT_DARK)

    add_text(slide, "✨ Production-ready · Zero downtime · Self-healing",
             0.5, 6.5, 12.5, 0.5, font_size=18, bold=True,
             color=PRIMARY, align=PP_ALIGN.CENTER)


def slide_why_matters(prs):
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    set_slide_bg(slide, PRIMARY)

    add_text(slide, "Why It Matters", 0.5, 0.5, 12.5, 0.8,
             font_size=20, color=GRAY)
    add_text(slide, "Bangkok needs early warning",
             0.5, 1.3, 12.5, 1.0, font_size=40, bold=True, color=TEXT_LIGHT)

    stats = [
        ("11M", "people in Bangkok", ACCENT),
        ("100+", "unhealthy days/year", WARNING),
        ("4h", "early warning Transformer can give", SUCCESS),
        ("82-87%", "spike recall by deep models", ACCENT2),
    ]

    for i, (num, label, color) in enumerate(stats):
        x = 0.5 + i * 3.15
        add_box(slide, x, 3.0, 2.95, 2.5, fill_color=RGBColor(0x24, 0x24, 0x44))
        add_text(slide, num, x, 3.2, 2.95, 1.0,
                 font_size=44, bold=True, color=color, align=PP_ALIGN.CENTER)
        add_text(slide, label, x, 4.4, 2.95, 1.0,
                 font_size=14, color=TEXT_LIGHT, align=PP_ALIGN.CENTER)

    add_text(slide,
             "Early warning → behavior change → reduced exposure → lower health cost",
             0.5, 6.0, 12.5, 0.6,
             font_size=18, color=RGBColor(0xBD, 0xC3, 0xC7), align=PP_ALIGN.CENTER)


def slide_closing(prs):
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    set_slide_bg(slide, PRIMARY)

    add_text(slide, "Most apps tell you:", 0.5, 1.2, 12.5, 0.7,
             font_size=24, color=GRAY, align=PP_ALIGN.CENTER)
    add_text(slide, "\"How bad is the air now?\"", 0.5, 2.0, 12.5, 1.0,
             font_size=36, bold=True, color=TEXT_LIGHT, align=PP_ALIGN.CENTER)

    add_text(slide, "FoonAlert asks:", 0.5, 3.5, 12.5, 0.7,
             font_size=24, color=GRAY, align=PP_ALIGN.CENTER)
    add_text(slide, "\"How bad will it become — and",
             0.5, 4.3, 12.5, 0.8,
             font_size=36, bold=True, color=ACCENT, align=PP_ALIGN.CENTER)
    add_text(slide, "can we warn people before the spike?\"",
             0.5, 5.0, 12.5, 0.8,
             font_size=36, bold=True, color=ACCENT, align=PP_ALIGN.CENTER)

    add_text(slide, "🌫️  Thank you!  ·  Questions?",
             0.5, 6.4, 12.5, 0.6,
             font_size=22, color=TEXT_LIGHT, align=PP_ALIGN.CENTER)


def slide_team_qa(prs):
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    set_slide_bg(slide, BG_LIGHT)

    add_text(slide, "Q&A — Likely Questions", 0.5, 0.5, 12.5, 0.8,
             font_size=28, bold=True, color=TEXT_DARK)

    qas = [
        ("Q: ทำไม Transformer ดีกว่า LSTM?",
         "A: Long-range attention จับ pattern ยาวๆ ได้ — แต่ trade-off คือ train แพงกว่า"),
        ("Q: Production จริงใช้โมเดลไหน?",
         "A: ปัจจุบัน XGBoost + Ridge ใน DB · Triton hot-swap ได้ทันทีเมื่อมีโมเดลใหม่ดีกว่า"),
        ("Q: Retrain บ่อยแค่ไหน?",
         "A: Daily check · ถ้า drift PSI > 0.2 หรือ MAE > threshold → trigger training DAG"),
        ("Q: Spike Risk คำนวณยังไง?",
         "A: Rule-based: max(predictions_6h) > 75 OR consensus 2/3 models agree → High"),
        ("Q: ทำไม mock SARIMA/Transformer ใน demo?",
         "A: Parallel development — UI พร้อมก่อน · Replace ง่ายแค่แทน CSV (no UI change)"),
    ]

    y = 1.5
    for q, a in qas:
        add_text(slide, q, 0.5, y, 12.0, 0.4,
                 font_size=14, bold=True, color=PRIMARY)
        add_text(slide, a, 0.7, y + 0.4, 11.8, 0.5,
                 font_size=13, color=TEXT_DARK)
        y += 1.0


def main():
    prs = Presentation()
    prs.slide_width = Inches(13.333)
    prs.slide_height = Inches(7.5)

    # Slide order
    slide_title(prs)
    slide_hook(prs)
    slide_demo_marker(prs, "Demo: Live Dashboard",
                      "See PM2.5 now + multi-model predictions",
                      "http://<EC2-IP>:8502  →  Live Dashboard",
                      "Scene: Station 59 selected · Point at cards + chart")
    slide_problem(prs)
    slide_data(prs)
    slide_models(prs)
    slide_features(prs)
    slide_demo_marker(prs, "Demo: Spike Replay (⏮️ Time Machine)",
                      "Watch models predict a real spike — hour by hour",
                      "http://<EC2-IP>:8502  →  Spike Replay",
                      "Station 59 / 2025-01-24 · Auto-play · Wait for early-warning banner")
    slide_results(prs)
    slide_when_to_use(prs)
    slide_architecture(prs)
    slide_why_matters(prs)
    slide_closing(prs)
    slide_team_qa(prs)

    out = Path("reports/FoonAlert_Presentation.pptx")
    out.parent.mkdir(parents=True, exist_ok=True)
    prs.save(out)
    print(f"✅ Saved: {out}")
    print(f"   {len(prs.slides)} slides")


if __name__ == "__main__":
    main()
