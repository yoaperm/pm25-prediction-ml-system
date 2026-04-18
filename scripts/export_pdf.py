"""
Export c4_architecture.md to PDF with rendered Mermaid diagrams.

Pipeline:
  1. Extract mermaid code blocks from the markdown
  2. Render each to inline SVG via kroki.io API (no local install needed)
  3. Build a self-contained HTML with embedded SVGs
  4. Use Microsoft Edge headless to print HTML → PDF
"""

import base64
import re
import subprocess
import sys
import tempfile
import urllib.request
import urllib.error
from pathlib import Path

EDGE    = "/Applications/Microsoft Edge.app/Contents/MacOS/Microsoft Edge"
MD_FILE = Path(__file__).parent.parent / "reports" / "c4_architecture.md"
OUT_PDF = Path(__file__).parent.parent / "reports" / "c4_architecture.pdf"
KROKI   = "https://kroki.io/mermaid/svg"

# ── 1. Read source markdown ───────────────────────────────────────────────────
md_text = MD_FILE.read_text(encoding="utf-8")

# ── 2. Render each mermaid block → SVG via kroki.io ──────────────────────────
MERMAID_FENCE = re.compile(r"```mermaid\n(.*?)```", re.DOTALL)

def render_svg(diagram: str) -> str:
    """POST diagram source to kroki.io and return SVG string."""
    payload = diagram.strip().encode("utf-8")
    req = urllib.request.Request(
        f"{KROKI}",
        data=payload,
        headers={
            "Content-Type": "text/plain",
            "User-Agent": "curl/7.88.1",
            "Accept": "*/*",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            svg = resp.read().decode("utf-8")
        # Remove fixed width/height so SVG scales to container
        svg = re.sub(r'\s(width|height)="[^"]*"', '', svg, count=2)
        return svg
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")[:300]
        print(f"  [WARN] kroki.io HTTP {exc.code}: {body}")
        label = diagram.strip().split("\n")[0][:60]
        return f'<div style="border:2px dashed #ccc;padding:20px;color:#888">[Diagram not rendered: {label}]</div>'
    except urllib.error.URLError as exc:
        print(f"  [WARN] kroki.io failed: {exc}")
        label = diagram.strip().split("\n")[0][:60]
        return f'<div style="border:2px dashed #ccc;padding:20px;color:#888">[Diagram not rendered: {label}]</div>'

diagrams = MERMAID_FENCE.findall(md_text)
print(f"Found {len(diagrams)} mermaid diagram(s). Rendering via kroki.io...")
svgs = []
for i, d in enumerate(diagrams, 1):
    first_line = d.strip().split("\n")[0]
    print(f"  [{i}/{len(diagrams)}] {first_line[:50]}...")
    svgs.append(render_svg(d))

# ── 3. Convert markdown to HTML (markdown-it-py) ─────────────────────────────
try:
    from markdown_it import MarkdownIt
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "markdown-it-py"])
    from markdown_it import MarkdownIt

mdit = MarkdownIt()
raw_html = mdit.render(md_text)

# ── 4. Replace ```mermaid blocks with pre-rendered SVGs ──────────────────────
# markdown-it encodes fenced blocks as <pre><code class="language-mermaid">
svg_iter = iter(svgs)
def _replace_with_svg(m):
    svg = next(svg_iter, "<div>[missing SVG]</div>")
    return f'<div class="diagram">{svg}</div>'

body_html = re.sub(
    r'<pre><code class="language-mermaid">.*?</code></pre>',
    _replace_with_svg,
    raw_html,
    flags=re.DOTALL,
)

# ── 5. Build the final HTML page ──────────────────────────────────────────────
PAGE = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<title>C4 Architecture — FoonAlert PM2.5</title>
<style>
  :root {{ font-family: "Helvetica Neue", Arial, sans-serif; font-size: 12.5px; line-height: 1.6; }}
  body {{ max-width: 1080px; margin: 0 auto; padding: 32px 44px; color: #1a1a1a; }}
  h1 {{ font-size: 1.75em; border-bottom: 2.5px solid #4f46e5; padding-bottom: 6px; margin-bottom: 0.4em; }}
  h2 {{ font-size: 1.3em; color: #4f46e5; margin-top: 2.8em; border-bottom: 1px solid #ddd; padding-bottom: 4px; }}
  h3 {{ font-size: 1.05em; color: #374151; margin-top: 1.6em; }}
  h4 {{ font-size: 0.95em; color: #6b7280; margin-top: 1.2em; }}
  code {{ background: #f3f4f6; padding: 1px 5px; border-radius: 3px; font-size: 0.88em; font-family: "SF Mono", Consolas, monospace; }}
  pre code {{ background: none; padding: 0; }}
  blockquote {{ border-left: 4px solid #818cf8; background: #f5f3ff; margin: 1em 0 1em 0; padding: 8px 16px; border-radius: 4px; color: #4338ca; font-style: italic; }}
  table {{ border-collapse: collapse; width: 100%; margin: 1em 0; font-size: 0.88em; }}
  th {{ background: #4f46e5; color: white; padding: 8px 12px; text-align: left; font-weight: 600; }}
  td {{ border: 1px solid #e5e7eb; padding: 6px 12px; vertical-align: top; }}
  tr:nth-child(even) td {{ background: #f9fafb; }}
  strong {{ color: #1e1b4b; }}
  .diagram {{ background: #f8faff; border: 1px solid #c7d2fe; border-radius: 10px;
              padding: 20px 12px; margin: 1.8em 0; text-align: center; overflow-x: auto; }}
  .diagram svg {{ max-width: 100%; height: auto; }}
  hr {{ border: none; border-top: 1px solid #e5e7eb; margin: 2.5em 0; }}
  ul, ol {{ padding-left: 1.6em; }}
  li {{ margin: 0.25em 0; }}
  @media print {{
    @page {{ size: A4; margin: 15mm 18mm; }}
    body {{ max-width: none; padding: 0; font-size: 11px; }}
    h2 {{ page-break-before: always; }}
    h2:first-of-type {{ page-break-before: avoid; }}
    .diagram {{ page-break-inside: avoid; border: 1px solid #c7d2fe; }}
    a {{ text-decoration: none; color: inherit; }}
    table {{ font-size: 10px; }}
  }}
</style>
</head>
<body>
{body_html}
</body>
</html>"""

# ── 6. Write HTML to temp file ────────────────────────────────────────────────
with tempfile.NamedTemporaryFile(suffix=".html", mode="w", encoding="utf-8", delete=False) as f:
    f.write(PAGE)
    html_path = Path(f.name)

print(f"\nHTML written → {html_path}")

# ── 7. Edge headless → PDF ────────────────────────────────────────────────────
print(f"Running Edge headless → {OUT_PDF}")
cmd = [
    EDGE,
    "--headless=new",
    "--disable-gpu",
    "--no-sandbox",
    "--run-all-compositor-stages-before-draw",
    f"--print-to-pdf={OUT_PDF}",
    "--print-to-pdf-no-header",
    f"file://{html_path}",
]
result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)

if OUT_PDF.exists() and OUT_PDF.stat().st_size > 10_000:
    size_kb = OUT_PDF.stat().st_size // 1024
    print(f"\nPDF exported successfully: {OUT_PDF}  ({size_kb} KB)")
else:
    print("Edge stderr:", result.stderr[:800])
    print(f"\nOpen HTML manually in Edge/Safari and press Cmd+P → Save as PDF:")
    print(f"  file://{html_path}")
