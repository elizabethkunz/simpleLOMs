"""
schematics/viewer.py
--------------------
Wrap a rendered SVG in a self-contained, zoomable HTML document.

No external libraries or CDNs: pan/zoom is ~40 lines of vanilla JavaScript
operating on the SVG ``viewBox`` (wheel to zoom about the cursor, drag to pan,
double-click to reset).  Because it is fully self-contained it works offline
and can be embedded in a Jupyter cell via a sandboxed ``srcdoc`` iframe (which,
unlike raw ``<script>`` in cell output, still executes in JupyterLab).
"""
from __future__ import annotations

import html as _html
import uuid

from simpleLOMs.schematics.netlist import Schematic
from simpleLOMs.schematics.render_svg import schematic_to_svg, Theme


_PANZOOM_JS = """
(function(){
  var wrap = document.getElementById('%(uid)s');
  var svg = wrap.querySelector('svg');
  var vb = svg.getAttribute('viewBox').split(/\\s+/).map(Number);
  var base = vb.slice();
  var view = vb.slice();
  function apply(){ svg.setAttribute('viewBox', view.join(' ')); }
  function pt(e){
    var r = svg.getBoundingClientRect();
    var w = r.width || svg.clientWidth || view[2];
    var h = r.height || svg.clientHeight || view[3];
    return { x: view[0] + (e.clientX - r.left)/w*view[2],
             y: view[1] + (e.clientY - r.top)/h*view[3] };
  }
  wrap.addEventListener('wheel', function(e){
    e.preventDefault();
    var p = pt(e);
    var k = e.deltaY < 0 ? 0.9 : 1.1;
    var nw = Math.min(base[2]*8, Math.max(base[2]*0.15, view[2]*k));
    var nh = nw * base[3]/base[2];
    view[0] = p.x - (p.x - view[0]) * nw/view[2];
    view[1] = p.y - (p.y - view[1]) * nh/view[3];
    view[2] = nw; view[3] = nh; apply();
  }, {passive:false});
  var drag=false, last=null;
  wrap.addEventListener('mousedown', function(e){ drag=true; last=e; wrap.style.cursor='grabbing'; });
  window.addEventListener('mouseup', function(){ drag=false; wrap.style.cursor='grab'; });
  window.addEventListener('mousemove', function(e){
    if(!drag) return;
    var r = svg.getBoundingClientRect();
    var w = r.width || svg.clientWidth || view[2];
    var h = r.height || svg.clientHeight || view[3];
    view[0] -= (e.clientX-last.clientX)/w*view[2];
    view[1] -= (e.clientY-last.clientY)/h*view[3];
    last=e; apply();
  });
  wrap.addEventListener('dblclick', function(){ view=base.slice(); apply(); });
})();
"""


def _panzoom_block(uid: str, svg: str, height: int) -> str:
    return (
        f'<div id="{uid}" class="loms-schematic" '
        f'style="width:100%;height:{height}px;overflow:hidden;cursor:grab;'
        f'background:{Theme.bg};border:1px solid #e6e8ec;border-radius:10px;">'
        f'{svg}</div>'
        f'<script>{_PANZOOM_JS % {"uid": uid}}</script>'
    )


def _sized_svg(sch: Schematic) -> tuple[str, int]:
    """Render an SVG that fills its container (width/height 100%)."""
    svg = schematic_to_svg(sch, standalone=False)
    svg = svg.replace('<svg ', '<svg style="width:100%;height:100%;display:block" ', 1)
    return svg, 360


def schematic_to_html(sch: Schematic, title: str | None = None,
                      height: int = 360) -> str:
    """Full standalone HTML page with a zoomable schematic."""
    uid = "sch_" + uuid.uuid4().hex[:8]
    svg, _ = _sized_svg(sch)
    page_title = title or sch.meta.get("title", "Circuit schematic")
    hint = ("scroll to zoom &middot; drag to pan &middot; double-click to reset")
    return (
        "<!doctype html><html><head><meta charset='utf-8'>"
        f"<title>{_html.escape(page_title)}</title>"
        "<meta name='viewport' content='width=device-width, initial-scale=1'>"
        "<style>"
        f"body{{margin:0;padding:24px;background:#f6f7f9;color:{Theme.title};"
        f"font-family:{Theme.font};}}"
        ".loms-hint{color:#8b93a1;font-size:12px;margin:10px 2px 0;}"
        "</style></head><body>"
        f"{_panzoom_block(uid, svg, height)}"
        f"<div class='loms-hint'>{hint}</div>"
        "</body></html>"
    )


def schematic_to_iframe(sch: Schematic, height: int = 380) -> str:
    """HTML for inline Jupyter display: a sandboxed iframe running the viewer."""
    doc = schematic_to_html(sch, height=height - 20)
    srcdoc = _html.escape(doc, quote=True)
    return (
        f'<iframe srcdoc="{srcdoc}" '
        f'style="width:100%;height:{height}px;border:0;" '
        f'sandbox="allow-scripts"></iframe>'
    )
