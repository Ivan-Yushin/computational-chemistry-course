"""Просмотрщик ORCA под ячейкой ноутбука.

    import sys
    sys.path.append("/home/jupyter/project/computational-chemistry-course/tools")
    from orca_tools import show

    show("water.out")
    show("h2o2.out", "h2o2.mo5.cube")
    show("scan.out", height=700)
    show("qtaim_c6h6.out", "qtaim_c6h6.wfn", qtaim=True)
    show("elf_water.out", "elf_water.wfn", elf=True)
"""

import base64
import json
import os
import uuid
import zlib
from html import escape
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

try:
    from IPython.display import HTML, display
except ImportError:
    HTML = display = None

__all__ = ["show", "render_html", "VIEWER_URL"]

VIEWER_URL = os.environ.get(
    "ORCA_VIEWER_URL",
    "https://physchem-nsu.ru/scourses/comp-chem-course/extras/orca-viewer",
)


def _with_query(url, **params):
    parts = urlsplit(url)
    query = dict(parse_qsl(parts.query))
    query.update(params)
    return urlunsplit(parts._replace(query=urlencode(query)))


def _origin(url):
    parts = urlsplit(url)
    return "{}://{}".format(parts.scheme, parts.netloc)


def _payload(paths, compress, qtaim=False, elf=False):
    files = []
    for path in paths:
        with open(path, "rb") as fh:
            raw = fh.read()
        entry = {"name": os.path.basename(path)}
        if compress:
            entry["gz"] = base64.b64encode(zlib.compress(raw, 6)).decode("ascii")
        else:
            entry["text"] = raw.decode("utf-8", errors="replace")
        files.append(entry)
    data = {"type": "orca-viewer", "files": files}
    options = {k: True for k, on in (("qtaim", qtaim), ("elf", elf)) if on}
    if options:
        data["options"] = options
    text = json.dumps(data, ensure_ascii=True)
    return text.replace("</", "<\\/")  # «</» внутри <script> закрыл бы тег


def render_html(*paths, height=600, compress=True, qtaim=False, elf=False):
    """HTML вывода ячейки: iframe просмотрщика и скрипт передачи файлов."""
    if not paths:
        raise ValueError("Укажите хотя бы один файл: show('water.out')")
    missing = [p for p in paths if not os.path.isfile(p)]
    if missing:
        raise FileNotFoundError("Нет файла: " + ", ".join(missing))
    # к NEB-расчёту прикладываем путь <имя>_MEP_trj.xyz, если он лежит рядом
    paths = [os.fspath(p) for p in paths]
    for p in list(paths):
        trj = p[:-4] + "_MEP_trj.xyz"
        if p.endswith(".out") and os.path.isfile(trj) and os.path.abspath(trj) not in map(os.path.abspath, paths):
            paths.append(trj)
    # qtaim=True / elf=True: к .out прикладываем <имя>.wfn (orca_2aim) и сразу открываем вкладку QTAIM / ELF
    for p in list(paths):
        wfn = p[:-4] + ".wfn"
        if (qtaim or elf) and p.endswith(".out") and os.path.isfile(wfn) and os.path.abspath(wfn) not in map(os.path.abspath, paths):
            paths.append(wfn)

    frame_id = "ov" + uuid.uuid4().hex[:10]
    return """
<div style="border:1px solid #d0d7de;border-radius:8px;overflow:hidden;background:#fff">
  <iframe id="{fid}" src="{src}" title="Просмотрщик ORCA"
          style="width:100%;height:{height}px;border:0;display:block"
          allow="clipboard-write"></iframe>
</div>
<script>
(function () {{
  var frame = document.getElementById("{fid}");
  var payload = {payload};
  var sent = false;
  // просмотрщик сообщает «готов», и только тогда ему уходят файлы
  window.addEventListener("message", function (e) {{
    if (sent || e.source !== frame.contentWindow) return;
    var d = e.data;
    if (d && d.type === "orca-viewer" && d.event === "ready") {{
      sent = true;
      frame.contentWindow.postMessage(payload, "{origin}");
    }}
  }});
}})();
</script>
""".format(
        fid=frame_id,
        src=escape(_with_query(VIEWER_URL, embed="1"), quote=True),
        height=int(height),
        payload=_payload(paths, compress, qtaim, elf),
        origin=_origin(VIEWER_URL),
    )


def show(*paths, height=600, compress=True, qtaim=False, elf=False):
    """Показать файлы ORCA (.out, .xyz, .allxyz, .cube, .inp, .wfn) под ячейкой."""
    if HTML is None:
        raise RuntimeError("show() работает только внутри Jupyter")
    display(HTML(render_html(*paths, height=height, compress=compress, qtaim=qtaim, elf=elf)))
