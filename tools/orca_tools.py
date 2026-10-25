import base64
import json
import os
import shutil
import subprocess
import uuid
import zlib
from html import escape
from urllib.parse import parse_qsl, quote, urlencode, urlsplit, urlunsplit

try:
    from IPython.display import HTML, display
except ImportError:
    HTML = display = None

__all__ = ["show", "render_html", "VIEWER_URL"]

VIEWER_URL = os.environ.get(
    "ORCA_VIEWER_URL",
    "https://physchem-nsu.ru/scourses/comp-chem-course/extras/orca-viewer",
)

ESP_GRID = 81

_ESP_ANSWERS = ["1", "2", "y", "4", "{n} {n} {n}", "5", "7", "11", "1", "43", "{base}.scfp", "11", "12"]


def _hub_viewer():
    prefix = os.environ.get("JUPYTERHUB_SERVICE_PREFIX", "").strip()
    static = os.environ.get("COURSE_VIEWER_STATIC", "").strip()
    if not (prefix.startswith("/") and static.startswith("/")):
        return None
    return prefix.rstrip("/"), static


def _server_path(path):
    root = os.path.realpath(os.environ.get("JUPYTER_SERVER_ROOT") or os.path.expanduser("~"))
    rel = os.path.relpath(os.path.realpath(path), root)
    parts = rel.split(os.sep)
    if os.path.isabs(rel) or any(not p or p.startswith(".") for p in parts):
        return None
    return "/".join(parts)


def _hub_src(hub, rels, **flags):
    prefix, static = hub
    query = ["base=" + quote(prefix, safe="/"), "files=" + ",".join(quote(r, safe="/") for r in rels)]
    for key, on in flags.items():
        if on:
            query.append(key + "=" + (quote(on, safe="") if isinstance(on, str) else "1"))
    return prefix + static + "?" + "&".join(query)


def _with_query(url, **params):
    parts = urlsplit(url)
    query = dict(parse_qsl(parts.query))
    query.update(params)
    return urlunsplit(parts._replace(query=urlencode(query)))


def _origin(url):
    parts = urlsplit(url)
    return "{}://{}".format(parts.scheme, parts.netloc)


def _find_orca_plot(orca_plot=None):
    cands = [orca_plot, os.environ.get("ORCA_PLOT"), shutil.which("orca_plot")]
    for orca in (shutil.which("orca"), os.environ.get("ORCA")):
        if orca:
            cands.append(os.path.join(os.path.dirname(os.path.abspath(orca)), "orca_plot"))
    cands.append("/opt/orca/orca_plot")
    for c in cands:
        if c and os.path.isfile(c) and os.access(c, os.X_OK):
            return c
    raise FileNotFoundError("Не найден orca_plot: укажите show(..., orca_plot='/opt/orca/orca_plot') "
                            "или переменную окружения ORCA_PLOT")


def _esp_cubes(out_path, orca_plot=None):
    base = out_path[:-4]
    dens, pot = base + ".eldens.cube", base + ".scfp.esp.cube"
    if os.path.isfile(dens) and os.path.isfile(pot):
        return [dens, pot]
    if not os.path.isfile(base + ".gbw"):
        raise FileNotFoundError("Нет файла: " + base + ".gbw (нужен для кубов ESP)")
    name = os.path.basename(base)
    answers = "\n".join(a.format(n=ESP_GRID, base=name) for a in _ESP_ANSWERS) + "\n"
    subprocess.run([_find_orca_plot(orca_plot), name + ".gbw", "-i"], cwd=os.path.dirname(os.path.abspath(out_path)),
                   input=answers, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    for f in (dens, pot):
        if not os.path.isfile(f):
            raise RuntimeError("orca_plot не построил " + os.path.basename(f))
    return [dens, pot]


def _payload(paths, compress, **flags):
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
    options = {k: on if isinstance(on, str) else True for k, on in flags.items() if on}
    if options:
        data["options"] = options
    text = json.dumps(data, ensure_ascii=True)
    return text.replace("<", "\\u003c")


def _calc_out(p):
    out = os.path.join(p, os.path.basename(os.path.normpath(p)) + ".out")
    return out if os.path.isdir(p) and os.path.isfile(out) else p


def render_html(*paths, height=600, compress=True, qtaim=False, elf=False, esp=False, ECD=False, ecd=False, orca_plot=None,
                nmr_ref=None):
    if not paths:
        raise ValueError("Укажите хотя бы один файл: show('water.out')")
    paths = [_calc_out(p) for p in paths]
    missing = [p for p in paths if not os.path.isfile(p)]
    if missing:
        raise FileNotFoundError("Нет файла: " + ", ".join(missing))
    paths = [os.fspath(p) for p in paths]
    ref_name = ref = None
    if nmr_ref is not None:
        ref = os.fspath(_calc_out(nmr_ref))
        if not os.path.isfile(ref):
            raise FileNotFoundError("Нет файла: " + ref)
        if os.path.abspath(ref) not in map(os.path.abspath, paths):
            paths.append(ref)
        ref_name = os.path.basename(ref)
    for p in list(paths):
        trj = p[:-4] + "_MEP_trj.xyz"
        if p.endswith(".out") and os.path.isfile(trj) and os.path.abspath(trj) not in map(os.path.abspath, paths):
            paths.append(trj)
    for p in list(paths):
        wfn = p[:-4] + ".wfn"
        if (qtaim or elf) and p.endswith(".out") and os.path.isfile(wfn) and os.path.abspath(wfn) not in map(os.path.abspath, paths):
            paths.append(wfn)
    if esp:
        for p in list(paths):
            if p.endswith(".out") and not (ref and os.path.abspath(p) == os.path.abspath(ref)):
                for c in _esp_cubes(p, orca_plot):
                    if os.path.abspath(c) not in map(os.path.abspath, paths):
                        paths.append(c)

    flags = dict(qtaim=qtaim, elf=elf, esp=esp, ecd=ECD or ecd)
    note = ""
    hub = _hub_viewer()
    if hub:
        rels = [_server_path(p) for p in paths]
        if None not in rels:
            return """
<div style="border:1px solid #d0d7de;border-radius:8px;overflow:hidden;background:#fff">
  <iframe src="{src}" title="Просмотрщик ORCA" style="width:100%;height:{height}px;border:0;display:block"
          allow="clipboard-write"></iframe>
</div>
""".format(src=escape(_hub_src(hub, rels, nmr_ref=ref_name, **flags), quote=True), height=int(height))
        outside = [p for p, r in zip(paths, rels) if r is None]
        note = ('<p style="margin:0 0 6px;color:#9a6700">Файлы вне папки сервера (или скрытые): {} — '
                'переданы в просмотрщик целиком через ноутбук.</p>').format(escape(", ".join(outside)))

    frame_id = "ov" + uuid.uuid4().hex[:10]
    return note + """
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
        payload=_payload(paths, compress, qtaim=qtaim, elf=elf, esp=esp, ecd=ECD or ecd, nmrRef=ref_name),
        origin=_origin(VIEWER_URL),
    )


def show(*paths, height=600, compress=True, qtaim=False, elf=False, esp=False, ECD=False, ecd=False, orca_plot=None,
         nmr_ref=None):
    if HTML is None:
        raise RuntimeError("show() работает только внутри Jupyter")
    display(HTML(render_html(*paths, height=height, compress=compress, qtaim=qtaim, elf=elf, esp=esp, ECD=ECD, ecd=ecd,
                             orca_plot=orca_plot, nmr_ref=nmr_ref)))
