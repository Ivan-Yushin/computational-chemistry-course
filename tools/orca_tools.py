"""Просмотрщик ORCA под ячейкой ноутбука.

    import sys
    sys.path.append("/home/jupyter/project/computational-chemistry-course/tools")
    from orca_tools import show

    show("water.out")
    show("h2o2.out", "h2o2.mo5.cube")
    show("scan.out", height=700)
    show("qtaim_c6h6.out", "qtaim_c6h6.wfn", qtaim=True)
    show("elf_water.out", "elf_water.wfn", elf=True)
    show("esp_h2o.out", esp=True)     # кубы плотности и ESP строит orca_plot
    show("carvone_r/carvone_r.out", ECD=True)
    show("meoh_nmr", nmr_ref="tms_nmr")   # δ = σ_ref − σ, σ_ref — среднее по атомам элемента в ТМС
"""

import base64
import json
import os
import shutil
import subprocess
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

ESP_GRID = 81  # точек сетки orca_plot по каждой оси

# ответы на меню orca_plot -i (ORCA 6.1.1): тип графика 2 — плотность (y — по умолчанию <base>.scfp),
# 4 — сетка, 5/7 — формат cube, 11 — построить <base>.eldens.cube; затем тип 43 — ESP
# от плотности <base>.scfp, 11 — построить <base>.scfp.esp.cube, 12 — выход
_ESP_ANSWERS = ["1", "2", "y", "4", "{n} {n} {n}", "5", "7", "11", "1", "43", "{base}.scfp", "11", "12"]


def _with_query(url, **params):
    parts = urlsplit(url)
    query = dict(parse_qsl(parts.query))
    query.update(params)
    return urlunsplit(parts._replace(query=urlencode(query)))


def _origin(url):
    parts = urlsplit(url)
    return "{}://{}".format(parts.scheme, parts.netloc)


def _find_orca_plot(orca_plot=None):
    """orca_plot: аргумент, $ORCA_PLOT, PATH, рядом с orca из PATH или из $ORCA, /opt/orca."""
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
    """Кубы плотности и ESP к <base>.out; если их нет — строим из <base>.gbw."""
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
    options = {k: on if isinstance(on, str) else True for k, on in flags.items() if on}  # qtaim, elf, esp, ecd; nmrRef — имя файла
    if options:
        data["options"] = options
    text = json.dumps(data, ensure_ascii=True)
    return text.replace("</", "<\\/")  # «</» внутри <script> закрыл бы тег


def _calc_out(p):
    """Папка расчёта вместо файла: show("acrolein_tddft") → acrolein_tddft/acrolein_tddft.out."""
    out = os.path.join(p, os.path.basename(os.path.normpath(p)) + ".out")
    return out if os.path.isdir(p) and os.path.isfile(out) else p


def render_html(*paths, height=600, compress=True, qtaim=False, elf=False, esp=False, ECD=False, ecd=False, orca_plot=None,
                nmr_ref=None):
    """HTML вывода ячейки: iframe просмотрщика и скрипт передачи файлов."""
    if not paths:
        raise ValueError("Укажите хотя бы один файл: show('water.out')")
    paths = [_calc_out(p) for p in paths]
    missing = [p for p in paths if not os.path.isfile(p)]
    if missing:
        raise FileNotFoundError("Нет файла: " + ", ".join(missing))
    paths = [os.fspath(p) for p in paths]
    # nmr_ref: расчёт ЯМР эталона (ТМС) — прикладываем его, просмотрщик берёт σ_ref по элементам оттуда
    ref_name = ref = None
    if nmr_ref is not None:
        ref = os.fspath(_calc_out(nmr_ref))
        if not os.path.isfile(ref):
            raise FileNotFoundError("Нет файла: " + ref)
        if os.path.abspath(ref) not in map(os.path.abspath, paths):
            paths.append(ref)
        ref_name = os.path.basename(ref)
    # к NEB-расчёту прикладываем путь <имя>_MEP_trj.xyz, если он лежит рядом
    for p in list(paths):
        trj = p[:-4] + "_MEP_trj.xyz"
        if p.endswith(".out") and os.path.isfile(trj) and os.path.abspath(trj) not in map(os.path.abspath, paths):
            paths.append(trj)
    # qtaim=True / elf=True: к .out прикладываем <имя>.wfn (orca_2aim) и сразу открываем вкладку QTAIM / ELF
    for p in list(paths):
        wfn = p[:-4] + ".wfn"
        if (qtaim or elf) and p.endswith(".out") and os.path.isfile(wfn) and os.path.abspath(wfn) not in map(os.path.abspath, paths):
            paths.append(wfn)
    # esp=True: к .out прикладываем <имя>.eldens.cube и <имя>.scfp.esp.cube (при нужде строим) и открываем ESP;
    # эталону ЯМР (nmr_ref) кубы не нужны — у него может не быть .gbw
    if esp:
        for p in list(paths):
            if p.endswith(".out") and not (ref and os.path.abspath(p) == os.path.abspath(ref)):
                for c in _esp_cubes(p, orca_plot):
                    if os.path.abspath(c) not in map(os.path.abspath, paths):
                        paths.append(c)

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
        payload=_payload(paths, compress, qtaim=qtaim, elf=elf, esp=esp, ecd=ECD or ecd, nmrRef=ref_name),
        origin=_origin(VIEWER_URL),
    )


def show(*paths, height=600, compress=True, qtaim=False, elf=False, esp=False, ECD=False, ecd=False, orca_plot=None,
         nmr_ref=None):
    """Показать файлы ORCA (.out, .xyz, .allxyz, .cube, .inp, .wfn) под ячейкой.

    ECD=True (или ecd=True) — на вкладке «Спектр» ещё и спектр ЭКД.
    nmr_ref="tms_nmr/tms_nmr.out" — эталон ЯМР: на вкладке «ЯМР» сразу δ относительно него.
    """
    if HTML is None:
        raise RuntimeError("show() работает только внутри Jupyter")
    display(HTML(render_html(*paths, height=height, compress=compress, qtaim=qtaim, elf=elf, esp=esp, ECD=ECD, ecd=ecd,
                             orca_plot=orca_plot, nmr_ref=nmr_ref)))
