"""Просмотр структур и траекторий GROMACS в ноутбуке: iframe с WebMol — просмотрщиком курса.

    from md_tools import show
    show("box.gro")                   # одна структура
    show("md.gro", "md.xtc")          # структура + траектория (порядок любой)
    show("npt.gro", "npt.trr", wrap=False, height=500)
    show("md.gro", "md.xtc", cmd="hide everything; show sticks, resn ETH; show lines, solvent; color marine, resn NA")
    show("md.gro", "md.xtc", b=rmsf_nm, cmd="spectrum b, blue_white_red, name CA")
    show("md.gro", "md.xtc", data={"rmsf": rmsf_nm}, cmd="show sticks, rmsf > 0.1; label name CA, rmsf")
    show("md.gro", "md.xtc", stride=5)  # каждый 5-й кадр
    show("alphaR/ala2.pdb", "beta/ala2.pdb")  # одноимённые файлы — состояния одного объекта, разноимённые — разные объекты
"""
import base64
import json
import math
import os
import re
import struct
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
    "GMX_VIEWER_URL",
    "https://physchem-nsu.ru/scourses/comp-chem-course/extras/gromacs-viewer",
)

STRUCTURES = (".gro", ".pdb")
TRAJECTORIES = (".xtc", ".trr")
MAX_PAYLOAD_MB = 30
MAX_FRAMES = 100


def _xtc_frames(raw):
    frames, pos = [], 0
    while pos + 56 <= len(raw):
        magic, natoms = struct.unpack_from(">ii", raw, pos)
        if magic not in (1995, 2023):
            raise ValueError("испорченный .xtc: нет метки кадра на байте {}".format(pos))
        end = pos + 56
        if natoms <= 9:
            end += 12 * natoms
        else:
            end += 32
            if magic == 2023:
                nbytes, = struct.unpack_from(">q", raw, end)
                end += 8
            else:
                nbytes, = struct.unpack_from(">i", raw, end)
                end += 4
            end += (nbytes + 3) // 4 * 4
        frames.append((pos, end))
        pos = end
    return frames


def _trr_frames(raw):
    frames, pos = [], 0
    while pos + 24 <= len(raw):
        if struct.unpack_from(">i", raw, pos)[0] != 1993:
            raise ValueError("испорченный .trr: нет метки кадра на байте {}".format(pos))
        slen, = struct.unpack_from(">i", raw, pos + 8)
        head = pos + 12 + (slen + 3) // 4 * 4
        sizes = struct.unpack_from(">13i", raw, head)
        natoms = sizes[10]
        double = sizes[2] == 72 or (natoms and 24 * natoms in (sizes[7], sizes[8], sizes[9]))
        end = head + 13 * 4 + (16 if double else 8) + sum(sizes[:10])
        frames.append((pos, end))
        pos = end
    return frames


def _thin(path, raw, stride, max_frames):
    frames = (_xtc_frames if path.lower().endswith(".xtc") else _trr_frames)(raw)
    n = len(frames)
    if stride is None:
        stride = max(1, math.ceil(n / max_frames)) if max_frames else 1
    if not isinstance(stride, int) or stride < 1:
        raise ValueError("stride — целое число ≥ 1")
    if stride == 1:
        return raw
    return b"".join(raw[a:b] for a, b in frames[::stride])


def _hub_viewer():
    prefix = os.environ.get("JUPYTERHUB_SERVICE_PREFIX", "").strip()
    static = os.environ.get("COURSE_GMX_VIEWER_STATIC", "").strip()
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


def _cmd_list(cmd):
    if cmd is None:
        return []
    if isinstance(cmd, str):
        return [cmd] if cmd.strip() else []
    if isinstance(cmd, (list, tuple)) and all(isinstance(c, str) for c in cmd):
        return [c for c in cmd if c.strip()]
    raise TypeError("cmd — строка команд WebMol (синтаксис PyMOL, через «;») или список строк")


RESERVED = set("b q x y z name resn resi chain elem index id segi alt ss type pepseq model count all none "
               "and or not in like of within around expand gap beyond near_to first last byres bymol".split())


def _natoms(path):
    with open(path, encoding="utf-8", errors="replace") as fh:
        if path.lower().endswith(".gro"):
            fh.readline()
            try:
                return int(fh.readline().split()[0])
            except (IndexError, ValueError):
                raise ValueError("{}: во второй строке нет числа атомов".format(path))
        n = 0
        for line in fh:
            if line.startswith(("ATOM  ", "HETATM")):
                n += 1
            elif line.startswith(("ENDMDL", "END")) and n:
                break
        return n


def _numbers(values, n, what, struct):
    try:
        out = [float(x) for x in values]
    except (TypeError, ValueError):
        raise ValueError("{}: нужен список чисел по атомам (список или одномерный массив)".format(what))
    if len(out) != n:
        raise ValueError("{}: длина {}, а атомов в {} — {}".format(what, len(out), os.path.basename(struct), n))
    bad = [i + 1 for i, x in enumerate(out) if not math.isfinite(x)]
    if bad:
        raise ValueError("{}: не число (nan/inf) у атома {}".format(what, bad[0]))
    return out


def _atom_data(paths, data, b):
    if data is None and b is None:
        return None, None
    structs = [p for p in paths if p.lower().endswith(STRUCTURES)]
    if len(structs) != 1:
        raise ValueError("data= и b= — только для одной структуры (.gro или .pdb), передано: {}".format(len(structs)))
    struct = structs[0]
    n = _natoms(struct)
    clean = None
    if data is not None:
        if not isinstance(data, dict):
            raise ValueError("data — словарь {имя: список чисел по атомам}, например data={'rmsf': rmsf}")
        clean = {}
        for name, values in data.items():
            if not isinstance(name, str) or not re.match(r"^[A-Za-z_]\w*$", name):
                raise ValueError("data: имя поля — латиница, цифры и _ (не с цифры): {!r}".format(name))
            if name.lower() in RESERVED:
                raise ValueError("data: имя {!r} занято языком выборок{}".format(
                    name, " — для поля b есть параметр b=" if name.lower() == "b" else ", возьмите другое"))
            clean[name] = _numbers(values, n, "data[{!r}]".format(name), struct)
    return clean, (_numbers(b, n, "b", struct) if b is not None else None)


def _hub_src(hub, rels, wrap, cmds=()):
    prefix, static = hub
    query = ["base=" + quote(prefix, safe="/"), "files=" + ",".join(quote(r, safe="/") for r in rels)]
    if not wrap:
        query.append("wrap=0")
    if cmds:
        query.append("cmd=" + quote("; ".join(cmds), safe=""))
    return prefix + static + "?" + "&".join(query)


def _with_query(url, **params):
    parts = urlsplit(url)
    query = dict(parse_qsl(parts.query))
    query.update(params)
    return urlunsplit(parts._replace(query=urlencode(query)))


def _origin(url):
    parts = urlsplit(url)
    return "{}://{}".format(parts.scheme, parts.netloc)


def _check(paths):
    if not paths:
        raise ValueError("Укажите файл: show('box.gro') или show('md.gro', 'md.xtc')")
    paths = [os.fspath(p) for p in paths]
    missing = [p for p in paths if not os.path.isfile(p)]
    if missing:
        raise FileNotFoundError("Нет файла: " + ", ".join(missing))
    other = [p for p in paths if not p.lower().endswith(STRUCTURES + TRAJECTORIES)]
    if other:
        raise ValueError("Не файл GROMACS: {} (нужны .gro, .pdb, .xtc, .trr)".format(", ".join(other)))
    tops = [p for p in paths if p.lower().endswith(STRUCTURES)]
    trajs = [p for p in paths if p.lower().endswith(TRAJECTORIES)]
    if not tops:
        raise ValueError("Нужен хотя бы один файл структуры (.gro или .pdb)")
    if len(trajs) > 1:
        raise ValueError("Траектория — не больше одной (.xtc или .trr), передано: {}".format(len(trajs)))
    if trajs and len(tops) != 1:
        raise ValueError("С траекторией — ровно одна структура (.gro или .pdb), передано: {}".format(len(tops)))
    return paths


def _read(paths, stride, max_frames):
    contents = {}
    for path in paths:
        with open(path, "rb") as fh:
            raw = fh.read()
        if path.lower().endswith(TRAJECTORIES):
            raw = _thin(path, raw, stride, max_frames)
        contents[path] = raw
    return contents


def _payload(contents, compress, wrap, cmds=(), atom_data=None, b=None):
    total = sum(len(raw) for raw in contents.values())
    if total > MAX_PAYLOAD_MB * 2 ** 20:
        raise ValueError("Файлы занимают {:.0f} МБ — больше {} МБ, ноутбук такое не утащит. "
                         "Уменьшите max_frames или увеличьте stride".format(total / 2 ** 20, MAX_PAYLOAD_MB))
    files = []
    for path, raw in contents.items():
        entry = {"name": os.path.basename(path)}
        if compress or path.lower().endswith(TRAJECTORIES):
            entry["gz"] = base64.b64encode(zlib.compress(raw, 6)).decode("ascii")
        else:
            entry["text"] = raw.decode("utf-8", errors="replace")
        files.append(entry)
    data = {"type": "gromacs-viewer", "files": files}
    options = {}
    if not wrap:
        options["wrap"] = False
    if cmds:
        options["cmd"] = list(cmds)
    if atom_data:
        options["data"] = atom_data
    if b is not None:
        options["b"] = b
    if options:
        data["options"] = options
    return json.dumps(data, ensure_ascii=True).replace("<", "\\u003c")


def render_html(*paths, height=600, compress=True, wrap=True, cmd=None, data=None, b=None,
                stride=None, max_frames=MAX_FRAMES):
    """HTML ячейки с просмотрщиком (то же, что show(), но без вывода)."""
    paths = _check(paths)
    cmds = _cmd_list(cmd)
    data, b = _atom_data(paths, data, b)
    contents = _read(paths, stride, max_frames)
    thinned = any(len(raw) != os.path.getsize(p) for p, raw in contents.items())
    note = ""
    hub = _hub_viewer()
    if hub:
        rels = [_server_path(p) for p in paths]
        if None not in rels and not data and b is None and not thinned:
            return """
<div style="border:1px solid #d0d7de;border-radius:8px;overflow:hidden;background:#fff">
  <iframe src="{src}" title="WebMol — просмотрщик GROMACS" style="width:100%;height:{height}px;border:0;display:block"
          allow="clipboard-write"></iframe>
</div>
""".format(src=escape(_hub_src(hub, rels, wrap, cmds), quote=True), height=int(height))
        outside = [p for p, r in zip(paths, rels) if r is None]
        if outside:
            why = "Файлы вне папки сервера (или скрытые): {} — переданы".format(escape(", ".join(outside)))
        elif data or b is not None:
            why = "Данные по атомам (data=, b=) в адрес окна не помещаются: файлы и данные переданы"
        else:
            why = None
        if why:
            note = '<p style="margin:0 0 6px;color:#9a6700">{} в просмотрщик целиком через ноутбук.</p>'.format(why)

    frame_id = "gv" + uuid.uuid4().hex[:10]
    return note + """
<div style="border:1px solid #d0d7de;border-radius:8px;overflow:hidden;background:#fff">
  <iframe id="{fid}" src="{src}" title="WebMol — просмотрщик GROMACS"
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
    if (d && d.type === "gromacs-viewer" && d.event === "ready") {{
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
        payload=_payload(contents, compress, wrap, cmds, data, b),
        origin=_origin(VIEWER_URL),
    )


def show(*paths, height=600, compress=True, wrap=True, cmd=None, data=None, b=None,
         stride=None, max_frames=MAX_FRAMES):
    """Показать структуру и, по желанию, траекторию GROMACS под ячейкой в WebMol.

    paths    -- .gro/.pdb и не больше одного .xtc/.trr (порядок любой; с траекторией — одна структура);
                одноимённые структуры — состояния одного объекта, разноимённые — разные объекты:
                show("alphaR/ala2.pdb", "beta/ala2.pdb")
    height   -- высота окна, px
    compress -- сжимать текстовые файлы при передаче (.xtc/.trr сжимаются всегда)
    wrap     -- «целые молекулы»: собрать молекулы, разрезанные границей ячейки (переключается и в окне)
    cmd      -- команды WebMol (синтаксис PyMOL) после загрузки: строка (через «;») или список строк, например
                show("md.gro", "md.xtc", cmd="hide everything; show sticks, resn ETH; show lines, solvent; color marine, resn NA")
    data     -- {имя: числа по атомам}: свойства для выборок (rmsf > 0.1), spectrum, label, alter
    b        -- числа по атомам в поле b, например RMSF в нм:
                show("md.gro", "md.xtc", b=rmsf_nm, cmd="spectrum b, blue_white_red, name CA")
    stride   -- показывать каждый stride-й кадр траектории; по умолчанию подбирается так, чтобы кадров было
                не больше max_frames (100); явный stride отменяет max_frames, stride=1 — все кадры
    Длина data[…] и b — число атомов структуры, иначе ValueError. На хабе с data/b и с прореженной траекторией
    всё идёт через ноутбук.
    """
    if HTML is None:
        raise RuntimeError("show() работает только внутри Jupyter")
    display(HTML(render_html(*paths, height=height, compress=compress, wrap=wrap, cmd=cmd, data=data, b=b,
                             stride=stride, max_frames=max_frames)))
