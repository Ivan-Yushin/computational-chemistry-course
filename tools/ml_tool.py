import re

import numpy as np
import plotly.graph_objects as go

__all__ = ["show", "figure"]

RED, BLUE, GREY = "#d62728", "#1f77b4", "#999999"
SPAN = 4.2       # высота самого высокого столбца по оси y: 8 нейронов через 0.6
CIRCLE = 36      # диаметр кружка, px
PAD = 0.3        # поле слева и справа от крайних столбцов, единицы оси x
CHAR_PX = 7      # средняя ширина буквы подписи, px
NAME_CHARS = 24  # длиннее — обрезаем
HOVER_PX = 15    # шаг точек вдоль линии при verbose, px: подсказка видна по всей длине линии
VALUE_FONT = 11  # подписи значений нейронов при input, px
VALUE_BG = "white"
MIN_UNDER = 28   # значения под кружками во всех столбцах, если кружок остаётся не меньше, px


def _untrained(model, hidden, n_in, n_out, activations):
    """Необученная модель: сеть нужных размеров с нулевыми весами; число признаков берётся из features."""
    if n_in is None:
        raise ValueError("Модель {} ещё не обучена, число признаков неизвестно: передайте features=[…] "
                         "или сначала вызовите model.fit(X, y)".format(type(model).__name__))
    pairs = list(zip([n_in] + hidden, hidden + [n_out]))
    return {"weights": [np.zeros(p) for p in pairs], "biases": [np.zeros(p[1]) for p in pairs],
            "activations": activations, "features": None}


def _pipeline_names(pipe):
    names = getattr(pipe, "feature_names_in_", None)
    if names is not None and len(pipe.steps) > 1:
        try:
            names = pipe[:-1].get_feature_names_out()
        except Exception:
            pass
    return names


def _from_sklearn(model, n_in=None, n_out=1):
    """Модель scikit-learn -> веса (n_in, n_out), смещения, активации по слоям и имена признаков; None — не она.

    У необученной модели весов ещё нет: рисуется сеть с нулевыми весами, n_in и n_out — число признаков и выходов.
    """
    names = None
    if hasattr(model, "steps"):
        names = _pipeline_names(model)
        model = model.steps[-1][1]
    if names is None:
        names = getattr(model, "feature_names_in_", None)
    if hasattr(model, "hidden_layer_sizes"):
        if not hasattr(model, "coefs_"):
            hidden = [int(n) for n in np.ravel(model.hidden_layer_sizes)]
            return _untrained(model, hidden, n_in, n_out, [model.activation] * len(hidden) + [None])
        weights = [np.asarray(w, dtype=float) for w in model.coefs_]
        biases = [np.asarray(b, dtype=float) for b in model.intercepts_]
        activations = [model.activation] * (len(weights) - 1) + [model.out_activation_]
    elif hasattr(model, "coef_") or hasattr(model, "fit_intercept"):
        if not hasattr(model, "coef_"):
            return _untrained(model, [], n_in, n_out, [None])
        coef = np.asarray(model.coef_, dtype=float)
        weights = [coef.reshape(1, -1).T if coef.ndim == 1 else coef.T]
        n_out = weights[0].shape[1]
        intercept = np.ravel(np.asarray(getattr(model, "intercept_", 0.0), dtype=float))
        biases = [np.broadcast_to(intercept, (n_out,)).copy()]
        activations = [None if hasattr(model, "classes_") else "identity"]
    else:
        return None
    return {"weights": weights, "biases": biases, "activations": activations, "features": names}


TORCH_ACTIVATIONS = {"ReLU": "relu", "Sigmoid": "logistic", "Tanh": "tanh", "Softmax": "softmax",
                     "Identity": "identity"}
TORCH_UNSUPPORTED = "Слой {} нарисовать нельзя: из слоёв с весами рисуются только Linear и Conv1d"
GRAPH_HINT = (". Если это модель для графов, передайте ей граф: show(model, input=(X, A, E)) или "
              "show(model, input=(X, A)), где X — признаки вершин, A — матрица смежности, E — признаки рёбер")


def _array(tensor):
    return tensor.detach().cpu().double().numpy()


def _torch_layer(layer):
    """Слой Linear или Conv1d -> вес (входы, выходы) и смещение; Conv1d — одно окно: вход c·K + k, выход — фильтр."""
    name = type(layer).__name__
    if name == "Linear":
        W = _array(layer.weight).T
    elif name == "Conv1d":
        w = _array(layer.weight)  # (фильтры, каналы в группе, окно)
        n_f, n_c, k = w.shape
        W = np.zeros((layer.in_channels, k, n_f))
        for j in range(n_f):
            c = j // (n_f // layer.groups) * n_c
            W[c:c + n_c, :, j] = w[j]
        W = W.reshape(-1, n_f)
    else:
        raise TypeError(TORCH_UNSUPPORTED.format(name))
    return W, np.zeros(W.shape[1]) if layer.bias is None else _array(layer.bias)


def _from_torch(model):
    """Модель PyTorch -> то же, что у _from_sklearn; None — не она.

    Слои с весами идут цепочкой в том порядке, в каком их отдаёт model.modules(); слои без весов
    (активации, пулинг, Flatten, Dropout) не рисуются.
    """
    if not all(callable(getattr(model, a, None)) for a in ("modules", "parameters", "state_dict")):
        return None
    weights, biases, activations, layers = [], [], [], []
    for layer in model.modules():
        name = type(layer).__name__
        own = any(True for _ in layer.parameters(recurse=False))
        if any(True for _ in layer.children()):
            if own:
                raise TypeError(TORCH_UNSUPPORTED.format(name))
            continue
        if not own:
            if layers and activations[-1] is None and (
                    name in TORCH_ACTIVATIONS or type(layer).__module__.endswith(".activation")):
                activations[-1] = TORCH_ACTIVATIONS.get(name, name.lower())
            continue
        W, b = _torch_layer(layer)
        n_in = layer.in_features if name == "Linear" else layer.in_channels
        if layers:
            prev, n = type(layers[-1]).__name__, weights[-1].shape[1]
            if n != n_in:
                raise TypeError("Слои не складываются в цепочку, которую можно нарисовать: у слоя {} выходов {}, "
                                "а у следующего за ним слоя {} входов {}".format(prev, n, name, n_in))
            if n != W.shape[0]:
                raise TypeError("Слои не складываются в цепочку, которую можно нарисовать: Conv1d с kernel_size > 1 "
                                "рисуется только первым слоем (у слоя {} выходов {}, а окно следующего Conv1d — "
                                "{} входов)".format(prev, n, W.shape[0]))
        weights.append(W)
        biases.append(b)
        activations.append(None)
        layers.append(layer)
    if not layers:
        raise TypeError("В модели {} нет слоёв Linear или Conv1d — рисовать нечего".format(type(model).__name__))
    first = layers[0]
    return {"weights": weights, "biases": biases, "activations": [a or "identity" for a in activations],
            "features": None, "window": type(first).__name__ == "Conv1d" and first.in_channels == 1,
            "layers": layers}


def _count(names):
    if names is None or isinstance(names, (int, np.integer)):
        return names
    return 1 if isinstance(names, str) else len(names)


def _network(model, features=None, output=None, n_x=None):
    """n_x — число значений в input: по нему необученной модели без features задаётся число признаков."""
    n_in = _count(features)
    net = _from_sklearn(model, n_x if n_in is None else n_in, _count(output) or 1)
    if net is None:
        try:
            net = _from_torch(model)
        except TypeError as e:  # свой модуль со своими весами — может быть, слой модели для графов
            if str(e) != TORCH_UNSUPPORTED.format(type(model).__name__) or type(model).__module__.startswith("torch."):
                raise
            raise TypeError(str(e) + GRAPH_HINT) from None
    if net is None:
        raise TypeError(
            "Рисуются многослойный перцептрон (MLPRegressor, MLPClassifier), линейные модели (LinearRegression, Ridge, "
            "Lasso, LogisticRegression, …), Pipeline, который ими заканчивается, и модели PyTorch (nn.Sequential "
            "или модуль из слоёв Linear и Conv1d); получено: {}".format(type(model).__name__))
    return net


def _names(names, n, arg, what, strict):
    if names is None:
        return None
    if isinstance(names, (int, np.integer)):
        if names != n:
            raise ValueError("{} = {}, а {} у модели {}".format(arg, names, what, n))
        return None
    names = [names] if isinstance(names, str) else [str(s) for s in names]
    if len(names) != n:
        if strict:
            raise ValueError("{}: имён {}, а {} у модели {}".format(arg, len(names), what, n))
        return None
    return [s if len(s) <= NAME_CHARS else s[:NAME_CHARS - 1] + "…" for s in names]


def _text_px(text):
    return CHAR_PX * max(len(line) for line in re.sub("<(?!br>)[^>]+>", "", text).split("<br>"))


def _label_em(label):
    small = "".join(re.findall(r"<su[bp]>(.*?)</su[bp]>", label))
    return 0.6 * len(re.sub(r"<su[bp]>.*?</su[bp]>", "", label)) + 0.4 * len(small) + 0.2 * small.count("+")


def _shown(n, max_neurons):
    """Какие нейроны рисовать и номер места «⋮» среди мест столбца (None — рисуются все)."""
    if n <= max_neurons:
        return list(range(n)), None
    head = (max_neurons + 1) // 2
    return list(range(head)) + list(range(n - (max_neurons - head), n)), head


def _x_layout(n_cols, plot_w, left_px, right_px):
    """Поля слева и справа (единицы оси x) и px на единицу; поле не меньше PAD и вмещает подписи."""
    fixed = [None, None]
    for _ in range(3):
        free = sum(f is None for f in fixed)
        u = max((plot_w - sum(f or 0 for f in fixed)) / (n_cols - 1 + PAD * free), 20.0)
        for i, need in enumerate((left_px, right_px)):
            if fixed[i] is None and need > PAD * u:
                fixed[i] = need
    return [PAD if f is None else f / u for f in fixed], u


def _num(v):
    return "{:.4g}".format(float(v) + 0.0)  # + 0.0: -0.0 -> 0.0


def _par(v):
    """Число внутри формулы: отрицательное — в скобках."""
    s = _num(v)
    return "({})".format(s) if s.startswith("-") else s


# ---------- input: значения нейронов для одного объекта ----------

def _softmax(z):
    e = np.exp(z - z.max())
    return e / e.sum()


ACTIVATIONS = {"identity": lambda z: z, "logistic": lambda z: 0.5 * (1 + np.tanh(z / 2)), "tanh": np.tanh,
               "relu": lambda z: np.maximum(z, 0.0), "softmax": _softmax}


def _row(x):
    """input — один объект -> строка (1, n); строка таблицы pandas остаётся таблицей: в ней имена признаков."""
    if hasattr(x, "detach"):  # тензор PyTorch
        x = _array(x)
    if hasattr(x, "to_frame"):  # pandas Series, например df.iloc[0]
        x = x.to_frame().T
    if not hasattr(x, "columns"):
        x = np.asarray(x, dtype=float)
        x = x.reshape(1, -1) if x.ndim < 2 else x
    if x.ndim != 2 or x.shape[0] != 1:
        raise ValueError("input — один объект: значения его признаков (список, массив, тензор или строка таблицы), "
                         "а передано " + ("строк {}".format(x.shape[0]) if x.ndim == 2
                                          else "массив формы {}".format(tuple(x.shape))))
    return x


def _check_count(n_x, n):
    if n_x != n:
        raise ValueError("input: значений {}, а признаков у модели {}".format(n_x, n))


def _sklearn_values(model, net, row):
    """Прямой проход в NumPy по весам и смещениям net: [(z, a)] по столбцам, у входов z = None.

    input у Pipeline — то, что передают в pipe.predict: сначала его преобразуют шаги перед моделью.
    """
    est = model.steps[-1][1] if hasattr(model, "steps") else model
    a = np.asarray(row, dtype=float).ravel()
    if hasattr(model, "steps") and len(model.steps) > 1:
        n_raw = getattr(model, "n_features_in_", None)
        if n_raw is not None:
            _check_count(a.size, n_raw)
        try:
            a = np.asarray(model[:-1].transform(row), dtype=float).ravel()
        except Exception as e:
            if type(e).__name__ != "NotFittedError":
                raise
            raise ValueError("Pipeline ещё не обучен: шаги перед моделью не могут преобразовать input — "
                             "сначала вызовите model.fit(X, y)") from e
    _check_count(a.size, net["weights"][0].shape[0])
    proba = hasattr(est, "predict_proba")
    values = [(None, a)]
    for W, b, act in zip(net["weights"], net["biases"], net["activations"]):
        z = a @ W + b
        if act is None:  # выход линейного классификатора или необученной модели
            act = ("logistic" if z.size == 1 else "softmax") if proba else "identity"
        a = ACTIVATIONS[act](z)
        values.append((z, a))
    if proba and hasattr(est, "coef_") and hasattr(est, "classes_"):  # обученный линейный классификатор
        p = np.asarray(model.predict_proba(row), dtype=float).ravel()
        values[-1] = (values[-1][0], p[1:] if p.size == 2 and a.size == 1 else p)
    return values


def _flat(t, n, what):
    """Тензор из n значений -> массив NumPy; иначе прямой проход не сходится с рисунком."""
    if not hasattr(t, "detach"):
        raise ValueError("Значения не сходятся с рисунком: {} — не тензор".format(what))
    if t.numel() != n:
        raise ValueError("Значения не сходятся с рисунком: {} — значений {}, а нейронов в столбце {}".format(
            what, t.numel(), n))
    return _array(t).ravel()


def _torch_values(model, net, row, sizes):
    """Прямой проход самой модели PyTorch: z столбца — выход его слоя с весами, значение скрытого столбца —
    вход следующего слоя с весами, последнего — выход модели. Так верно для любой активации."""
    layers = net["layers"]
    if any(type(layer).__name__ == "Conv1d" for layer in layers):
        raise ValueError("input для модели с Conv1d не показывается: на рисунке одно окно свёртки, "
                         "а не весь вход модели")
    _check_count(row.size, sizes[0])
    seen = [[] for _ in layers]
    modes = [(m, m.training) for m in model.modules()]
    hooks = []
    try:
        for k, layer in enumerate(layers):
            hooks.append(layer.register_forward_hook(
                lambda m, inp, out, k=k: seen[k].append((inp[0] if inp else None, out))))
        model.eval()
        out = model(next(model.parameters()).new_tensor(np.asarray(row, dtype=float).reshape(1, -1)))
    finally:
        for h in hooks:
            h.remove()
        for m, mode in modes:
            m.training = mode
    for layer, calls in zip(layers, seen):
        if len(calls) != 1:
            raise ValueError("Слой {} при прямом проходе {}, а на рисунке он один — значения показать нельзя".format(
                type(layer).__name__, "не вызывается" if not calls else "вызывается {} раз(а)".format(len(calls))))
    name = [type(layer).__name__ for layer in layers]
    values = [(None, _flat(seen[0][0][0], sizes[0], "вход слоя " + name[0]))]
    for k in range(len(layers)):
        z = _flat(seen[k][0][1], sizes[k + 1], "выход слоя " + name[k])
        a = (_flat(seen[k + 1][0][0], sizes[k + 1], "вход слоя " + name[k + 1]) if k + 1 < len(layers)
             else _flat(out, sizes[-1], "выход модели"))
        values.append((z, a))
    return values


def _formula(label, inputs, w, b, z, a, linear):
    """Счёт нейрона для подсказки: z — взвешенная сумма, φ — функция активации (linear — слой без неё)."""
    if len(inputs) > 8:
        s = "Σ вход · вес + смещение"
    else:
        s = "".join(("" if i == 0 else "<br>+ " if i % 4 == 0 else " + ") + "{} · {}".format(_par(v), _par(wi))
                    for i, (v, wi) in enumerate(zip(inputs, w))) + " + " + _par(b)
    if linear:
        return "<br>{} = {} = {}".format(label, s, _num(z))
    return "<br>z = {} = {}<br>{} = φ(z) = {}".format(s, _num(z), label, _num(a))


def _values(model, net, row, sizes):
    values = _torch_values(model, net, row, sizes) if "layers" in net else _sklearn_values(model, net, row)
    return [a for _, a in values], [z for z, _ in values]


# ---------- граф: input=(X, A) или (X, A, E) ----------

EDGE = "#555555"   # цвет рёбер графа
EDGE_PX = 2.0      # толщина ребра, если у рёбер нет одного числа-признака, px
EDGE_FONT = 10     # признак ребра у середины линии, px
MAX_EDGE = 120     # самое длинное ребро на рисунке — не длиннее, px


def _is_graph(x):
    """input — граф: кортеж или список (X, A) или (X, A, E) из тензоров или массивов NumPy (а не чисел)."""
    return isinstance(x, (list, tuple)) and len(x) in (2, 3) and all(len(getattr(t, "shape", ())) > 0 for t in x)


def _graph_arrays(parts):
    """(X, A[, E]) -> X (n, F), A (n, n), E (n, n, K) или None — массивы NumPy; неверная форма — ValueError."""
    X, A, *E = [_array(t) if hasattr(t, "detach") else np.asarray(t, dtype=float) for t in parts]
    arg = "input=(X, A, E)" if E else "input=(X, A)"
    if A.ndim != 2 or A.shape[0] != A.shape[1] or not len(A):
        raise ValueError("{}: A — матрица смежности n × n (n — число вершин), а у A форма {}".format(arg, A.shape))
    n = len(A)
    if X.ndim not in (1, 2) or len(X) != n:
        raise ValueError("{}: X — признаки вершин, по строке на вершину ({} × F или {} чисел: вершин в A {}), "
                         "а у X форма {}".format(arg, n, n, n, X.shape))
    E = E[0] if E else None
    if E is not None and (E.ndim not in (2, 3) or E.shape[:2] != (n, n)):
        raise ValueError("{}: E — признаки рёбер {} × {} × K (или {} × {}: вершин в A {}), а у E форма {}".format(
            arg, n, n, n, n, n, E.shape))
    return X[:, None] if X.ndim == 1 else X, A, E[..., None] if E is not None and E.ndim == 2 else E


def _graph_values(model, parts, n):
    """Прямой проход самой модели в режиме eval: model(X, A, E) -> значения вершин после слоя (n, units)."""
    call = "model(X, A, E)" if len(parts) == 3 else "model(X, A)"
    if not (callable(model) and all(callable(getattr(model, a, None)) for a in ("modules", "parameters"))):
        raise TypeError("input=(X, A, …) — граф: так показывается только модель PyTorch, которая вызывается как "
                        "{}; получено: {}".format(call, type(model).__name__))
    ref = next(model.parameters(), None)  # массив NumPy -> тензор того же типа, что веса модели
    args = [t if ref is None or hasattr(t, "detach") else ref.new_tensor(np.asarray(t, dtype=float)) for t in parts]
    modes = [(m, m.training) for m in model.modules()]
    try:
        model.eval()
        out = model(*args)
    except Exception as e:
        raise ValueError("Модель на графе вызывается как {} (X — признаки вершин, A — матрица смежности{}), "
                         "а этот вызов не удался: {}: {}".format(call, ", E — признаки рёбер" if len(parts) == 3
                                                                 else "", type(e).__name__, e)) from e
    finally:
        for m, mode in modes:
            m.training = mode
    shape = tuple(getattr(out, "shape", ()))
    if len(shape) not in (1, 2) or shape[0] != n:
        raise ValueError("Модель вызывается как {} и должна вернуть по строке значений на вершину — {} строк "
                         "(n × units или n чисел), а вернула {}".format(
                             call, n, "форму {}".format(shape) if shape else type(out).__name__))
    return (_array(out) if hasattr(out, "detach") else np.asarray(out, dtype=float)).reshape(n, -1)


def _messages(model, X, A, E, out):
    """Счёт слоя для подсказок: m_ij = φ(W·[x_j, e_ij]) — сообщение от вершины j вершине i, h_i = φ(Σ_j A_ij m_ij + b).

    Только у модели с атрибутами w_msg (слой с весом (1, F + K)) и b (одно число) и при F + K ≤ 4; φ сообщения —
    ReLU или никакой, φ вершины — identity, ReLU, σ или tanh: та, при которой счёт сходится с самой моделью.
    Если у модели есть w_self (слой с весом (1, F)), к сумме сообщений добавляется вклад самой вершины w_self·x_i.
    Не сходится — None, подсказки без счёта.
    """
    lin, b = getattr(model, "w_msg", None), getattr(model, "b", None)
    W, c = getattr(lin, "weight", None), getattr(lin, "bias", None)
    K = 0 if E is None else E.shape[2]
    if not (hasattr(W, "detach") and hasattr(b, "detach")) or out.shape[1] != 1 or X.shape[1] + K > 4:
        return None
    W, b, c = _array(W), _array(b), None if c is None else _array(c)
    if W.shape != (1, X.shape[1] + K) or b.shape != (1,) or c is not None and c.shape != (1,):
        return None
    own, Ws = None, getattr(getattr(model, "w_self", None), "weight", None)
    if hasattr(Ws, "detach"):  # вклад самой вершины: w_self·x_i (+ смещение слоя w_self)
        Ws, cs = _array(Ws), getattr(model.w_self, "bias", None)
        cs = None if cs is None else _array(cs)
        if Ws.shape != (1, X.shape[1]) or cs is not None and cs.shape != (1,):
            return None
        own = dict(w=Ws[0], c=cs, v=X @ Ws[0] + (0.0 if cs is None else cs[0]))
    n = len(X)
    V = np.concatenate([np.broadcast_to(X, (n,) + X.shape)] + ([E] if K else []), axis=2)  # [x_j, e_ij] у пары i, j
    Z = V @ W[0] + (0.0 if c is None else c[0])
    for relu in (True, False):
        M = np.maximum(Z, 0.0) if relu else Z
        S = (A * M).sum(1)
        T = S + b[0] + (0.0 if own is None else own["v"])  # до функции активации вершины
        if any(np.allclose(ACTIVATIONS[act](T), out[:, 0], rtol=1e-4, atol=1e-6)
               for act in ("identity", "relu", "logistic", "tanh")):
            return dict(V=V, Z=Z, M=M, S=S, W=W[0], b=b[0], c=c, own=own, T=T)
    return None


def _nums(v):
    return ", ".join(_num(x) for x in np.ravel(v))


def _vertex_tip(k, name, X, A, out, calc):
    tip = "вершина {}<br>признак: {}<br>после слоя: {}".format(name(k), _nums(X[k]), _nums(out[k]))
    if calc is None:
        return tip
    J = np.flatnonzero(A[k])
    M, S, own = calc["M"][k], calc["S"][k], calc["own"]
    if own is not None:
        s = " + ".join("{} · {}".format(_par(v), _par(w)) for v, w in zip(X[k], own["w"]))
        s += "" if own["c"] is None else " + " + _par(own["c"][0])
        tip += "<br>от самой вершины: {} = {}".format(s, _num(own["v"][k]))
    if len(J) <= 8:
        for j in J:
            s = " + ".join("{} · {}".format(_par(v), _par(w)) for v, w in zip(calc["V"][k, j], calc["W"]))
            s += "" if calc["c"] is None else " + " + _par(calc["c"][0])
            tip += "<br>от вершины {}: {} = {}".format(name(j), s if M[j] == calc["Z"][k, j] else "ReLU({})".format(s),
                                                       _num(M[j]))
    terms = [_par(M[j]) if A[k, j] == 1 else "{} · {}".format(_par(A[k, j]), _par(M[j])) for j in J]
    full = 1 < len(J) <= 8 or len(J) == 1 and A[k, J[0]] != 1
    tip += "<br>сумма сообщений: {}<br>смещение: {}".format(
        " + ".join(terms) + " = " + _num(S) if full else _num(S), _num(calc["b"]))
    if own is not None:
        tip += "<br>всего: {} + {} + {} = {}".format(_par(own["v"][k]), _par(S), _par(calc["b"]), _num(calc["T"][k]))
    if not np.isclose(out[k, 0], calc["T"][k], rtol=1e-4, atol=1e-6):
        tip += "<br>после функции активации: " + _num(out[k, 0])
    return tip


def _edge_tip(i, j, name, A, E, calc):
    tip = "ребро {} – {}".format(name(i), name(j))
    if E is not None:
        tip += "<br>признак ребра: " + _nums(E[i, j])
    for to, frm in ((i, j), (j, i)):
        if calc is not None and A[to, frm]:
            tip += "<br>сообщение {} → {}: {}".format(name(frm), name(to), _num(calc["M"][to, frm]))
    return tip


def _hops(A):
    """Число рёбер на кратчайшем пути между вершинами; между несвязанными — на 1 больше самого длинного пути."""
    n = len(A)
    one = np.eye(n, dtype=bool)
    adj = ((A != 0) | (A.T != 0)) & ~one
    D = np.where(one, 0.0, np.inf)
    seen = front = one
    for d in range(1, n):
        front = (front.astype(float) @ adj > 0) & ~seen
        if not front.any():
            break
        D[front] = d
        seen = seen | front
    D[np.isinf(D)] = D[np.isfinite(D)].max() + 1
    return D


def _layout(A):
    """Координаты вершин, единица — длина ребра: классическое MDS по числу рёбер между вершинами, затем
    stress majorization (SMACOF, вес пары 1/d², как neato в Graphviz). Длинная сторона — по горизонтали;
    от запуска к запуску одно и то же.
    """
    n = len(A)
    if n == 1:
        return np.zeros((1, 2))
    D = _hops(A)
    J = np.eye(n) - 1.0 / n
    lam, vec = np.linalg.eigh(-0.5 * J @ D ** 2 @ J)
    xy = vec[:, ::-1][:, :2] * np.sqrt(np.maximum(lam[::-1][:2], 0.0))
    xy = xy + 1e-3 * np.random.default_rng(0).standard_normal((n, 2))  # у звезды MDS кладёт вершины друг на друга
    W = np.where(D > 0, 1.0 / np.maximum(D, 1.0) ** 2, 0.0)
    Vinv = np.linalg.pinv(np.diag(W.sum(1)) - W)
    for _ in range(300):
        dist = np.sqrt(((xy[:, None] - xy[None]) ** 2).sum(-1))
        B = -W * D / np.where(dist > 1e-12, dist, np.inf)
        B[np.diag_indices(n)] = -B.sum(1)
        xy = Vinv @ B @ xy
    xy = xy - xy.mean(0)
    xy = xy @ np.linalg.eigh(xy.T @ xy)[1][:, ::-1]
    for a, sign in ((0, -1), (1, 1)):  # первая вершина не у центра — слева и сверху
        k = np.argmax(np.abs(xy[:, a]) > 0.1)
        if xy[k, a] * sign < 0:
            xy[:, a] = -xy[:, a]
    return xy


def _box(theta, g, w, h):
    """Подпись w × h px у кружка в сторону theta на расстоянии g px: сдвиг и привязка для аннотации и края
    подписи от центра кружка, px (лево, право, низ, верх; ось y вверх)."""
    c, s = np.cos(theta), np.sin(theta)
    xa = "left" if c > 0.3 else "right" if c < -0.3 else "center"
    ya = "bottom" if s > 0.3 else "top" if s < -0.3 else "middle"
    x0 = g * c - {"left": 0, "center": w / 2, "right": w}[xa]
    y0 = g * s - {"bottom": 0, "middle": h / 2, "top": h}[ya]
    return dict(xshift=g * c, yshift=g * s, xanchor=xa, yanchor=ya), (x0, x0 + w, y0, y0 + h)


def _spots(xy, k, nbrs, r, w, h):
    """Места для подписи w × h px у вершины k по порядку предпочтения: ближе к кружку (r + 3 … r + 39 px от центра),
    из одинаково близких — в середине более широкого промежутка между её рёбрами (из почти равных — смотрящего от
    центра графа, потом — ближе к «вниз»); у вершины без рёбер — справа, снизу, слева, сверху."""
    if nbrs:
        a = np.sort([np.arctan2(*(xy[j] - xy[k])[::-1]) for j in nbrs])
        gaps = np.diff(np.append(a, a[0] + 2 * np.pi))
        mid = a + gaps / 2
        out = xy[k] - xy.mean(0)
        far = np.abs(xy - xy.mean(0)).max()
        score = (gaps + 0.2 * np.cos(mid - np.arctan2(out[1], out[0])) * np.hypot(*out) / max(far, 1e-9)
                 - 0.01 * np.abs(np.angle(np.exp(1j * (mid + np.pi / 2)))))
        angles = mid[np.argsort(-score, kind="stable")]
    else:
        angles = [0.0, -np.pi / 2, np.pi, np.pi / 2]
    return [_box(t, g, w, h) for g in np.arange(r + 3, r + 40, 3.0) for t in angles]


def _hits(box, pts, circles, r, boxes):
    """Сколько задевает прямоугольник подписи (x0, x1, y0, y1, px; зазор 2 px): точки рёбер (через 3 px) — по 1,
    чужие кружки и подписи — по 100."""
    x0, x1, y0, y1 = box[0] - 2, box[1] + 2, box[2] - 2, box[3] + 2
    dx = np.maximum(np.maximum(x0 - circles[:, 0], circles[:, 0] - x1), 0)
    dy = np.maximum(np.maximum(y0 - circles[:, 1], circles[:, 1] - y1), 0)
    b = np.array(boxes).reshape(-1, 4)
    return (((pts[:, 0] > x0) & (pts[:, 0] < x1) & (pts[:, 1] > y0) & (pts[:, 1] < y1)).sum()
            + 100 * (dx ** 2 + dy ** 2 < r ** 2).sum()
            + 100 * ((b[:, 0] < x1) & (b[:, 1] > x0) & (b[:, 2] < y1) & (b[:, 3] > y0)).sum())


def _label_places(xy, s, r, edges, nbrs, wh, texts, boxes):
    """Подписи значений у вершин при s px на единицу: каждой — первое из мест _spots, где она ничего не задевает,
    а нет такого — где задевает меньше всего. boxes — подписи рёбер (px). Сначала вершины, у которых больше рёбер."""
    P = xy * s
    pts = np.array([P[i] + (P[j] - P[i]) * t for i, j in edges
                    for t in np.linspace(0, 1, int(np.hypot(*(P[j] - P[i])) // 3) + 2)]).reshape(-1, 2)  # рёбра
    boxes, place = list(boxes), [(None, (0, 0, 0, 0))] * len(xy)
    for k in sorted(range(len(xy)), key=lambda k: -len(nbrs[k])):
        if texts[k]:
            spots = _spots(xy, k, nbrs[k], r, *wh[k])
            shifted = [(x0 + P[k, 0], x1 + P[k, 0], y0 + P[k, 1], y1 + P[k, 1]) for _, (x0, x1, y0, y1) in spots]
            cost = []
            for b in shifted:
                cost.append(_hits(b, pts, np.delete(P, k, 0), r, boxes))
                if not cost[-1]:
                    break
            i = int(np.argmin(cost))  # первое из лучших
            place[k] = spots[i]
            boxes.append(shifted[i])
    return place


def _scale(pts, ext, plot_w, plot_h):
    """px на единицу длины: самая большая (не больше MAX_EDGE), при которой точки с краями ext вокруг
    (px: влево, вправо, вниз, вверх) влезают в рисунок."""
    s = float(MAX_EDGE)
    for a, size in ((0, plot_w), (1, plot_h)):
        d = pts[:, a][:, None] - pts[:, a][None]
        need = ext[:, 2 * a + 1][:, None] + ext[:, 2 * a][None]
        ok = d > 1e-9
        if ok.any():
            s = min(s, ((size - need[ok]) / d[ok]).min())
    return max(s, 1.0)


def _graph_figure(model, parts, features, width, height, verbose):
    """Рисунок для input=(X, A) или (X, A, E): сам граф, у вершин — значения до и после слоя."""
    X, A, E = _graph_arrays(parts)
    n = len(A)
    out = _graph_values(model, parts, n)
    names = None if features is None else [str(s) for s in np.ravel(np.asarray(features, dtype=object))]
    if names is not None and len(names) != n:
        raise ValueError("features: подписей {}, а вершин в графе {}".format(len(names), n))
    labels = [str(k) for k in range(n)] if names is None else [
        s if len(s) <= NAME_CHARS else s[:NAME_CHARS - 1] + "…" for s in names]

    def name(k):
        return str(k) if names is None else "{} ({})".format(k, labels[k])

    calc = _messages(model, X, A, E, out) if verbose else None
    xy = _layout(A)
    edges = [(i, j) for i in range(n) for j in range(i + 1, n) if A[i, j] or A[j, i]]
    nbrs = [[] for _ in range(n)]
    for i, j in edges:
        nbrs[i].append(j)
        nbrs[j].append(i)
    if X.shape[1] == out.shape[1] == 1:
        texts = ["{} → {}".format(_num(x), _num(y)) for x, y in zip(X[:, 0], out[:, 0])]
    else:
        texts = ["<br>".join(map(_num, y)) if len(y) <= 3 else "" for y in out]
    one = E is not None and E.shape[2] == 1  # у ребра одно число: толщина линии и подпись у середины
    ev = [E[i, j, 0] for i, j in edges] if one else []
    big = max(map(abs, ev), default=0.0)
    widths = [0.5 + 5 * abs(e) / big if big > 0 else 1.0 for e in ev] if one else [EDGE_PX] * len(edges)
    etexts = [_num(e) for e in ev]

    plot_w, plot_h = width - 20, height - 20
    wh = [(0.62 * VALUE_FONT * max(map(len, t.split("<br>"))) + 2, (t.count("<br>") + 1) * 1.3 * VALUE_FONT + 2)
          for t in texts]
    mids = np.array([(xy[i] + xy[j]) / 2 for i, j in edges]).reshape(-1, 2)
    half = [((0.6 * EDGE_FONT * len(t) + 4) / 2,) * 2 + ((1.3 * EDGE_FONT + 4) / 2,) * 2 for t in etexts]
    pts = np.vstack([xy, mids[:len(etexts)]])
    closest = (np.sqrt(((xy[:, None] - xy[None]) ** 2).sum(-1)) + np.diag(np.full(n, np.inf))).min()
    size, s = CIRCLE, None
    for _ in range(4):  # подписи <-> масштаб; кружки мельче, если граф большой (тогда и подписи ближе к центрам)
        r, used = size / 2, s
        if s is None:
            place = [_spots(xy, k, nbrs[k], r, *wh[k])[0] if texts[k] else (None, (0, 0, 0, 0)) for k in range(n)]
        else:
            eboxes = [(x * s - a, x * s + b, y * s - c, y * s + d) for (x, y), (a, b, c, d) in zip(mids, half)]
            place = _label_places(xy, s, r, edges, nbrs, wh, texts, eboxes)
        ext = np.array([[max(r, -x0), max(r, x1), max(r, -y0), max(r, y1)] for _, (x0, x1, y0, y1) in place]
                       + half).reshape(-1, 4)
        s = _scale(pts, ext, plot_w - 8, plot_h - 8)
        size = min(size, max(8.0, 0.8 * closest * s))
        if used is not None and abs(s - used) <= 0.01 * s:
            break
    r = size / 2
    ranges = []
    for a, span in ((0, plot_w), (1, plot_h)):
        c = (np.min(pts[:, a] - ext[:, 2 * a] / s) + np.max(pts[:, a] + ext[:, 2 * a + 1] / s)) / 2
        ranges.append([c - span / (2 * s), c + span / (2 * s)])

    fig = go.Figure()
    for (i, j), w in zip(edges, widths):
        (x0, y0), (x1, y1) = xy[i], xy[j]
        xs, ys, hover = [x0, x1], [y0, y1], dict(hoverinfo="skip")
        if verbose:  # точки подсказки — от края до края кружков, иначе у вершины подсказка ребра мешает своей
            L = max(np.hypot(x1 - x0, y1 - y0) * s, 1e-9)
            t = np.linspace(min(0.5, (r + 1) / L), max(0.5, 1 - (r + 1) / L), int(L // HOVER_PX) + 2)
            xs, ys = x0 + (x1 - x0) * t, y0 + (y1 - y0) * t
            hover = dict(hovertemplate=_edge_tip(i, j, name, A, E, calc) + "<extra></extra>")
        fig.add_trace(go.Scatter(x=xs, y=ys, mode="lines", line=dict(color=EDGE, width=w), **hover))
    font = 12 if size >= 30 else max(8.0, 0.4 * size)
    hover = dict(hoverinfo="skip")
    if verbose:
        hover = dict(hovertemplate=[_vertex_tip(k, name, X, A, out, calc) + "<extra></extra>" for k in range(n)])
    fig.add_trace(go.Scatter(x=xy[:, 0], y=xy[:, 1], mode="markers+text" if size >= 16 else "markers", text=labels,
                             textfont=dict(size=round(min(font, 0.85 * size / max(map(_label_em, labels))), 1)),
                             marker=dict(size=size, color="white", line=dict(color="black", width=1.5)), **hover))
    for (mx, my), t in zip(mids, etexts):
        fig.add_annotation(x=mx, y=my, text=t, showarrow=False, bgcolor=VALUE_BG, borderpad=1,
                           font=dict(size=EDGE_FONT, color="#444444"))
    for (x, y), t, (kw, _) in zip(xy, texts, place):
        if t:
            fig.add_annotation(x=x, y=y, text="<b>{}</b>".format(t), showarrow=False, bgcolor=VALUE_BG, borderpad=1,
                               font=dict(size=VALUE_FONT, color="#222222"), align=kw["xanchor"], **kw)
    fig.update_layout(width=width, height=height, showlegend=False, template="plotly_white", plot_bgcolor="white",
                      paper_bgcolor="white", margin=dict(l=10, r=10, t=10, b=10))
    if verbose:
        fig.update_layout(hovermode="closest",
                          hoverlabel=dict(bgcolor="white", font=dict(color="#222222"), align="left"))
    fig.update_xaxes(visible=False, range=ranges[0])
    fig.update_yaxes(visible=False, range=ranges[1])
    return fig


def figure(model, features=None, output=None, max_neurons=12, width=800, height=480, verbose=False, input=None):
    """Рисунок plotly (то же, что show(), но без вывода)."""
    if _is_graph(input):
        return _graph_figure(model, input, features, width, height, verbose)
    if max_neurons < 2:
        raise ValueError("max_neurons — не меньше 2")
    row = None if input is None else _row(input)
    net = _network(model, features, output, None if row is None else row.shape[1])
    weights = net["weights"]
    sizes = [weights[0].shape[0]] + [w.shape[1] for w in weights]
    n_cols, n_hidden = len(sizes), len(weights) - 1
    vals, zs = (None, None) if row is None else _values(model, net, row, sizes)
    features = _names(features if features is not None else net["features"], sizes[0], "features", "признаков",
                      strict=features is not None)
    outputs = _names(output, sizes[-1], "output", "выходов", strict=True)

    cols = []
    for c, n in enumerate(sizes):
        shown, gap = _shown(n, max_neurons)
        slots = len(shown) + (gap is not None)
        step = min(1.0, SPAN / (slots - 1)) if slots > 1 else 1.0
        slot_y = [step * ((slots - 1) / 2 - s) for s in range(slots)]
        if c == 0 and net.get("window"):
            labels = ["x<sub>i{}</sub>".format("+{}".format(k) if k else "") for k in shown]
        elif c == 0:
            labels = ["x<sub>{}</sub>".format(k + 1) for k in shown]
        elif c == n_cols - 1:
            labels = ["ŷ" if n == 1 else "ŷ<sub>{}</sub>".format(k + 1) for k in shown]
        else:
            sup = "<sup>({})</sup>".format(c) if n_hidden > 1 else ""
            labels = ["h<sub>{}</sub>{}".format(k + 1, sup) for k in shown]
        cols.append({
            "shown": shown,
            "y": [slot_y[p if gap is None or p < gap else p + 1] for p in range(len(shown))],
            "gap": None if gap is None else slot_y[gap],
            "step": step,
            "labels": labels,
        })

    plot_w, plot_h = width - 20, height - 20
    side = [CIRCLE / 2 + 6 + max(_text_px(s) for s in names) + 4 if names else 0.0 for names in (features, outputs)]
    step = min(col["step"] for col in cols)
    py = max(plot_h - CIRCLE - 12, 0.3 * plot_h) / SPAN
    shrink, shift, after = False, 0.0, 0.0
    if vals is not None:  # значения нейронов: под кружком, если в столбце есть место, иначе справа от кружка
        texts = [[_num(vals[c][k]) for k in col["shown"]] for c, col in enumerate(cols)]
        box = VALUE_FONT * 1.3 + 2
        py, shift = max(plot_h - CIRCLE - 12 - box - 2, 0.3 * plot_h) / SPAN, (box + 2) / 2  # место внизу
        shrink = step * py - box - 4 >= MIN_UNDER  # место есть во всех столбцах, кружки чуть меньше
        size = min(CIRCLE, 0.8 * step * py)
        for col in cols:
            col["under"] = shrink or col["step"] * py - box - 4 >= size
        vfont = VALUE_FONT if shrink else round(min(VALUE_FONT, max(9.0, 0.4 * step * py)), 1)
        vw = [max(0.62 * vfont * len(s) + 2 for s in t) for t in texts]
        if cols[0]["under"]:
            side[0] = max(side[0], vw[0] / 2 + 4)
        if cols[-1]["under"]:
            side[1] = max(side[1], vw[-1] / 2 + 4)
        else:
            after = vw[-1] + 1
            side[1] = side[1] + after if outputs else CIRCLE / 2 + 1 + after + 4
    (left, right), u = _x_layout(n_cols, plot_w, *side)

    step_px = step * py
    size = min(CIRCLE, 0.8 * step_px)
    if shrink:
        size = min(size, step_px - box - 4)
    r = size / 2
    font = 12 if size >= 30 else max(8.0, 0.4 * size)
    name_font = 12 if step_px >= 20 else round(max(7.0, 0.55 * step_px), 1)
    y_range = [-plot_h / (2 * py), plot_h / (2 * py)]
    if shift:
        y_range = [y - shift / py for y in y_range]

    fig = go.Figure()
    for i, W in enumerate(weights):
        big = np.abs(W).max()
        a, b = cols[i], cols[i + 1]
        for ka, ya, la in zip(a["shown"], a["y"], a["labels"]):
            for kb, yb, lb in zip(b["shown"], b["y"], b["labels"]):
                w = W[ka, kb]
                xs, ys, hover = [i, i + 1], [ya, yb], dict(hoverinfo="skip")
                if verbose:
                    n = int(np.hypot(u, (yb - ya) * py) // HOVER_PX) + 2
                    xs, ys = np.linspace(i, i + 1, n), np.linspace(ya, yb, n)
                    tip = "{} → {}<br>вес: {}".format(la, lb, _num(w))
                    if vals is not None:
                        v = vals[i][ka]
                        tip += "<br>вход · вес: {} · {} = {}".format(_par(v), _par(w), _num(v * w))
                    hover = dict(hovertemplate=tip + "<extra></extra>")
                fig.add_trace(go.Scatter(x=xs, y=ys, mode="lines", **hover,
                                         line=dict(color=RED if w > 0 else BLUE if w < 0 else GREY,
                                                   width=0.5 + 5 * abs(w) / big if big > 0 else 1.0)))
    for c, col in enumerate(cols):
        fit = round(min(font, 0.85 * size / max(map(_label_em, col["labels"]))), 1)
        hover = dict(hoverinfo="skip")
        if verbose and c > 0:
            bias = net["biases"][c - 1]
            tips = ["{}<br>смещение: {}".format(s, _num(bias[k])) for k, s in zip(col["shown"], col["labels"])]
            if vals is not None:
                W = weights[c - 1]  # одна строка счёта — только у слоя с линейной активацией, иначе z и φ(z)
                linear = net["activations"][c - 1] in ("identity", None) and np.array_equal(zs[c], vals[c])
                tips = [t + _formula(s, vals[c - 1], W[:, k], bias[k], zs[c][k], vals[c][k], linear)
                        for t, k, s in zip(tips, col["shown"], col["labels"])]
            hover = dict(hovertemplate=[t + "<extra></extra>" for t in tips])
        fig.add_trace(go.Scatter(x=[c] * len(col["y"]), y=col["y"], mode="markers+text" if size >= 16 else "markers",
                                 text=col["labels"], textfont=dict(size=fit), **hover,
                                 marker=dict(size=size, color="white", line=dict(color="black", width=1.5))))
        if col["gap"] is not None:
            fig.add_trace(go.Scatter(x=[c], y=[col["gap"]], mode="text", text=["⋮"], hoverinfo="skip",
                                     textfont=dict(size=round(max(10.0, min(22.0, 0.75 * col["step"] * py)), 1))))
    for c, names, anchor, sign in ((0, features, "right", -1), (n_cols - 1, outputs, "left", 1)):
        if not names:
            continue
        for j, y in zip(cols[c]["shown"], cols[c]["y"]):
            fig.add_annotation(x=c, y=y, xshift=sign * (r + 6 + (after if sign > 0 else 0)), text=names[j],
                               showarrow=False, xanchor=anchor, font=dict(size=name_font))
    if vals is not None:
        for c, col in enumerate(cols):
            place = (dict(yshift=-(r + 2), xanchor="center", yanchor="top") if col["under"]
                     else dict(xshift=r + 1, xanchor="left", yanchor="middle"))
            for s, y in zip(texts[c], col["y"]):
                fig.add_annotation(x=c, y=y, text="<b>{}</b>".format(s), showarrow=False, bgcolor=VALUE_BG,
                                   borderpad=1, font=dict(size=vfont, color="#222222"), **place)
    fig.update_layout(width=width, height=height, showlegend=False, template="plotly_white", plot_bgcolor="white",
                      paper_bgcolor="white", margin=dict(l=10, r=10, t=10, b=10))
    if verbose:
        fig.update_layout(hovermode="closest",
                          hoverlabel=dict(bgcolor="white", font=dict(color="#222222"), align="left"))
    fig.update_xaxes(visible=False, range=[-left, n_cols - 1 + right])
    fig.update_yaxes(visible=False, range=y_range)
    return fig


def show(model, features=None, output=None, max_neurons=12, width=800, height=480, verbose=False, input=None):
    """Нарисовать обученную модель под ячейкой как сеть: кружки — нейроны, линии — веса.

    model       -- модель scikit-learn: MLPRegressor, MLPClassifier, линейная модель (LinearRegression,
                   Ridge, Lasso, LogisticRegression, …) или Pipeline, который ею заканчивается,
                   или модель PyTorch: nn.Sequential или модуль из слоёв Linear и Conv1d, которые
                   идут друг за другом в том порядке, в каком заданы (активации, пулинг, Flatten,
                   Dropout не рисуются). Conv1d рисуется как одно окно: входы — kernel_size соседних
                   значений каждого канала (x_i, x_i+1, …), выходы — фильтры.
                   Необученная модель scikit-learn рисуется с нулевыми весами; для неё нужны
                   features или input: по ним определяется число признаков
    features    -- число признаков или их имена (подписи слева от входов); по умолчанию
                   model.feature_names_in_ — есть, если модель обучали на DataFrame
    output      -- число выходов, имя выхода (строка) или список имён (подписи справа от выходов)
    max_neurons -- сколько нейронов слоя рисовать: из слоя побольше — первые и последние, между ними «⋮»
    width, height -- размер рисунка, px
    verbose     -- True: подсказки при наведении мыши: на линии — вес и нейроны, которые она соединяет,
                   на нейроне скрытого слоя или выхода — его смещение; вместе с input — и счёт для этого
                   объекта: вход · вес на линии, взвешенная сумма z и φ(z) на нейроне (φ — функция активации)
    input       -- один объект: значения его признаков (список, массив NumPy, строка таблицы pandas,
                   тензор PyTorch). Рядом с каждым нарисованным нейроном пишется его значение для этого
                   объекта: у входов — признаки, у скрытых нейронов — выход функции активации, у выходов —
                   то, что выдаёт модель (предсказание ŷ; у классификатора scikit-learn с predict_proba —
                   вероятности классов, при двух классах — вероятность второго). У Pipeline input — то, что передают в pipe.predict, а у
                   входов пишутся значения после шагов перед моделью. Для модели с Conv1d не работает
    Цвет линии — знак веса (красный > 0, синий < 0, серый — вес равен нулю), толщина — |вес| относительно
    самого большого |веса| между теми же двумя слоями. Смещения не рисуются.

    Граф: input=(X, A, E) или input=(X, A) — тензоры или массивы NumPy: X — признаки вершин (n × F или n чисел),
    A — матрица смежности (n × n), E — признаки рёбер (n × n × K или n × n). Модель PyTorch вызывается так же:
    model(X, A, E) или model(X, A), и должна вернуть по строке на вершину (n × units или n чисел). Рисуется сам
    граф: кружки — вершины (внутри — номер вершины, строка X), линии — рёбра (A_ij ≠ 0). У вершины пишется
    «признак → после слоя», если это по одному числу, иначе — числа после слоя (до трёх) друг под другом.
    Если у ребра одно число-признак, оно пишется у середины линии, а толщина линии — |e| относительно самого
    большого |e|. features — подписи вершин вместо номеров (например, символы элементов); output и max_neurons
    не используются. verbose — подсказки: у вершины — её признак и значение после слоя, у ребра — его признак;
    у слоя вида m_ij = ReLU(W·[x_j, e_ij]), h_i = φ(Σ_j m_ij + b) (атрибуты w_msg и b, по желанию w_self — вклад самой вершины; units = 1, F + K ≤ 4) —
    и счёт: сообщение от каждой соседней вершины, их сумма, смещение.
    """
    figure(model, features=features, output=output, max_neurons=max_neurons, width=width, height=height,
           verbose=verbose, input=input).show()
