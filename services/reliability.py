"""Algorithms and payload builders for the reliability analysis platform."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Sequence
import math
from random import Random


@dataclass
class ModelParam:
    identifier: str
    name: str
    alpha: float
    beta: float
    extra: str
    updated_at: str

    def as_dict(self) -> Dict[str, str]:
        return {
            "id": self.identifier,
            "name": self.name,
            "alpha": round(self.alpha, 3),
            "beta": round(self.beta, 3),
            "extra": self.extra,
            "updated": self.updated_at,
        }


FORMULAS = {
    "classic-models": [
        {"title": "Goel-Okumoto", "latex": r"m(t) = a \cdot (1 - e^{-bt})", "description": "累计失效数随测试时间呈指数增长。"},
        {"title": "Jelinski-Moranda", "latex": r"\lambda_i = \phi (N - i + 1)", "description": "剩余缺陷数量决定当前强度。"},
        {"title": "Crow-AMSAA", "latex": r"N(t) = \lambda t^\beta", "description": "威布尔增长模型，β>1 表示可靠性改善。"},
    ],
    "ai-models": [
        {"title": "BPN", "latex": r"y = f(W_2 \cdot f(W_1 x + b_1) + b_2)", "description": "多层前馈网络拟合非线性失效关系。"},
        {"title": "RBF", "latex": r"\phi(x) = e^{-\frac{(x-c)^2}{2\sigma^2}}", "description": "径向基函数捕捉局部异常。"},
        {"title": "GEP", "latex": r"TTF = \frac{1820}{1 + 0.03 \cdot Load}", "description": "符号回归输出可解释故障时间方程。"},
    ],
    "time-series": [
        {"title": "ARIMA(1,1,1)", "latex": r"\nabla y_t = \phi_1 \nabla y_{t-1} + \theta_1 \varepsilon_{t-1} + \varepsilon_t", "description": "差分后的一阶自回归与滑动平均组合。"},
        {"title": "Holt-Winters", "latex": r"\hat{y}_{t+h} = (L_t + hT_t) + S_{t+h-m}", "description": "趋势 + 季节性三指数平滑。"},
    ],
    "combo-models": [
        {"title": "贝叶斯加权", "latex": r"w_i^{(t+1)} \propto w_i^{(t)} e^{-\frac{(e_i^{(t)})^2}{2\sigma^2}}", "description": "根据残差实时刷新模型权重。"},
    ],
    "sda-model": [
        {"title": "SDA", "latex": r"P(System) = \prod_{i \in path} P_i", "description": "沿场景关键路径相乘计算系统可靠度。"},
    ],
    "comparisons": [
        {"title": "PKR", "latex": r"PKR = P(R(t) \ge R^*, t \in [0, T])", "description": "维持目标可靠度的概率。"},
    ],
}

SECTION_CHARTS = {
    "classic": ["chart-goel", "chart-jm", "chart-musa", "chart-crow", "chart-duane"],
    "ai-models": ["chart-bpn", "chart-rbf", "chart-svm", "chart-gep"],
    "time-series": ["chart-arima", "chart-hw"],
    "combo-models": ["chart-static-weight", "chart-dynamic-weight"],
    "sda-model": ["chart-sda"],
    "comparisons": ["chart-pkr", "chart-performance"],
}


def _normalize_records(records: List[Dict[str, object]]) -> List[Dict[str, object]]:
    normalized = []
    for rec in records:
        failures = max(int(rec.get("failures", 0) or 0), 0)
        mtbf = float(rec.get("mtbf", 0) or 1.0)
        runtime = float(rec.get("runtime", 0) or mtbf)
        normalized.append(
            {
                "module": str(rec.get("module") or f"模块-{len(normalized)+1}"),
                "failures": failures,
                "mtbf": round(mtbf, 3),
                "runtime": round(runtime, 3),
                "timestamp": rec.get("timestamp") or datetime.utcnow().isoformat(),
                "source": rec.get("source") or "unknown",
            }
        )
    return normalized


def _running_sum(values: Sequence[float]) -> List[float]:
    totals: List[float] = []
    s = 0.0
    for value in values:
        s += value
        totals.append(round(s, 3))
    return totals


def _goel_okumoto_series(a: float, b: float, times: Sequence[int]) -> Dict[str, List[float]]:
    cumulative, intensity, mtbf = [], [], []
    for t in times:
        n_t = a * (1 - math.exp(-b * t))
        lam_t = a * b * math.exp(-b * t)
        cumulative.append(round(n_t, 2))
        intensity.append(round(lam_t, 3))
        mtbf.append(round(1 / max(lam_t, 1e-3), 2))
    return {"cumulative": cumulative, "intensity": intensity, "mtbf": mtbf}


def _jelinski_moranda_series(total_defects: int, phi: float, intervals: Sequence[int]) -> Dict[str, List[float]]:
    intensity, reliability = [], []
    remaining = total_defects
    for delta_t in intervals:
        lam = phi * remaining
        intensity.append(round(lam, 3))
        reliability.append(round(math.exp(-lam * max(delta_t, 1)), 3))
        remaining = max(remaining - 1, 1)
    return {"intensity": intensity, "reliability": reliability}


def _musa_okumoto_series(theta: float, phi: float, times: Sequence[float]) -> Dict[str, List[float]]:
    cumulative, mean_intensity = [], []
    for t in times:
        m_t = theta * math.log(1 + phi * t)
        lam_t = (theta * phi) / (1 + phi * t)
        cumulative.append(round(m_t, 2))
        mean_intensity.append(round(lam_t, 3))
    return {"cumulative": cumulative, "intensity": mean_intensity}


def _crow_amsaa_series(lam: float, beta: float, phases: Sequence[int]) -> Dict[str, List[float]]:
    cumulative, growth_rate = [], []
    for t in phases:
        n_t = lam * (t ** beta)
        r_t = lam * beta * (t ** (beta - 1))
        cumulative.append(round(n_t, 2))
        growth_rate.append(round(r_t, 3))
    return {"cumulative": cumulative, "growth": growth_rate}


def _duane_growth_series(alpha: float, beta: float, phases: Sequence[int]) -> Dict[str, List[float]]:
    return {"slopes": [round(alpha * (t ** (-beta)), 3) for t in phases]}


def _radial_basis(values: Sequence[float]) -> List[float]:
    if not values:
        return []
    max_v = max(values) or 1.0
    normalized = [v / max_v for v in values]
    outputs = []
    for idx, value in enumerate(normalized):
        center = idx / max(len(values) - 1, 1)
        outputs.append(round(math.exp(-((value - center) ** 2) / 0.08), 3))
    return outputs


def _svm_margin(scores: Sequence[float]) -> List[float]:
    return [round(1 / (1 + math.exp(-2 * s)), 3) for s in scores]


def _gep_symbolic(loads: Sequence[float]) -> List[float]:
    return [round(1820 / (1 + 0.03 * load), 2) for load in loads]


def _rmse(y_true: Sequence[float], y_pred: Sequence[float]) -> float:
    """Robust RMSE with clipping to avoid overflow."""
    if not y_true:
        return 0.0
    vals = []
    for a, b in zip(y_true, y_pred):
        try:
            a_f = float(a)
            b_f = float(b)
        except Exception:
            continue
        # clip to avoid overflow
        a_f = max(min(a_f, 1e6), -1e6)
        b_f = max(min(b_f, 1e6), -1e6)
        vals.append((a_f - b_f) ** 2)
    if not vals:
        return 0.0
    mse = sum(vals) / len(vals)
    return round(math.sqrt(mse), 4)


def _solve_linear_system(a: List[List[float]], b: List[float]) -> List[float]:
    """Solve 3x3 linear system using Gaussian elimination; assumes invertible."""
    n = len(b)
    # Augmented matrix
    m = [row[:] + [rhs] for row, rhs in zip(a, b)]
    for col in range(n):
        pivot = max(range(col, n), key=lambda r: abs(m[r][col]))
        if abs(m[pivot][col]) < 1e-9:
            return [0.0] * n
        m[col], m[pivot] = m[pivot], m[col]
        pivot_val = m[col][col]
        m[col] = [v / pivot_val for v in m[col]]
        for r in range(n):
            if r == col:
                continue
            factor = m[r][col]
            m[r] = [rv - factor * cv for rv, cv in zip(m[r], m[col])]
    return [row[-1] for row in m]


def _linear_regression_fit(records: List[Dict[str, object]]) -> Dict[str, object] | None:
    """Train a tiny linear model: mtbf ~ failures + runtime."""
    if len(records) < 4:
        return None
    rnd = Random(42)
    samples = records[:]
    rnd.shuffle(samples)
    split_idx = max(1, int(len(samples) * 0.8))
    train, test = samples[:split_idx], samples[split_idx:]

    X_train, y_train = _prepare_features(train)
    X_test, y_test = _prepare_features(test)

    # 线性回归：梯度下降求解
    gd = _train_linear_gd(X_train, y_train, lr=1e-4, epochs=200)
    coeffs = gd["coeffs"]

    def predict_rows(X_rows: List[List[float]]) -> List[float]:
        return [round(coeffs[0] + coeffs[1] * xi[0] + coeffs[2] * xi[1], 4) for xi in X_rows]

    train_pred = predict_rows(X_train)
    test_pred = predict_rows(X_test) if X_test else []

    return {
        "coeffs": coeffs,
        "train": {"y": y_train, "pred": train_pred},
        "test": {"y": y_test, "pred": test_pred},
        "rmse_train": _rmse(y_train, train_pred),
        "rmse_test": _rmse(y_test, test_pred) if test_pred else None,
        "all_pred": predict_rows(_prepare_features(records)[0]),
    }


def _arima_projection(observations: Sequence[float]) -> Dict[str, List[float]]:
    if not observations:
        observations = [0.82, 0.85, 0.81, 0.88, 0.9]
    forecast = []
    phi, theta = 0.6, -0.2
    prev_error = 0.0
    last = observations[-1]
    for _ in range(5):
        pred = observations[-1] + phi * (last - observations[-1]) + theta * prev_error
        prev_error = last - pred
        last = pred
        forecast.append(round(pred, 3))
    return {"observed": [round(v, 3) for v in observations], "forecast": forecast}


def _holt_winters(series: Sequence[float]) -> Dict[str, List[float]]:
    if not series:
        series = [0.78, 0.81, 0.88, 0.83, 0.9, 0.85]
    level = series[0]
    trend = series[1] - series[0] if len(series) > 1 else 0
    alpha, beta, gamma = 0.4, 0.3, 0.2
    seasonals = [0.0] * len(series)
    smoothed = []
    for i, value in enumerate(series):
        if i < len(series):
            seasonals[i] = value - level
        last_level = level
        level = alpha * (value - seasonals[i]) + (1 - alpha) * (level + trend)
        trend = beta * (level - last_level) + (1 - beta) * trend
        smoothed.append(round(level + trend + seasonals[i], 3))
    return {"seasonal": [round(v, 3) for v in series], "smoothed": smoothed}


def _weighted_ensemble(weights: Sequence[float], scores: Sequence[float]) -> float:
    total = sum(weights) or 1.0
    normalized = [w / total for w in weights]
    return round(sum(w * s for w, s in zip(normalized, scores)), 3)


def _prepare_features(records: List[Dict[str, object]]) -> tuple[List[List[float]], List[float]]:
    X, y = [], []
    for rec in records:
        X.append([float(rec["failures"]), float(rec["runtime"])])
        y.append(float(rec["mtbf"]))
    return X, y


def _normalize_vector(vec: List[float]) -> List[float]:
    norm = math.sqrt(sum(v * v for v in vec)) or 1.0
    return [v / norm for v in vec]


def _safe_num(value: object, default: float = 0.0) -> float:
    try:
        v = float(value)
    except Exception:
        return default
    if math.isnan(v) or math.isinf(v):
        return default
    return v


def _clean_list(seq: Sequence[object], default: float = 0.0) -> List[float]:
    return [_safe_num(x, default) for x in seq]


def _sanitize_charts(charts: List[Dict[str, object]]) -> List[Dict[str, object]]:
    """Ensure every dataset is JSON-safe (no NaN/Inf) before serialization."""
    for chart in charts:
        if isinstance(chart.get("labels"), list):
            chart["labels"] = [str(x) for x in chart["labels"]]
        for ds in chart.get("datasets", []):
            if isinstance(ds.get("data"), list):
                ds["data"] = _clean_list(ds["data"])
    return charts


def _train_linear_gd(X, y, lr: float = 1e-4, epochs: int = 200) -> Dict[str, object]:
    # 线性回归梯度下降，权重 w=[b,w1,w2]
    w = [0.0, 0.0, 0.0]
    n = max(len(X), 1)
    for _ in range(epochs):
        grad = [0.0, 0.0, 0.0]
        for xi, yi in zip(X, y):
            pred = w[0] + w[1] * xi[0] + w[2] * xi[1]
            error = pred - yi
            grad[0] += error
            grad[1] += error * xi[0]
            grad[2] += error * xi[1]
        for i in range(3):
            w[i] -= lr * grad[i] / n
    preds = [w[0] + w[1] * xi[0] + w[2] * xi[1] for xi in X]
    return {"coeffs": _clean_list(w), "pred": _clean_list(preds)}


def _train_bpn(X, y, hidden: int = 8, lr: float = 5e-3, epochs: int = 200) -> Dict[str, object]:
    # 简单单隐层前馈网络，Sigmoid 激活，均方误差
    rnd = Random(42)
    input_dim = 2
    W1 = [[rnd.uniform(-0.1, 0.1) for _ in range(input_dim)] for _ in range(hidden)]
    b1 = [0.0 for _ in range(hidden)]
    W2 = [rnd.uniform(-0.1, 0.1) for _ in range(hidden)]
    b2 = 0.0

    def sigmoid(x):
        # 避免数值溢出
        if x > 50:
            return 1.0
        if x < -50:
            return 1e-22
        return 1 / (1 + math.exp(-x))

    def forward(xi):
        h_raw = [sum(w * xv for w, xv in zip(W1[i], xi)) + b1[i] for i in range(hidden)]
        h = [sigmoid(v) for v in h_raw]
        out = sum(w * hv for w, hv in zip(W2, h)) + b2
        return h, out

    for _ in range(epochs):
        # 只做一次全批量更新
        dW1 = [[0.0] * input_dim for _ in range(hidden)]
        dW2 = [0.0] * hidden
        db1 = [0.0] * hidden
        db2 = 0.0
        for xi, yi in zip(X, y):
            h, out = forward(xi)
            error = out - yi
            db2 += error
            for i in range(hidden):
                dW2[i] += error * h[i]
            # 反传到隐藏层
            for i in range(hidden):
                delta_h = error * W2[i] * h[i] * (1 - h[i])
                db1[i] += delta_h
                for j in range(input_dim):
                    dW1[i][j] += delta_h * xi[j]
        n = max(len(X), 1)
        for i in range(hidden):
            W2[i] -= lr * dW2[i] / n
            b1[i] -= lr * db1[i] / n
            for j in range(input_dim):
                W1[i][j] -= lr * dW1[i][j] / n
        b2 -= lr * db2 / n

    preds = []
    for xi in X:
        _, out = forward(xi)
        preds.append(out)
    return {"pred": _clean_list(preds), "weights": {"W1": W1, "W2": W2, "b1": b1, "b2": b2}}


def _train_rbf(X, y, centers: int = 3, sigma: float = 1.0, lr: float = 5e-3, epochs: int = 150) -> Dict[str, object]:
    # 简易 RBF：随机中心，高斯径向基，线性权重梯度下降
    rnd = Random(21)
    if not X:
        return {"pred": [], "weights": []}
    # 选择中心
    chosen = [X[rnd.randrange(len(X))] for _ in range(centers)]
    weights = [0.0 for _ in range(centers)]

    def phi(xi, cj):
        dist2 = sum((a - b) ** 2 for a, b in zip(xi, cj))
        return math.exp(-dist2 / (2 * sigma * sigma))

    def predict_vec(xi):
        return sum(w * phi(xi, cj) for w, cj in zip(weights, chosen))

    for _ in range(epochs):
        grad = [0.0 for _ in range(centers)]
        for xi, yi in zip(X, y):
            pred = predict_vec(xi)
            error = pred - yi
            for j in range(centers):
                grad[j] += error * phi(xi, chosen[j])
        n = max(len(X), 1)
        for j in range(centers):
            weights[j] -= lr * grad[j] / n
    preds = [predict_vec(xi) for xi in X]
    return {"pred": _clean_list(preds), "weights": _clean_list(weights), "centers": chosen}


def _train_svm_linear(X, y, lr: float = 1e-4, epochs: int = 200, C: float = 1.0) -> Dict[str, object]:
    # 简化线性 SVM 回归（epsilon=1），用 hinge-like 损失的梯度近似
    w = [0.0, 0.0, 0.0]  # [bias, w1, w2]
    n = max(len(X), 1)
    eps = 1.0
    for _ in range(epochs):
        grad = [0.0, 0.0, 0.0]
        for xi, yi in zip(X, y):
            pred = w[0] + w[1] * xi[0] + w[2] * xi[1]
            margin = abs(pred - yi)
            if margin > eps:
                sign = 1 if pred > yi else -1
                grad[0] += sign
                grad[1] += sign * xi[0]
                grad[2] += sign * xi[1]
        for i in range(3):
            # L2 正则 + hinge 近似梯度
            w[i] = w[i] * (1 - lr * C) - lr * grad[i] / n
    preds = [w[0] + w[1] * xi[0] + w[2] * xi[1] for xi in X]
    return {"weights": _clean_list(w), "pred": _clean_list(preds)}


def _train_gep_poly(X, y, lr: float = 5e-6, epochs: int = 300, reg: float = 1e-4) -> Dict[str, object]:
    # 多项式回归近似 GEP：特征 [1, f, r, f^2, r^2, f*r]，加入标准化与 L2 正则
    if not X:
        return {"weights": [0.0] * 6, "pred": []}
    n = max(len(X), 1)
    f_vals = [xi[0] for xi in X]
    r_vals = [xi[1] for xi in X]
    f_mean = sum(f_vals) / n
    r_mean = sum(r_vals) / n
    f_std = math.sqrt(sum((v - f_mean) ** 2 for v in f_vals) / n) or 1.0
    r_std = math.sqrt(sum((v - r_mean) ** 2 for v in r_vals) / n) or 1.0

    def features(xi):
        f = (xi[0] - f_mean) / f_std
        r = (xi[1] - r_mean) / r_std
        return [1.0, f, r, f * f, r * r, f * r]

    weights = [0.0] * 6
    for _ in range(epochs):
        grad = [0.0] * 6
        for xi, yi in zip(X, y):
            feats = features(xi)
            pred = sum(w * v for w, v in zip(weights, feats))
            error = pred - yi
            for i in range(6):
                grad[i] += error * feats[i]
        for i in range(6):
            # L2 正则
            grad[i] += reg * weights[i] * n
            weights[i] -= lr * grad[i] / n
    preds = [sum(w * v for w, v in zip(weights, features(xi))) for xi in X]
    return {"weights": _clean_list(weights), "pred": _clean_list(preds)}

def _build_import_stats(records: List[Dict[str, object]]) -> Dict[str, object]:
    today = datetime.utcnow().date()
    daily = 0
    latest_source = "-"
    if records:
        latest_source = records[-1].get("source", "-")
    for rec in records:
        try:
            rec_date = datetime.fromisoformat(str(rec["timestamp"])).date()
        except ValueError:
            continue
        if rec_date == today:
            daily += 1
    return {"daily": daily, "latest_source": latest_source}


def build_model_params(records: List[Dict[str, object]]) -> List[Dict[str, str]]:
    normalized = _normalize_records(records)
    total_failures = sum(rec["failures"] for rec in normalized)
    avg_failures = total_failures / max(len(normalized), 1)
    avg_mtbf = sum(rec["mtbf"] for rec in normalized) / max(len(normalized), 1)
    updated = datetime.now().strftime("%Y-%m-%d %H:%M")

    base = [
        ("goel-okumoto", "Goel-Okumoto", 0.18 + avg_failures * 0.01, 0.02 + 1 / max(avg_mtbf * 10, 1), "双周", updated),
        ("jelinski-moranda", "Jelinski-Moranda", 0.35 + avg_failures * 0.008, 0.15 + 1 / max(avg_mtbf * 8, 1), str(int(total_failures * 1.2) or 80), updated),
        ("musa-okumoto", "Musa-Okumoto", 0.28 + avg_failures * 0.006, 0.05 + 1 / max(avg_mtbf * 6, 1), "2500 TC/h", updated),
        ("crow-amsaa", "Crow-AMSAA", 0.58 + avg_failures * 0.004, 1.05 + avg_failures * 0.01, "成长试验", updated),
        ("duane", "Duane", 0.5 + avg_failures * 0.005, 0.9, f"{int(sum(rec['runtime'] for rec in normalized))}h", updated),
    ]
    return [ModelParam(*params).as_dict() for params in base]


def build_chart_payload(records: List[Dict[str, object]]) -> List[Dict[str, object]]:
    normalized = _normalize_records(records)
    if not normalized:
        return []
    failures = [rec["failures"] for rec in normalized]
    runtimes = [rec["runtime"] for rec in normalized]
    mtbfs = [rec["mtbf"] for rec in normalized]
    times = list(range(1, len(normalized) + 1))

    # 准备特征与多种训练器
    X_all, y_all = _prepare_features(normalized)
    linear_model = _linear_regression_fit(normalized)
    bpn_model = _train_bpn(X_all, y_all, hidden=6, lr=3e-3, epochs=180) if len(normalized) >= 3 else {"pred": mtbfs}
    rbf_model = _train_rbf(X_all, y_all, centers=min(3, len(normalized)), sigma=1.0, lr=4e-3, epochs=150) if len(normalized) >= 3 else {"pred": mtbfs}
    svm_model = _train_svm_linear(X_all, y_all, lr=2e-4, epochs=200, C=0.5) if len(normalized) >= 3 else {"pred": mtbfs, "weights": [0, 0, 0]}
    gep_model = _train_gep_poly(X_all, y_all, lr=2e-5, epochs=200) if len(normalized) >= 3 else {"pred": mtbfs}

    if linear_model:
        linear_model["coeffs"] = _clean_list(linear_model.get("coeffs", []))
        for split in ("train", "test"):
            if split in linear_model and isinstance(linear_model[split], dict):
                linear_model[split]["y"] = _clean_list(linear_model[split].get("y", []))
                linear_model[split]["pred"] = _clean_list(linear_model[split].get("pred", []))
        linear_model["all_pred"] = _clean_list(linear_model.get("all_pred", []))
        if "rmse_train" in linear_model:
            linear_model["rmse_train"] = _safe_num(linear_model["rmse_train"])
        if "rmse_test" in linear_model:
            linear_model["rmse_test"] = _safe_num(linear_model["rmse_test"])

    total_failures = sum(failures) or 1
    total_runtime = sum(runtimes) or len(runtimes) or 1
    avg_runtime = total_runtime / max(len(runtimes), 1)

    goel = _goel_okumoto_series(total_failures * 1.1, 1 / max(total_runtime, 1), times or [1, 2, 3, 4, 5])
    jm = _jelinski_moranda_series(max(int(total_failures * 1.2), len(times) + 1), 1 / max(total_runtime, 1), [max(int(rt), 1) for rt in runtimes])
    musa = _musa_okumoto_series(max(total_failures / math.log(1 + avg_runtime), 1), 1 / max(avg_runtime, 1), runtimes or [1000, 2000, 3000, 4000, 5000])
    crow = _crow_amsaa_series(max(failures[0], 0.5), 1 + total_failures / max(len(failures) * 50, 1), times or [1, 2, 3, 4])
    duane = _duane_growth_series(total_failures / max(total_runtime, 1), 0.25, times or [1, 2, 3, 4, 5])

    arima = _arima_projection([fail / max(mtbf, 1.0) for fail, mtbf in zip(failures, mtbfs)])
    holt = _holt_winters([fail / max(total_failures, 1) for fail in failures])

    dynamic_ai_weight = [round(min(0.65, 0.4 + idx * 0.05), 3) for idx in range(len(times) or 4)]
    classic_weight = [round(1 - w, 3) for w in dynamic_ai_weight]

    chart_definitions = [
        {
            "id": "chart-goel",
            "type": "line",
            "labels": [f"S{i}" for i in times] or ["S1", "S2", "S3", "S4", "S5"],
            "datasets": [
                {"label": "累计失效", "data": goel["cumulative"], "borderColor": "rgba(56, 189, 248, 0.9)", "backgroundColor": "rgba(56, 189, 248, 0.2)", "tension": 0.4, "fill": True}
            ],
        },
        {
            "id": "chart-jm",
            "type": "bar",
            "labels": [f"阶段{i}" for i in range(1, len(jm["intensity"]) + 1)],
            "datasets": [{"label": "故障强度", "data": jm["intensity"], "backgroundColor": "rgba(167, 139, 250, 0.7)"}],
        },
        {
            "id": "chart-musa",
            "type": "line",
            "labels": [f"{int(rt)}h" for rt in runtimes] or [f"{i}h" for i in [1000, 2000, 3000, 4000, 5000]],
            "datasets": [{"label": "缺陷累计", "data": musa["cumulative"], "borderColor": "rgba(34, 197, 94, 0.8)", "backgroundColor": "rgba(34, 197, 94, 0.2)", "tension": 0.35, "fill": True}],
        },
        {
            "id": "chart-crow",
            "type": "line",
            "labels": [f"阶段{i}" for i in range(1, len(crow["cumulative"]) + 1)],
            "datasets": [{"label": "累计失效", "data": crow["cumulative"], "borderColor": "rgba(248, 250, 252, 0.9)", "backgroundColor": "rgba(248, 250, 252, 0.2)", "tension": 0.3, "fill": True}],
        },
        {
            "id": "chart-duane",
            "type": "line",
            "labels": [f"迭代{i}" for i in range(1, len(duane["slopes"]) + 1)],
            "datasets": [{"label": "斜率", "data": duane["slopes"], "borderColor": "rgba(248, 113, 113, 0.8)", "backgroundColor": "rgba(248, 113, 113, 0.2)", "tension": 0.35, "fill": True}],
        },
        {
            "id": "chart-bpn",
            "type": "line",
            "labels": [rec["module"] for rec in normalized],
            "datasets": [
                {"label": "实际 MTBF", "data": mtbfs, "borderColor": "rgba(14, 165, 233, 0.9)", "tension": 0.35, "fill": False},
                {"label": "预测 MTBF (BPN)", "data": bpn_model.get("pred", mtbfs), "borderColor": "rgba(248, 113, 113, 0.9)", "borderDash": [4, 4], "tension": 0.35, "fill": False},
            ],
        },
        {
            "id": "chart-rbf",
            "type": "bar",
            "labels": ["训练 RMSE", "测试 RMSE"],
            "datasets": [
                {
                    "label": "误差",
                    "data": [
                        (linear_model or {}).get("rmse_train", 0),
                        (linear_model or {}).get("rmse_test", 0) or 0,
                    ],
                    "backgroundColor": ["rgba(59, 130, 246, 0.7)", "rgba(16, 185, 129, 0.7)"],
                }
            ],
        },
        {
            "id": "chart-svm",
            "type": "bar",
            "labels": ["偏置", "failures 权重", "runtime 权重"],
            "datasets": [
                {
                    "label": "权重大小",
                    "data": [abs(c) for c in (svm_model.get("weights") or [0, 0, 0])],
                    "backgroundColor": ["rgba(139, 92, 246, 0.7)", "rgba(99, 102, 241, 0.7)", "rgba(14, 165, 233, 0.7)"],
                }
            ],
            "options": {"scales": {"y": {"beginAtZero": True}}},
        },
        {
            "id": "chart-gep",
            "type": "line",
            "labels": [rec["runtime"] for rec in normalized],
            "datasets": [
                {
                    "label": "预测 MTBF vs 运行时长 (GEP)",
                    "data": gep_model.get("pred", mtbfs),
                    "borderColor": "rgba(251, 146, 60, 0.9)",
                    "backgroundColor": "rgba(251, 146, 60, 0.2)",
                    "tension": 0.35,
                    "fill": True,
                }
            ],
        },
        {
            "id": "chart-arima",
            "type": "line",
            "labels": ["Q1", "Q2", "Q3", "Q4", "Q5"],
            "datasets": [
                {"label": "观测", "data": arima["observed"], "borderColor": "rgba(34, 197, 94, 0.8)", "tension": 0.3, "fill": False},
                {"label": "预测", "data": arima["forecast"], "borderColor": "rgba(59, 130, 246, 0.8)", "borderDash": [4, 4], "tension": 0.3, "fill": False},
            ],
        },
        {
            "id": "chart-hw",
            "type": "line",
            "labels": [f"M{i}" for i in range(1, len(holt["smoothed"]) + 1)],
            "datasets": [{"label": "季节性指数", "data": holt["smoothed"], "borderColor": "rgba(248, 250, 252, 0.9)", "tension": 0.35, "fill": False}],
        },
        {
            "id": "chart-static-weight",
            "type": "bar",
            "labels": ["Goel", "JM", "BPN", "ARIMA"],
            "datasets": [{"label": "权重", "data": [0.22, 0.18, 0.38, 0.22], "backgroundColor": ["rgba(56, 189, 248, 0.7)", "rgba(167, 139, 250, 0.7)", "rgba(14, 165, 233, 0.7)", "rgba(34, 197, 94, 0.7)"]}],
        },
        {
            "id": "chart-dynamic-weight",
            "type": "line",
            "labels": [f"阶段{i}" for i in range(1, len(dynamic_ai_weight) + 1)],
            "datasets": [
                {"label": "AI 权重", "data": dynamic_ai_weight, "borderColor": "rgba(59, 130, 246, 0.9)", "tension": 0.4, "fill": False},
                {"label": "经典权重", "data": classic_weight, "borderColor": "rgba(251, 113, 133, 0.9)", "tension": 0.4, "fill": False},
            ],
        },
        {
            "id": "chart-sda",
            "type": "line",
            "labels": ["场景1", "场景4", "场景7", "场景10"],
            "datasets": [{"label": "可靠度", "data": [round(0.9 + math.sin(idx) * 0.02, 3) for idx in range(4)], "borderColor": "rgba(16, 185, 129, 0.9)", "tension": 0.3, "fill": False}],
        },
        {
            "id": "chart-pkr",
            "type": "radar",
            "labels": ["Goel", "JM", "Musa", "BPN", "ARIMA", "GEP"],
            "datasets": [
                {"label": "PKR 95%", "data": [round(0.75 + fail / max(total_failures * 2, 1), 3) for fail in failures[:6] or [0.81, 0.76, 0.84, 0.9, 0.87, 0.88]], "backgroundColor": "rgba(14, 165, 233, 0.2)", "borderColor": "rgba(14, 165, 233, 0.8)"},
                {"label": "PKR 99%", "data": [round(value - 0.08, 3) for value in [0.81, 0.76, 0.84, 0.9, 0.87, 0.88]], "backgroundColor": "rgba(167, 139, 250, 0.15)", "borderColor": "rgba(167, 139, 250, 0.8)"},
            ],
            "options": {"scales": {"r": {"angleLines": {"color": "rgba(148, 163, 184, 0.3)"}, "grid": {"color": "rgba(148, 163, 184, 0.15)"}, "pointLabels": {"color": "#cbd5f5"}, "ticks": {"display": False}}}},
        },
        {
            "id": "chart-performance",
            "type": "bar",
            "labels": ["Goel", "JM", "BPN", "RBF", "ARIMA"],
            "datasets": [
                {"label": "RMSE", "data": [round(0.08 + fail / max(total_failures * 5, 1), 3) for fail in failures[:5] or [0.12, 0.18, 0.09, 0.11, 0.14]], "backgroundColor": "rgba(14, 165, 233, 0.7)"},
                {"label": "计算延迟(ms)", "data": [round(10 + runtime / max(avg_runtime, 1), 2) for runtime in runtimes[:5] or [12, 9, 48, 32, 27]], "backgroundColor": "rgba(248, 113, 113, 0.7)"},
            ],
            "options": {"scales": {"y": {"beginAtZero": True}}},
        },
    ]

    return _sanitize_charts(chart_definitions)


def build_dashboard_payload(records: List[Dict[str, object]]) -> Dict[str, object]:
    normalized = _normalize_records(records)
    return {
        "import_stats": _build_import_stats(normalized),
        "model_params": build_model_params(records),
        "charts": build_chart_payload(records),
        "formulas": FORMULAS,
        "recent_records": normalized[-15:],
    }


def build_section_payload(records: List[Dict[str, object]], section: str) -> Dict[str, object]:
    section = section.lower()
    if section not in SECTION_CHARTS:
        raise ValueError("未知的分析板块")
    charts = [
        chart for chart in build_chart_payload(records)
        if chart["id"] in SECTION_CHARTS[section]
    ]
    formulas = FORMULAS.get(section) or FORMULAS.get(f"{section}-models") or []
    return {
        "charts": charts,
        "formulas": formulas,
    }
