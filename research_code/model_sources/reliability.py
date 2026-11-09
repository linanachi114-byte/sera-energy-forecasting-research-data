from __future__ import annotations

import sys
import json
import hashlib
from pathlib import Path
from urllib.parse import urlparse

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

RESEARCH_DIR = Path(__file__).resolve().parents[2]
if str(RESEARCH_DIR) not in sys.path:
    sys.path.insert(0, str(RESEARCH_DIR))

from config import LLM_CACHE_DIR, LLM_CONFIG  # noqa: E402
from llm_feature import (  # noqa: E402
    ETT_COLUMN_DESC,
    ETT_TARGET_DESC,
    LLMFeatureEncoder,
    LLMFeatureExtractor,
    _compute_hash,
    _load_from_cache,
    _open_cache_db,
    _provider_identity,
    build_llm_prompt,
)

IMPACT_TO_VALUE = LLMFeatureEncoder._MAP

REASON_KEYWORDS = (
    "temperature", "temp", "humidity", "load", "energy", "price", "weather",
    "wind", "visibility", "activity", "hour", "occupancy", "负荷", "温度",
    "湿度", "能耗", "电价", "天气", "人员", "时间",
)

DATASET_COLUMN_DESC = {
    "EnergyConsumption": {
        "Temperature": "outdoor temperature",
        "Occupancy": "building occupancy level",
        "SquareFootage": "building floor area",
        "Hour": "hour of day",
    },
    "Appliances": {
        "T_out": "outdoor temperature",
        "RH_out": "outdoor relative humidity",
        "Windspeed": "wind speed",
        "Visibility": "visibility",
    },
    "energy_price": {
        "avg_temperature": "average temperature",
        "humidity": "humidity",
        "energy_consumption": "energy consumption",
        "industrial_activity_index": "industrial activity index",
    },
    "Z1": {
        "Temperature": "air temperature",
        "Humidity": "humidity",
        "Wind Speed": "wind speed",
        "general diffuse flows": "general diffuse solar flow",
        "diffuse flows": "diffuse solar flow",
    },
    "Z2": {
        "Temperature": "air temperature",
        "Humidity": "humidity",
        "Wind Speed": "wind speed",
        "general diffuse flows": "general diffuse solar flow",
        "diffuse flows": "diffuse solar flow",
    },
    "Z3": {
        "Temperature": "air temperature",
        "Humidity": "humidity",
        "Wind Speed": "wind speed",
        "general diffuse flows": "general diffuse solar flow",
        "diffuse flows": "diffuse solar flow",
    },
    "zonal_price": {
        "forecast_total_load": "forecast total system load",
        "forecast_zonal_load": "forecast zonal load",
    },
    "load": {
        "temperature_mean": "mean temperature across weather stations",
        "temperature_std": "cross-station temperature standard deviation",
        "temperature_min": "minimum temperature across weather stations",
        "temperature_max": "maximum temperature across weather stations",
    },
    "load_actual": {
        "load_forecast": "published electricity load forecast",
        "solar_generation": "observed solar generation at the forecast origin",
        "wind_generation": "observed wind generation at the forecast origin",
    },
    "Global_active_power": {
        "Global_reactive_power": "household global reactive power",
        "Voltage": "household voltage",
        "Global_intensity": "household current intensity",
        "Sub_metering_1": "kitchen sub-metering",
        "Sub_metering_2": "laundry-room sub-metering",
        "Sub_metering_3": "water-heater and air-conditioner sub-metering",
    },
    "total_load": {
        "client_std": "cross-client load standard deviation",
        "active_clients": "number of clients with positive load",
    },
    "site_electricity": {
        "airTemperature": "site air temperature",
        "dewTemperature": "site dew-point temperature",
        "seaLvlPressure": "sea-level pressure",
        "windSpeed": "site wind speed",
    },
    "Usage_kWh": {
        "Lagging_Current_Reactive.Power_kVarh": "lagging-current reactive energy",
        "Leading_Current_Reactive_Power_kVarh": "leading-current reactive energy",
        "Lagging_Current_Power_Factor": "lagging-current power factor",
        "Leading_Current_Power_Factor": "leading-current power factor",
    },
    "demand": {
        "hour_sin": "cyclic hour-of-day sine coordinate",
        "hour_cos": "cyclic hour-of-day cosine coordinate",
        "day_of_week_sin": "cyclic day-of-week sine coordinate",
        "day_of_week_cos": "cyclic day-of-week cosine coordinate",
    },
    "solar_total_power": {
        "hour_sin": "cyclic hour-of-day sine coordinate",
        "hour_cos": "cyclic hour-of-day cosine coordinate",
        "day_of_year_sin": "cyclic day-of-year sine coordinate",
        "day_of_year_cos": "cyclic day-of-year cosine coordinate",
        "site_power_std": "cross-site solar-power standard deviation",
        "active_sites": "number of solar sites with positive generation",
    },
}

TARGET_DESC = {
    "EnergyConsumption": "building energy consumption",
    "Appliances": "household appliance energy consumption",
    "energy_price": "German energy price",
    "OT": "transformer oil temperature",
    "Z1": "Tetuan zone 1 electricity load",
    "Z2": "Tetuan zone 2 electricity load",
    "Z3": "Tetuan zone 3 electricity load",
    "zonal_price": "GEFCom zonal electricity price",
    "load": "GEFCom system electricity load",
    "load_actual": "Germany electricity load",
    "Global_active_power": "household active electric power",
    "total_load": "aggregate electricity load across clients",
    "site_electricity": "aggregate building-site electricity load",
    "Usage_kWh": "steel-industry electricity consumption",
    "demand": "Victoria electricity demand",
    "solar_total_power": "regional solar power generation in Alabama",
}

ETT_COLUMN_DESC_EN = {
    "HUFL": "high-useful load",
    "HULL": "high-useless load",
    "MUFL": "middle-useful load",
    "MULL": "middle-useless load",
    "LUFL": "low-useful load",
    "LULL": "low-useless load",
}


def _numeric_row(df: pd.DataFrame, row_index: int, columns: list[str]) -> dict[str, float | str]:
    """一次读取指定行，并将所需列转换为可稳定序列化的 Python 标量。"""

    source = df.iloc[row_index]
    row: dict[str, float | str] = {}
    for col in columns:
        value = source[col]
        row[col] = float(value) if pd.api.types.is_numeric_dtype(df[col]) else str(value)
    return row


def _legacy_compute_hash(hist_data, curr_data, history_window, target) -> str:
    """计算旧版 LLM 语义缓存哈希。

    入参：
        hist_data：传给 LLM 的历史窗口数据。
        curr_data：当前外生变量快照。
        history_window：LLM 可见的历史步数。
        target：目标变量描述。

    实现功能：
        兼容早期实验中未加入 cache_namespace 的缓存文件。旧缓存仍然是真实 LLM
        返回结果，因此 strict 模式可以读取它们；但新生成的缓存仍使用新版哈希。

    返回值：
        旧版 MD5 哈希字符串。
    """

    raw = json.dumps(
        {
            "hist": history_window,
            "target": target,
            "hist_data": hist_data,
            "curr_data": curr_data,
        },
        sort_keys=True,
        ensure_ascii=False,
    )
    return hashlib.md5(raw.encode("utf-8")).hexdigest()


def _protocol_v1_compute_hash(hist_data, curr_data, history_window, target, cache_namespace) -> str:
    """计算上一版仅含 cache namespace 的缓存键。

    入参：历史数据、当前快照、历史窗口、目标描述和旧缓存命名空间。
    功能：只用于非 strict 的历史实验兼容；正式实验不得接受该协议。
    返回：旧版 MD5 内容键。
    """

    raw = json.dumps(
        {
            "cache_namespace": cache_namespace,
            "hist": history_window,
            "target": target,
            "hist_data": hist_data,
            "curr_data": curr_data,
        },
        sort_keys=True,
        ensure_ascii=False,
    )
    return hashlib.md5(raw.encode("utf-8")).hexdigest()


def _cache_keys(
    df: pd.DataFrame,
    target: str,
    ext_cols: list[str],
    index: int,
    forecast_horizon: int = 1,
) -> tuple[list[str], list[dict], dict]:
    """生成当前样本的 horizon-aware 新键和只读旧协议候选键。

    入参：
        df：完整数据表。
        target：目标变量列名。
        ext_cols：外生变量列名。
        index：当前语义样本对应的原始行号。

    实现功能：
        构造 LLM 输入所需的历史窗口和当前快照，并同时返回新版哈希和旧版哈希。
        这样正式实验可以复用历史真实 LLM 输出，同时避免因为哈希升级导致大量重复调用。

    返回值：
        (候选哈希列表, 历史窗口数据, 当前快照)。列表首项始终是正式新协议；
        后续键仅供显式开启旧缓存兼容的探索流程使用。
    """

    if forecast_horizon < 1:
        raise ValueError(f"forecast_horizon must be positive, got {forecast_horizon}")

    n_hist = min(LLM_CONFIG["history_window"], index)
    hist_items = []
    for j in range(max(0, index - n_hist), index):
        row = _numeric_row(df, j, ext_cols)
        row["target"] = float(df.iloc[j][target])
        hist_items.append(row)
    curr_items = _numeric_row(df, index, ext_cols)
    # 正式 lagged 协议下 index 正是预测起点前最后一个可观测时刻。
    # 将目标值显式放入快照，保证提示词中的比较基准真实存在于 LLM 输入中。
    curr_items["last_observed_target"] = float(df.iloc[index][target])

    is_ett = any(col in set(ETT_COLUMN_DESC) for col in ext_cols)
    target_name = ETT_TARGET_DESC if is_ett else target
    current_hash = _compute_hash(
        hist_items,
        curr_items,
        LLM_CONFIG["history_window"],
        target_name,
        cache_namespace=LLM_CONFIG.get("cache_namespace"),
        forecast_horizon=forecast_horizon,
        prompt_version=LLM_CONFIG.get("prompt_version"),
        schema_version=LLM_CONFIG.get("schema_version"),
        model=LLM_CONFIG.get("model"),
        temperature=LLM_CONFIG.get("temperature"),
        max_tokens=LLM_CONFIG.get("max_tokens"),
    )
    protocol_v1_hash = _protocol_v1_compute_hash(
        hist_items,
        curr_items,
        LLM_CONFIG["history_window"],
        target_name,
        LLM_CONFIG.get("cache_namespace"),
    )
    legacy_hash = _legacy_compute_hash(hist_items, curr_items, LLM_CONFIG["history_window"], target_name)
    keys = [current_hash]
    for candidate in (protocol_v1_hash, legacy_hash):
        if candidate not in keys:
            keys.append(candidate)
    return (
        keys,
        hist_items,
        curr_items,
    )


def _translate_prompt_data(data, descriptions: dict[str, str]):
    """按数据集变量说明递归改写提示词中的字段名。"""

    if isinstance(data, dict):
        return {
            descriptions.get(key, key): _translate_prompt_data(value, descriptions)
            if isinstance(value, (dict, list)) else value
            for key, value in data.items()
        }
    if isinstance(data, list):
        return [_translate_prompt_data(item, descriptions) for item in data]
    return data


def _expected_prompt_hash(
    target: str,
    ext_cols: list[str],
    hist_items: list[dict],
    curr_items: dict,
    forecast_horizon: int,
) -> str:
    """重建当前样本提示词并返回 SHA-256，用于 strict 来源校验。"""

    prompt = build_cache_prompt(
        target,
        ext_cols,
        hist_items,
        curr_items,
        forecast_horizon,
    )
    return hashlib.sha256(prompt.encode("utf-8")).hexdigest()


def build_cache_prompt(
    target: str,
    ext_cols: list[str],
    hist_items: list[dict],
    curr_items: dict,
    forecast_horizon: int,
) -> str:
    """重建正式缓存记录对应的完整单样本提示词。

    入参：目标列、外生列、预测起点前历史、预测起点快照和预测步长。
    功能：使用与训练时完全一致的变量说明和目标名称生成提示词；函数不
    接收未来标签，可供实时调用和 Batch File 两种传输方式共同使用。
    返回：可直接发送给 LLM 的提示词字符串。
    """

    is_ett = any(column in set(ETT_COLUMN_DESC) for column in ext_cols)
    target_name = TARGET_DESC.get(target, target)
    descriptions = ETT_COLUMN_DESC_EN if is_ett else DATASET_COLUMN_DESC.get(target, {})
    return build_llm_prompt(
        _translate_prompt_data(hist_items, descriptions),
        _translate_prompt_data(curr_items, descriptions),
        history_window=int(LLM_CONFIG["history_window"]),
        target_name=target_name,
        forecast_horizon=int(forecast_horizon),
        column_descriptions=descriptions,
    )


def _validate_cache_record(
    record: dict | None,
    forecast_horizon: int,
    expected_prompt_hash: str,
) -> tuple[dict | None, bool, str, dict]:
    """验证新缓存记录是否严格匹配本次语义协议。

    入参：完整缓存记录、预测步长和按当前样本重建的提示词哈希。
    功能：校验 response/provenance 结构、模型、服务来源、提示词与模式版本、
    生成参数、horizon、时间和重试审计字段。API 密钥不参与也不写入记录。
    返回：`(response, passed, reason, provenance)`。
    """

    if not isinstance(record, dict):
        return None, False, "not_a_json_object", {}
    if not isinstance(record.get("response"), dict) or not isinstance(record.get("provenance"), dict):
        return record, False, "legacy_cache_protocol", {}
    response = record["response"]
    provenance = record["provenance"]
    generation = provenance.get("generation_parameters")
    required_audit_fields = (
        "model",
        "provider",
        "base_url_host",
        "prompt_sha256",
        "prompt_version",
        "schema_version",
        "forecast_horizon",
        "created_at",
        "latency_seconds",
        "retry_count",
        "schema",
    )
    missing = [field for field in required_audit_fields if field not in provenance]
    if missing or not isinstance(generation, dict):
        return response, False, f"missing_provenance:{','.join(missing) or 'generation_parameters'}", provenance

    expected_host = (urlparse(str(LLM_CONFIG.get("base_url", ""))).hostname or "").lower()
    expected_provider, _ = _provider_identity(str(LLM_CONFIG.get("base_url", "")))
    try:
        checks = {
            "model": str(provenance.get("model")) == str(LLM_CONFIG.get("model", "")),
            "provider": str(provenance.get("provider", "")) == expected_provider,
            "base_url_host": str(provenance.get("base_url_host", "")).lower() == expected_host,
            "prompt_sha256": str(provenance.get("prompt_sha256", "")) == expected_prompt_hash,
            "prompt_version": str(provenance.get("prompt_version", "")) == str(LLM_CONFIG.get("prompt_version", "")),
            "schema_version": str(provenance.get("schema_version", "")) == str(LLM_CONFIG.get("schema_version", "")),
            "forecast_horizon": int(provenance.get("forecast_horizon", -1)) == int(forecast_horizon),
            "temperature": float(generation.get("temperature", float("nan"))) == float(LLM_CONFIG.get("temperature", 0.0)),
            "max_tokens": int(generation.get("max_tokens", -1)) == int(LLM_CONFIG.get("max_tokens", 120)),
        }
    except (TypeError, ValueError):
        return response, False, "malformed_provenance_value", provenance
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        return response, False, f"provenance_mismatch:{','.join(failed)}", provenance
    record_kind = str(provenance.get("record_kind", "llm_generation"))
    if record_kind == "deterministic_neutral_imputation":
        imputation_valid = (
            provenance.get("transport") == "deterministic_neutral_imputation"
            and provenance.get("imputation_policy") == "deterministic_neutral_imputation_v1"
            and response.get("impact") == "neutral"
            and isinstance(response.get("reason"), str)
            and bool(response.get("reason", "").strip())
        )
        if not imputation_valid:
            return response, False, "invalid_deterministic_imputation", provenance
    elif record_kind != "llm_generation":
        return response, False, f"unsupported_record_kind:{record_kind}", provenance
    return response, True, "ok", provenance


def reason_specificity(reason: str, ext_cols: list[str]) -> float:
    text = (reason or "").lower()
    if not text or text == "fallback":
        return 0.0
    hits = 0
    for col in ext_cols:
        if col.lower() in text:
            hits += 1
    for keyword in REASON_KEYWORDS:
        if keyword.lower() in text:
            hits += 1
    return min(1.0, hits / 2.0)


def validate_llm_output(llm_out: dict | None) -> tuple[bool, str]:
    """Check whether a cached/generated LLM semantic record is usable.

    Parameters:
        llm_out: Parsed JSON object loaded from cache or returned by the LLM.

    Returns:
        `(is_valid, reason)`.  Strict experiments should reject invalid records
        instead of silently treating them as neutral fallback semantics.
    """

    if isinstance(llm_out, dict) and isinstance(llm_out.get("response"), dict):
        llm_out = llm_out["response"]
    if not isinstance(llm_out, dict):
        return False, "not_a_json_object"
    impact = llm_out.get("impact")
    reason = llm_out.get("reason")
    if impact not in IMPACT_TO_VALUE:
        return False, "invalid_impact_label"
    if not isinstance(reason, str) or not reason.strip():
        return False, "empty_reason"
    if reason.strip().lower() == "fallback":
        return False, "fallback_reason"
    return True, "ok"


def external_trend_score(hist_items: list[dict], curr_items: dict, ext_cols: list[str]) -> float:
    signed_scores = []
    for col in ext_cols:
        try:
            hist_vals = np.asarray([float(item[col]) for item in hist_items], dtype=float)
            curr_val = float(curr_items[col])
        except (KeyError, TypeError, ValueError):
            continue
        if len(hist_vals) == 0:
            continue
        scale = np.std(hist_vals)
        if scale < 1e-6:
            scale = max(abs(float(np.mean(hist_vals))), 1.0)
        signed_scores.append((curr_val - float(np.mean(hist_vals))) / scale)
    if not signed_scores:
        return 0.0
    return float(np.tanh(np.mean(signed_scores)))


def trend_alignment(impact_value: float, trend_score: float) -> float:
    if impact_value == 0:
        return max(0.0, 1.0 - abs(trend_score))
    sign = 1.0 if impact_value > 0 else -1.0
    return float(np.clip(0.5 + 0.5 * sign * trend_score, 0.0, 1.0))


def legacy_target_aware_reliability(
    df: pd.DataFrame,
    target: str,
    ext_cols: list[str],
    semantic_indices: np.ndarray,
    future_targets: np.ndarray,
    train_slice: slice,
    semantic_frame: pd.DataFrame,
) -> tuple[np.ndarray, pd.DataFrame, dict[str, float]]:
    """仅用训练段估计外生变量方向，重算目标感知语义可靠性。

    入参：原始数据、目标/外生列、监督样本语义索引、未来目标、训练切片和 LLM 审计表。
    功能：估计每个外生变量与未来目标的训练段相关方向，以相关强度归一化加权局部趋势，
    再与 LLM 影响方向比较；验证、校准和测试标签均不参与权重估计。
    返回值：更新后的两维语义特征、审计表和外生变量相关权重。
    """

    if train_slice.start not in (None, 0) or train_slice.stop is None:
        raise ValueError("target-aware reliability requires an explicit leading training slice")
    if not ext_cols:
        features = semantic_frame[["impact_value", "semantic_reliability"]].to_numpy(dtype=np.float32)
        return features, semantic_frame, {}
    ext_values = df[ext_cols].to_numpy(dtype=float)
    train_indices = np.asarray(semantic_indices[train_slice], dtype=int)
    train_y = np.asarray(future_targets[train_slice], dtype=float).reshape(-1)
    correlations = []
    for column_index in range(len(ext_cols)):
        values = ext_values[train_indices, column_index]
        valid = np.isfinite(values) & np.isfinite(train_y)
        if valid.sum() < 3 or np.std(values[valid]) < 1e-12 or np.std(train_y[valid]) < 1e-12:
            correlations.append(0.0)
        else:
            correlations.append(float(np.corrcoef(values[valid], train_y[valid])[0, 1]))
    corr = np.nan_to_num(np.asarray(correlations, dtype=float), nan=0.0, posinf=0.0, neginf=0.0)
    magnitude = np.abs(corr)
    if magnitude.sum() < 1e-12:
        normalized = np.full(len(ext_cols), 1.0 / len(ext_cols), dtype=float)
        direction = np.ones(len(ext_cols), dtype=float)
    else:
        normalized = magnitude / magnitude.sum()
        direction = np.sign(corr)

    evidence_scores = []
    alignment_scores = []
    reliability_scores = []
    window = int(LLM_CONFIG.get("history_window", 5))
    for row_index, source_index in enumerate(np.asarray(semantic_indices, dtype=int)):
        start = max(0, source_index - window)
        history = ext_values[start:source_index]
        current = ext_values[source_index]
        if len(history) == 0:
            local_trend = np.zeros(len(ext_cols), dtype=float)
        else:
            center = np.nanmean(history, axis=0)
            scale = np.nanstd(history, axis=0)
            fallback_scale = np.maximum(np.abs(center), 1.0)
            scale = np.where(scale < 1e-6, fallback_scale, scale)
            local_trend = np.tanh(np.nan_to_num((current - center) / scale, nan=0.0))
        evidence = float(np.tanh(np.sum(normalized * direction * local_trend)))
        impact_value = float(semantic_frame.iloc[row_index]["impact_value"])
        alignment = trend_alignment(impact_value, evidence)
        specificity = float(semantic_frame.iloc[row_index].get("reason_specificity", 0.0))
        strength = float(semantic_frame.iloc[row_index].get("impact_strength", 0.5))
        reliability = 0.50 * alignment + 0.30 * specificity + 0.20 * strength
        evidence_scores.append(evidence)
        alignment_scores.append(alignment)
        reliability_scores.append(float(np.clip(reliability, 0.0, 1.0)))

    updated = semantic_frame.copy()
    updated["target_aware_trend_score"] = evidence_scores
    updated["trend_alignment"] = alignment_scores
    updated["semantic_reliability"] = reliability_scores
    weights = {
        column: float(normalized[index] * direction[index])
        for index, column in enumerate(ext_cols)
    }
    updated["target_correlation_weights"] = json.dumps(weights, ensure_ascii=False, sort_keys=True)
    features = updated[["impact_value", "semantic_reliability"]].to_numpy(dtype=np.float32)
    return features, updated, weights


def _expected_calibration_error(labels: np.ndarray, probabilities: np.ndarray, bins: int = 10) -> float:
    """计算二分类概率的等宽分箱期望校准误差。"""

    y = np.asarray(labels, dtype=float).reshape(-1)
    p = np.asarray(probabilities, dtype=float).reshape(-1)
    edges = np.linspace(0.0, 1.0, bins + 1)
    error = 0.0
    for left, right in zip(edges[:-1], edges[1:]):
        mask = (p >= left) & (p < right if right < 1.0 else p <= right)
        if np.any(mask):
            error += float(mask.mean()) * abs(float(y[mask].mean()) - float(p[mask].mean()))
    return float(error)


def _smoothed_prior(labels: np.ndarray) -> float:
    """使用 Beta(1,1) 平滑返回训练前缀中的正确率先验。"""

    values = np.asarray(labels, dtype=float).reshape(-1)
    return float((values.sum() + 1.0) / (len(values) + 2.0))


def _expanding_prior(labels: np.ndarray, gap: int) -> np.ndarray:
    """为训练样本生成不读取当前及未来标签的扩展窗口先验。"""

    values = np.asarray(labels, dtype=float).reshape(-1)
    out = np.full(len(values), 0.5, dtype=float)
    for index in range(len(values)):
        stop = max(0, index - int(gap))
        if stop:
            out[index] = _smoothed_prior(values[:stop])
    return out


def _origin_neutral_thresholds(
    target_values: np.ndarray,
    last_observed_indices: np.ndarray,
    scale: float,
    lookback: int = 24,
) -> np.ndarray:
    """仅用每个预测起点之前的目标变化估计样本级中性阈值。"""

    values = np.asarray(target_values, dtype=float).reshape(-1)
    origins = np.asarray(last_observed_indices, dtype=int).reshape(-1)
    thresholds = np.empty(len(origins), dtype=float)
    for row, origin in enumerate(origins):
        start = max(0, int(origin) - int(lookback))
        history = values[start:int(origin) + 1]
        differences = np.diff(history)
        local_scale = float(np.std(differences)) if len(differences) else 0.0
        if not np.isfinite(local_scale) or local_scale < 1e-8:
            local_scale = max(float(np.std(history)), 1e-8) if len(history) else 1e-8
        thresholds[row] = max(1e-8, float(scale) * local_scale)
    return thresholds


def _available_reliability_features(
    df: pd.DataFrame,
    target: str,
    ext_cols: list[str],
    semantic_indices: np.ndarray,
    last_observed_indices: np.ndarray,
    semantic_frame: pd.DataFrame,
) -> tuple[np.ndarray, list[str]]:
    """构造预测起点可见的语义、目标历史和外生状态特征。

    入参：原始数据、列定义、语义快照索引、最后目标观测索引和 LLM 审计表。
    功能：所有局部统计量只读取对应预测起点及其过去，不使用未来目标。
    返回：二维特征矩阵与稳定的特征名列表。
    """

    semantic_indices = np.asarray(semantic_indices, dtype=int)
    last_observed_indices = np.asarray(last_observed_indices, dtype=int)
    if len(semantic_indices) != len(semantic_frame) or len(last_observed_indices) != len(semantic_frame):
        raise ValueError("semantic indices, last-observed indices and semantic frame must align")
    target_values = df[target].to_numpy(dtype=float)
    ext_values = df[ext_cols].to_numpy(dtype=float) if ext_cols else np.empty((len(df), 0), dtype=float)
    window = int(LLM_CONFIG.get("history_window", 5))
    names = [
        "impact_value",
        "impact_is_neutral",
        "reason_specificity",
        "impact_strength",
        "external_trend_mean",
        "external_trend_abs_mean",
        "impact_external_alignment",
        "target_recent_slope",
        "target_recent_volatility",
        "impact_target_alignment",
        "external_missing_rate",
    ]
    names.extend(f"external_level_{column}" for column in ext_cols)
    names.extend(f"external_trend_{column}" for column in ext_cols)
    rows: list[list[float]] = []
    for row_number, (source_index, last_observed) in enumerate(
        zip(semantic_indices, last_observed_indices, strict=True)
    ):
        impact = float(semantic_frame.iloc[row_number].get("impact_value", 0.0))
        specificity = float(semantic_frame.iloc[row_number].get("reason_specificity", 0.0))
        strength = float(semantic_frame.iloc[row_number].get("impact_strength", 0.5))

        ext_start = max(0, int(source_index) - window)
        ext_history = ext_values[ext_start:int(source_index)]
        ext_current = ext_values[int(source_index)] if len(ext_cols) else np.empty(0, dtype=float)
        if len(ext_cols):
            center = np.nanmean(ext_history, axis=0) if len(ext_history) else np.zeros(len(ext_cols))
            scale = np.nanstd(ext_history, axis=0) if len(ext_history) else np.ones(len(ext_cols))
            scale = np.where(scale < 1e-6, np.maximum(np.abs(center), 1.0), scale)
            ext_level = np.nan_to_num((ext_current - center) / scale, nan=0.0, posinf=0.0, neginf=0.0)
            ext_trend = np.tanh(ext_level)
            missing_rate = float(np.mean(~np.isfinite(ext_current)))
        else:
            ext_level = np.empty(0, dtype=float)
            ext_trend = np.empty(0, dtype=float)
            missing_rate = 0.0
        trend_mean = float(np.mean(ext_trend)) if len(ext_trend) else 0.0
        trend_abs_mean = float(np.mean(np.abs(ext_trend))) if len(ext_trend) else 0.0

        target_start = max(0, int(last_observed) - window + 1)
        target_history = target_values[target_start:int(last_observed) + 1]
        if len(target_history) >= 2:
            target_scale = max(float(np.std(target_history)), 1e-6)
            target_slope = float(np.tanh((target_history[-1] - target_history[0]) / target_scale))
            target_volatility = float(np.std(np.diff(target_history)) / target_scale)
        else:
            target_slope = 0.0
            target_volatility = 0.0
        rows.append([
            impact,
            float(abs(impact) < 1e-12),
            specificity,
            strength,
            trend_mean,
            trend_abs_mean,
            trend_alignment(impact, trend_mean),
            target_slope,
            target_volatility,
            trend_alignment(impact, target_slope),
            missing_rate,
            *ext_level.tolist(),
            *ext_trend.tolist(),
        ])
    matrix = np.nan_to_num(np.asarray(rows, dtype=float), nan=0.0, posinf=0.0, neginf=0.0)
    return matrix, names


def outcome_calibrated_reliability(
    df: pd.DataFrame,
    target: str,
    ext_cols: list[str],
    semantic_indices: np.ndarray,
    target_indices: np.ndarray,
    future_targets: np.ndarray,
    train_slice: slice,
    semantic_frame: pd.DataFrame,
    forecast_horizon: int,
    n_splits: int = 5,
    minimum_train_samples: int = 200,
    minimum_class_samples: int = 30,
    neutral_scale: float = 0.05,
    neutral_lookback: int = 24,
) -> tuple[np.ndarray, pd.DataFrame, dict]:
    """估计 LLM 方向判断正确的训练段校准概率。

    训练样本的 rho 由带 H-1 gap 的扩展窗口交叉拟合产生；验证、校准和
    测试 rho 由全训练段拟合的 L2 逻辑回归产生。若样本、类别或 OOF Brier
    skill 不足，则退化为不读取当前及未来标签的扩展先验。
    """

    if train_slice.start not in (None, 0) or train_slice.stop is None:
        raise ValueError("outcome-calibrated reliability requires an explicit leading training slice")
    horizon = int(forecast_horizon)
    if horizon < 1:
        raise ValueError("forecast_horizon must be positive")
    semantic_indices = np.asarray(semantic_indices, dtype=int)
    target_indices = np.asarray(target_indices, dtype=int)
    future_targets = np.asarray(future_targets, dtype=float).reshape(-1)
    if not (len(semantic_indices) == len(target_indices) == len(future_targets) == len(semantic_frame)):
        raise ValueError("semantic indices, target indices, targets and semantic frame must align")
    last_observed_indices = target_indices - horizon
    if np.any(last_observed_indices < 0) or np.any(semantic_indices > last_observed_indices):
        raise RuntimeError("Reliability features include information after the forecast origin.")

    all_x, feature_names = _available_reliability_features(
        df, target, ext_cols, semantic_indices, last_observed_indices, semantic_frame
    )
    train_stop = int(train_slice.stop)
    x_train = all_x[:train_stop]
    impact_train = np.sign(semantic_frame.iloc[:train_stop]["impact_value"].to_numpy(dtype=float))
    baseline_train = df[target].to_numpy(dtype=float)[last_observed_indices[:train_stop]]
    delta_train = future_targets[:train_stop] - baseline_train
    neutral_thresholds = _origin_neutral_thresholds(
        df[target].to_numpy(dtype=float),
        last_observed_indices,
        scale=neutral_scale,
        lookback=neutral_lookback,
    )
    train_thresholds = neutral_thresholds[:train_stop]
    actual_direction = np.where(
        delta_train > train_thresholds,
        1.0,
        np.where(delta_train < -train_thresholds, -1.0, 0.0),
    )
    correctness = (impact_train == actual_direction).astype(int)
    gap = max(0, horizon - 1)
    fallback_train = _expanding_prior(correctness, gap=gap)
    full_prior = _smoothed_prior(correctness)
    rho_train = fallback_train.copy()
    oof_mask = np.zeros(train_stop, dtype=bool)
    oof_baseline = fallback_train.copy()
    folds: list[dict] = []
    class_counts = np.bincount(correctness, minlength=2)
    can_fit = train_stop >= minimum_train_samples and int(class_counts.min()) >= minimum_class_samples

    if can_fit:
        splitter = TimeSeriesSplit(n_splits=int(n_splits), gap=gap)
        for fold, (fit_indices, holdout_indices) in enumerate(splitter.split(x_train), start=1):
            if len(fit_indices) < minimum_train_samples or len(np.unique(correctness[fit_indices])) < 2:
                continue
            model = make_pipeline(
                StandardScaler(),
                LogisticRegression(C=1.0, solver="lbfgs", max_iter=1000, random_state=0),
            )
            model.fit(x_train[fit_indices], correctness[fit_indices])
            rho_train[holdout_indices] = model.predict_proba(x_train[holdout_indices])[:, 1]
            oof_baseline[holdout_indices] = _smoothed_prior(correctness[fit_indices])
            oof_mask[holdout_indices] = True
            folds.append({
                "fold": fold,
                "fit_start": int(fit_indices[0]),
                "fit_stop": int(fit_indices[-1]) + 1,
                "holdout_start": int(holdout_indices[0]),
                "holdout_stop": int(holdout_indices[-1]) + 1,
                "gap": gap,
            })

    brier = baseline_brier = brier_skill = auroc = logloss = ece = float("nan")
    if np.any(oof_mask):
        oof_y = correctness[oof_mask]
        oof_p = np.clip(rho_train[oof_mask], 0.05, 0.95)
        base_p = np.clip(oof_baseline[oof_mask], 0.05, 0.95)
        brier = float(brier_score_loss(oof_y, oof_p))
        baseline_brier = float(brier_score_loss(oof_y, base_p))
        brier_skill = float(1.0 - brier / baseline_brier) if baseline_brier > 0 else float("nan")
        auroc = float(roc_auc_score(oof_y, oof_p)) if len(np.unique(oof_y)) == 2 else float("nan")
        logloss = float(log_loss(oof_y, oof_p, labels=[0, 1]))
        ece = _expected_calibration_error(oof_y, oof_p)
    valid_fold_count = len(folds)
    oof_coverage = float(oof_mask.mean()) if train_stop else 0.0
    oof_class_counts = np.bincount(correctness[oof_mask], minlength=2) if np.any(oof_mask) else np.zeros(2, dtype=int)
    mode = "outcome_calibrated_logistic"
    if (
        not can_fit
        or valid_fold_count < 3
        or oof_coverage < 0.40
        or int(oof_class_counts.min()) < minimum_class_samples
        or not np.isfinite(brier_skill)
        or brier_skill <= 0.0
    ):
        mode = "expanding_prior_fallback"
        rho_train = fallback_train

    rho_all = np.full(len(semantic_frame), full_prior, dtype=float)
    rho_all[:train_stop] = rho_train
    grouping_rho = np.full(len(semantic_frame), full_prior, dtype=float)
    coefficients: dict[str, float] = {}
    if mode == "outcome_calibrated_logistic":
        final_model = make_pipeline(
            StandardScaler(),
            LogisticRegression(C=1.0, solver="lbfgs", max_iter=1000, random_state=0),
        )
        final_model.fit(x_train, correctness)
        grouping_rho = final_model.predict_proba(all_x)[:, 1]
        if train_stop < len(rho_all):
            rho_all[train_stop:] = grouping_rho[train_stop:]
        logistic = final_model.named_steps["logisticregression"]
        coefficients = {
            name: float(value)
            for name, value in zip(feature_names, logistic.coef_.reshape(-1), strict=True)
        }
    rho_all = np.clip(rho_all, 0.05, 0.95)
    grouping_rho = np.clip(grouping_rho, 0.05, 0.95)

    updated = semantic_frame.copy()
    updated["heuristic_reliability"] = updated["semantic_reliability"].astype(float)
    updated["semantic_reliability"] = rho_all
    updated["grouping_reliability"] = grouping_rho
    updated["reliability_source"] = "full_train_model" if mode == "outcome_calibrated_logistic" else "train_prior"
    if mode == "outcome_calibrated_logistic":
        updated.iloc[:train_stop, updated.columns.get_loc("reliability_source")] = "expanding_prior"
        updated.iloc[np.flatnonzero(oof_mask), updated.columns.get_loc("reliability_source")] = "expanding_time_crossfit"
    else:
        updated.iloc[:train_stop, updated.columns.get_loc("reliability_source")] = "expanding_prior"
    updated["semantic_direction_correct_train_only"] = np.nan
    updated.iloc[:train_stop, updated.columns.get_loc("semantic_direction_correct_train_only")] = correctness.astype(float)
    metadata = {
        "mode": mode,
        "definition": "P(LLM ternary direction equals future target direction | forecast-origin features)",
        "forecast_horizon": horizon,
        "neutral_threshold_definition": "neutral_scale_times_origin_only_recent_target_delta_std",
        "neutral_threshold_train_mean": float(np.mean(train_thresholds)),
        "neutral_threshold_train_median": float(np.median(train_thresholds)),
        "neutral_scale": float(neutral_scale),
        "neutral_lookback": int(neutral_lookback),
        "train_samples": train_stop,
        "class_counts": {"incorrect": int(class_counts[0]), "correct": int(class_counts[1])},
        "smoothed_train_prior": full_prior,
        "time_crossfit_splits": int(n_splits),
        "time_crossfit_gap": gap,
        "valid_time_crossfit_folds": valid_fold_count,
        "oof_samples": int(oof_mask.sum()),
        "oof_coverage": oof_coverage,
        "oof_class_counts": {"incorrect": int(oof_class_counts[0]), "correct": int(oof_class_counts[1])},
        "oof_brier": brier,
        "oof_baseline_brier": baseline_brier,
        "oof_brier_skill": brier_skill,
        "oof_auroc": auroc,
        "oof_log_loss": logloss,
        "oof_ece": ece,
        "probability_clip": [0.05, 0.95],
        "minimum_train_samples": int(minimum_train_samples),
        "minimum_class_samples": int(minimum_class_samples),
        "feature_names": feature_names,
        "standardized_logistic_coefficients": coefficients,
        "folds": folds,
    }
    updated["reliability_metadata"] = json.dumps(metadata, ensure_ascii=False, sort_keys=True)
    features = updated[["impact_value", "semantic_reliability"]].to_numpy(dtype=np.float32)
    return features, updated, metadata


def score_cached_output(llm_out: dict, hist_items: list[dict], curr_items: dict, ext_cols: list[str]) -> dict[str, float | str]:
    """把已验证的 LLM JSON 转换为语义内容与启发式可靠性审计字段。"""

    if isinstance(llm_out.get("response"), dict):
        llm_out = llm_out["response"]
    impact = llm_out.get("impact", "neutral")
    reason = llm_out.get("reason", "")
    impact_value = float(IMPACT_TO_VALUE.get(impact, 0.0))
    trend = external_trend_score(hist_items, curr_items, ext_cols)
    align = trend_alignment(impact_value, trend)
    specificity = reason_specificity(reason, ext_cols)
    clarity = 1.0 if impact in IMPACT_TO_VALUE else 0.0
    strength = 0.5 + 0.5 * min(abs(impact_value) / 2.0, 1.0)
    reliability = 0.42 * align + 0.33 * specificity + 0.15 * clarity + 0.10 * strength
    return {
        "impact": impact,
        "impact_value": impact_value,
        "reason_specificity": specificity,
        "trend_alignment": align,
        "impact_strength": strength,
        "semantic_reliability": float(np.clip(reliability, 0.0, 1.0)),
        "external_trend_score": trend,
        "cache_hit": 1.0,
    }


def build_semantic_matrix(
    df: pd.DataFrame,
    target: str,
    ext_cols: list[str],
    semantic_indices: np.ndarray,
    cache_policy: str = "fallback",
    max_llm_calls: int | None = None,
    allow_legacy_cache: bool = False,
    forecast_horizon: int = 1,
) -> tuple[np.ndarray, pd.DataFrame]:
    """构造样本级语义特征矩阵并记录每条语义的来源。

    入参：原始数据、目标列、外生列、样本索引、缓存策略、最大接口调用数和旧缓存许可。
    功能：`strict` 仅允许读取当前协议缓存；`generate` 可在缺失时访问 LLM；
    `fallback` 仅用于探索性实验。正式实验默认拒绝无法核验模型/提示词来源的旧缓存。
    返回值：由影响方向和可靠性组成的特征矩阵，以及逐样本审计表。
    """

    if cache_policy not in {"fallback", "generate", "strict"}:
        raise ValueError(f"Unsupported cache_policy: {cache_policy}")
    if forecast_horizon < 1:
        raise ValueError(f"forecast_horizon must be positive, got {forecast_horizon}")
    extractor = None
    llm_calls = 0
    rows = []
    features = []
    # 正式实验包含数十万条内容寻址记录；复用连接只优化读取开销，不改变缓存内容。
    cache_connection = _open_cache_db(LLM_CACHE_DIR)
    for index in semantic_indices:
        data_hashes, hist_items, curr_items = _cache_keys(
            df,
            target,
            ext_cols,
            int(index),
            forecast_horizon=forecast_horizon,
        )
        cached = None
        cache_variant = "miss"
        candidate_hashes = data_hashes if allow_legacy_cache and cache_policy != "strict" else data_hashes[:1]
        matched_hash = None
        cache_provenance: dict = {}
        protocol_reason = "cache_miss"
        expected_prompt_hash = _expected_prompt_hash(
            target,
            ext_cols,
            hist_items,
            curr_items,
            forecast_horizon,
        )
        for data_hash in candidate_hashes:
            record = _load_from_cache(
                LLM_CACHE_DIR,
                data_hash,
                return_record=True,
                connection=cache_connection,
            )
            if record is not None:
                if data_hash == data_hashes[0]:
                    response, protocol_valid, protocol_reason, cache_provenance = _validate_cache_record(
                        record,
                        forecast_horizon,
                        expected_prompt_hash,
                    )
                    if cache_policy == "strict" and not protocol_valid:
                        raise RuntimeError(
                            f"Invalid current-protocol LLM cache at source index {int(index)}: "
                            f"{protocol_reason}. Strict mode requires complete matching provenance."
                        )
                    cached = response
                    cache_variant = "current" if protocol_valid else "current_unverified"
                else:
                    cached = _load_from_cache(
                        LLM_CACHE_DIR,
                        data_hash,
                        connection=cache_connection,
                    )
                    cache_variant = "legacy"
                    protocol_reason = "legacy_cache_protocol"
                matched_hash = data_hash
                break
        if cached is None:
            if cache_policy == "strict":
                raise RuntimeError(
                    f"Missing current-protocol LLM cache at source index {int(index)}; "
                    "strict mode is cache-only. Run warmup_llm_cache.py first."
                )
            if cache_policy == "generate":
                if max_llm_calls is not None and llm_calls >= max_llm_calls:
                    raise RuntimeError(f"LLM cache miss budget exhausted at source index {int(index)}")
                if extractor is None:
                    extractor = LLMFeatureExtractor(
                        history_window=LLM_CONFIG["history_window"],
                        use_cache=True,
                        cache_dir=LLM_CACHE_DIR,
                    )
                is_ett = any(col in set(ETT_COLUMN_DESC) for col in ext_cols)
                cache_target_name = ETT_TARGET_DESC if is_ett else target
                target_name = TARGET_DESC.get(target, target)
                column_desc = ETT_COLUMN_DESC_EN if is_ett else DATASET_COLUMN_DESC.get(target, {})
                llm_out = extractor.extract(
                    hist_items,
                    curr_items,
                    target_name=target_name,
                    column_descriptions=column_desc,
                    cache_target_name=cache_target_name,
                    forecast_horizon=forecast_horizon,
                )
                llm_calls += 1
                valid, invalid_reason = validate_llm_output(llm_out)
                if not valid:
                    if cache_policy == "strict":
                        raise RuntimeError(
                            f"Invalid LLM generation for source index {int(index)}: {invalid_reason}; strict mode forbids neutral fallback."
                        )
                    item = {
                        "impact": "neutral",
                        "impact_value": 0.0,
                        "reason_specificity": 0.0,
                        "trend_alignment": 0.5,
                        "impact_strength": 0.5,
                        "semantic_reliability": 0.35,
                        "external_trend_score": external_trend_score(hist_items, curr_items, ext_cols),
                        "cache_hit": 0.0,
                        "llm_generated": 0.0,
                    }
                else:
                    item = score_cached_output(llm_out, hist_items, curr_items, ext_cols)
                    item["llm_generated"] = 1.0
                    item["cache_variant"] = "generated_current"
                    item["cache_hash"] = data_hashes[0]
                    item["cache_protocol"] = "generated_current"
            else:
                item = {
                    "impact": "neutral",
                    "impact_value": 0.0,
                    "reason_specificity": 0.0,
                    "trend_alignment": 0.5,
                    "impact_strength": 0.5,
                    "semantic_reliability": 0.35,
                    "external_trend_score": external_trend_score(hist_items, curr_items, ext_cols),
                    "cache_hit": 0.0,
                    "llm_generated": 0.0,
                }
        else:
            valid, invalid_reason = validate_llm_output(cached)
            if not valid and cache_policy == "strict":
                raise RuntimeError(
                    f"Invalid cached LLM output for source index {int(index)} ({cache_variant}): {invalid_reason}; strict mode requires usable JSON."
                )
            item = score_cached_output(cached, hist_items, curr_items, ext_cols)
            item["llm_generated"] = 0.0
            item["cache_variant"] = cache_variant
            item["cache_hash"] = matched_hash
            item["cache_protocol"] = protocol_reason
            if cache_provenance:
                item["cache_model"] = cache_provenance.get("model")
                item["cache_prompt_sha256"] = cache_provenance.get("prompt_sha256")
                item["cache_created_at"] = cache_provenance.get("created_at")
                item["cache_record_kind"] = cache_provenance.get("record_kind", "llm_generation")
                item["cache_transport"] = cache_provenance.get("transport", "")
        item["source_index"] = int(index)
        item["forecast_horizon"] = int(forecast_horizon)
        item["prompt_version"] = str(LLM_CONFIG.get("prompt_version", ""))
        item["schema_version"] = str(LLM_CONFIG.get("schema_version", ""))
        rows.append(item)
        features.append([float(item["impact_value"]), float(item["semantic_reliability"])])
    cache_connection.close()
    return np.asarray(features, dtype=np.float32), pd.DataFrame(rows)
