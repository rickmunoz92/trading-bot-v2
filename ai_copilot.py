# ai_copilot.py

import os
import json
import math
from typing import TypedDict, Dict, Any, Optional
from datetime import datetime
from pathlib import Path

# -------------------- Robust .env loading --------------------
def _manual_load_env(env_path: Path) -> None:
    """
    Minimal .env parser: KEY=VALUE lines, ignores comments and blanks.
    Does not support export/quotes/escapes; good enough for OPENAI_API_KEY.
    """
    try:
        if not env_path.exists():
            return
        with env_path.open("r", encoding="utf-8") as f:
            for raw in f:
                line = raw.strip()
                if not line or line.startswith("#"):
                    continue
                if "=" not in line:
                    continue
                k, v = line.split("=", 1)
                k = k.strip()
                v = v.strip().strip("'").strip('"')
                if k and (k not in os.environ):
                    os.environ[k] = v
    except Exception:
        # best effort only
        pass

def _load_env_robust() -> None:
    # Try python-dotenv first
    env_candidates = []
    here = Path(__file__).resolve().parent
    env_candidates.append(here / ".env")
    # Also try cwd and its parent (useful if launched from project root)
    try:
        cwd = Path.cwd()
        env_candidates.append(cwd / ".env")
        env_candidates.append(cwd.parent / ".env")
    except Exception:
        pass

    loaded = False
    try:
        from dotenv import load_dotenv  # type: ignore
        for p in env_candidates:
            if p.exists():
                load_dotenv(dotenv_path=str(p), override=False)
                loaded = True
                break
        if not loaded:
            load_dotenv(override=False)  # fallback to default search
            loaded = True
    except Exception:
        # No python-dotenv; do manual
        for p in env_candidates:
            _manual_load_env(p)

_load_env_robust()

# -------------------- Optional third-party deps --------------------
# yaml is optional (we'll gracefully skip YAML config if missing)
try:
    import yaml  # type: ignore
    _HAS_YAML = True
except Exception:
    yaml = None  # type: ignore
    _HAS_YAML = False

# OpenAI client is optional
try:
    import openai  # type: ignore
    openai.api_key = os.getenv("OPENAI_API_KEY")
except Exception:
    openai = None  # type: ignore


# ----------------------- Config loading helpers -----------------------------

def load_yaml_config(path: str = "copilot.config.yaml") -> Dict[str, Any]:
    """
    Load defaults from copilot.config.yaml if PyYAML is available.
    If PyYAML is missing or the file doesn't exist, return {}.
    """
    if not os.path.exists(path):
        return {}
    if not _HAS_YAML:
        print("Note: PyYAML not installed; skipping copilot.config.yaml. "
              "Install with `pip install pyyaml` to use YAML config.")
        return {}
    try:
        with open(path, "r") as f:
            return yaml.safe_load(f) or {}
    except Exception as e:
        print(f"Failed to load config {path}: {e}")
        return {}


# ---------------------------- Runtime config --------------------------------

class CopilotConfig:
    def __init__(self, args: Any = None):
        cfg = load_yaml_config()
        # Default parameters (can be overridden by CLI args)
        self.confidence_threshold: float = cfg.get("confidence_threshold", 0.70)
        self.weights: Dict[str, float] = cfg.get(
            "weights",
            {"regime": 0.30, "events": 0.20, "sentiment": 0.15, "xasset": 0.15, "risk": 0.10, "llm": 0.10},
        )
        self.event_cap_minutes: int = cfg.get("event_cap", {}).get("minutes", 30)
        self.event_cap_value: float = cfg.get("event_cap", {}).get("cap", 0.40)
        self.learning_enabled: bool = cfg.get("learning", {}).get("enabled", True)
        self.min_trades: int = cfg.get("learning", {}).get("min_trades", 50)
        self.retrain_every: int = cfg.get("learning", {}).get("retrain_every", 25)
        self.debug: bool = cfg.get("logging", {}).get("debug", False)
        # Default model set to a widely available chat model; can be overridden in YAML
        self.gpt_model: str = cfg.get("gpt", {}).get("model", "gpt-4o-mini")
        self.gpt_max_tokens: int = cfg.get("gpt", {}).get("max_tokens", 500)

        # Override with CLI args if provided
        if args:
            if getattr(args, "confidence_threshold", None) is not None:
                self.confidence_threshold = float(args.confidence_threshold)
            if getattr(args, "weights", None):
                try:
                    w: Dict[str, float] = {}
                    for kv in str(args.weights).split(","):
                        k, v = kv.split("=")
                        w[k.strip()] = float(v.strip())
                    self.weights = w
                except Exception:
                    print(f"Invalid weights format: {args.weights}. Using defaults.")
            if getattr(args, "debug", False):
                self.debug = True


# ---------------------------- Decision schema --------------------------------

class Decision(TypedDict):
    approved: bool
    confidence: float
    reasoning: str
    layman_explanation: str
    adjusted_position_size: Optional[float]
    filter_breakdown: Dict[str, Dict[str, float]]


# ------------------------------ Persistence ----------------------------------

MEMORY_FILE = "copilot_memory.json"

def reset_memory() -> None:
    """Reset stored learning data."""
    if os.path.exists(MEMORY_FILE):
        os.remove(MEMORY_FILE)
    print("AI Co-Pilot memory reset.")

def load_memory() -> Dict[str, Any]:
    if os.path.exists(MEMORY_FILE):
        try:
            with open(MEMORY_FILE, "r") as f:
                return json.load(f)
        except Exception:
            return {}
    return {}

def save_memory(mem: Dict[str, Any]) -> None:
    try:
        with open(MEMORY_FILE, "w") as f:
            json.dump(mem, f, indent=2)
    except Exception as e:
        print(f"Failed to save memory: {e}")


# ------------------------------- Math utils ----------------------------------

def logistic(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))

def clip01(x: float) -> float:
    return max(0.0, min(1.0, x))


# ------------------------------- Evaluator -----------------------------------

def evaluate(signal: Dict[str, Any], context: Dict[str, Any], config: CopilotConfig) -> Decision:
    """
    Evaluate a trading signal using a weighted multi-factor confidence model.
    Returns a Decision with approval, confidence, breakdown, and explanations.
    """
    decision: Decision = {
        "approved": True,
        "confidence": 0.0,
        "reasoning": "",
        "layman_explanation": "",
        "adjusted_position_size": None,
        "filter_breakdown": {},
    }

    symbol = context.get("symbol", "")
    timeframe = context.get("timeframe", "")
    _ = context.get("bars", [])  # recent bars (each with 'c','h','l','o', etc.)
    side = signal.get("action", "hold")

    # -- Market Regime Detection (placeholders; replace with real computations) --
    ema_alignment = 0.5    # TODO
    adx_norm = 0.5         # TODO
    macd_hist_norm = 0.5   # TODO
    ema_stack = 0.5        # TODO
    trend_score = 0.40 * ema_alignment + 0.30 * adx_norm + 0.20 * abs(macd_hist_norm) + 0.10 * ema_stack

    bb_squeeze = 0.5       # TODO
    chop_norm = 0.5        # TODO
    rsi_mid = 0.5          # TODO
    range_score = 0.45 * bb_squeeze + 0.35 * chop_norm + 0.20 * rsi_mid

    atr_expansion = 0.5    # TODO
    vol_ratio_norm = 0.5   # TODO
    vol_score = 0.60 * atr_expansion + 0.40 * vol_ratio_norm

    # Composite regime index
    regime_index = clip01(0.6 * trend_score + 0.2 * (1 - range_score) + 0.2 * vol_score)
    r_regime = regime_index
    decision["filter_breakdown"]["regime"] = {
        "score": r_regime,
        "weight": config.weights.get("regime", 0.0),
        "contrib": r_regime * config.weights.get("regime", 0.0),
    }

    # -- Event Awareness (r_events) --
    r_events = 1.0
    now = datetime.utcnow()
    for ev in context.get("events", []):
        ev_time = ev.get("time")
        if isinstance(ev_time, str):
            try:
                ev_time = datetime.fromisoformat(ev_time)
            except Exception:
                continue
        if isinstance(ev_time, datetime):
            if abs((ev_time - now).total_seconds()) <= config.event_cap_minutes * 60:
                r_events = 0.2
                break
    decision["filter_breakdown"]["events"] = {
        "score": r_events,
        "weight": config.weights.get("events", 0.0),
        "contrib": r_events * config.weights.get("events", 0.0),
    }

    # -- Sentiment/Flow (r_sent) --
    r_sent = 0.8  # TODO
    decision["filter_breakdown"]["sentiment"] = {
        "score": r_sent,
        "weight": config.weights.get("sentiment", 0.0),
        "contrib": r_sent * config.weights.get("sentiment", 0.0),
    }

    # -- Cross-Asset Confirmation (r_xasset) --
    r_xasset = 0.7  # TODO
    decision["filter_breakdown"]["xasset"] = {
        "score": r_xasset,
        "weight": config.weights.get("xasset", 0.0),
        "contrib": r_xasset * config.weights.get("xasset", 0.0),
    }

    # -- Risk Alignment (r_risk) --
    r_risk = 0.9  # TODO
    decision["filter_breakdown"]["risk"] = {
        "score": r_risk,
        "weight": config.weights.get("risk", 0.0),
        "contrib": r_risk * config.weights.get("risk", 0.0),
    }

    # -- LLM/News Reasoning (r_llm) --
    r_llm = 0.5
    tech_explanation = ""
    layman_expl = ""
    # Ensure openai is configured (module import + API key)
    has_openai = False
    try:
        if "openai" in globals() and openai is not None:
            if getattr(openai, "api_key", None) or os.getenv("OPENAI_API_KEY"):
                has_openai = True
    except Exception:
        has_openai = False

    if has_openai:
        try:
            prompt = (
                f"Trading signal for {symbol} ({timeframe}): action={side}. "
                "Given recent context, do you approve this trade? Explain shortly."
            )
            # Using Completions API for simplicity.
            resp = openai.Completion.create(model=getattr(config, "gpt_model", "gpt-4o-mini"),
                                            prompt=prompt,
                                            max_tokens=int(getattr(config, "gpt_max_tokens", 400)))
            text = resp.choices[0].text.strip()
            text_lower = text.lower()
            if any(neg in text_lower for neg in ["no", "avoid", "skip", "reject"]):
                r_llm = 0.1
            else:
                r_llm = 0.9
            tech_explanation = text
            layman_expl = text
        except Exception:
            tech_explanation = "LLM call failed or unavailable."
            layman_expl = tech_explanation
    else:
        tech_explanation = "OpenAI API not configured."
        layman_expl = tech_explanation
    decision["filter_breakdown"]["llm"] = {
        "score": r_llm,
        "weight": config.weights.get("llm", 0.0),
        "contrib": r_llm * config.weights.get("llm", 0.0),
    }

    # -- Composite Score and Confidence --
    S = sum(v["contrib"] for v in decision["filter_breakdown"].values())
    confidence = logistic(4 * S - 2)  # logistic calibration
    # Apply event cap if triggered
    if r_events < 1.0:
        confidence = min(confidence, config.event_cap_value)
    confidence = clip01(confidence)
    decision["confidence"] = confidence
    decision["approved"] = (confidence >= config.confidence_threshold)

    # Explanations
    decision["reasoning"] = tech_explanation or f"Regime score {r_regime:.2f}"
    if layman_expl:
        decision["layman_explanation"] = layman_expl
    else:
        decision["layman_explanation"] = (
            f"Overall confidence {confidence:.2f}. "
            f"Trade will be {'allowed' if decision['approved'] else 'blocked'}."
        )

    # -- Learning (storing features/outcome) --
    if getattr(config, "learning_enabled", True):
        memory = load_memory()
        trades = memory.get("trades", [])
        outcome = 1 if decision["approved"] else 0
        features = [r_regime, r_events, r_sent, r_xasset, r_risk, r_llm]
        trades.append({"features": features, "outcome": outcome, "time": datetime.utcnow().isoformat()})
        memory["trades"] = trades
        save_memory(memory)
        if len(trades) >= getattr(config, "min_trades", 50) and len(trades) % getattr(config, "retrain_every", 25) == 0:
            # TODO: retrain logistic regression on memory to adjust weights/calibration
            pass

    return decision
