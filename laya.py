"""
title: Laya Decide
description: Structured classification, scoring and probability judgments via a Laya decision model.
version: 0.3.0
requirements: aiohttp
"""

import asyncio
import json
import math
import os
from typing import Any, Optional

import aiohttp
from pydantic import BaseModel, Field

# The model-facing name "probability" is mapped to the server's "noul" type.
# Common synonyms are accepted so small wording slips don't cause failures.
TYPE_ALIASES = {
    "choice": "choice",
    "classify": "choice",
    "classification": "choice",
    "label": "choice",
    "category": "choice",
    "score": "score",
    "scale": "score",
    "rating": "score",
    "ordinal": "score",
    "probability": "noul",
    "noul": "noul",
    "yes_no": "noul",
    "yesno": "noul",
    "boolean": "noul",
    "bool": "noul",
    "true_false": "noul",
}

EXAMPLE_QUESTIONS = {
    "sentiment": {
        "type": "choice",
        "instructions": "What is the customer's overall sentiment?",
        "criteria": {
            "positive": "satisfied or happy",
            "neutral": "no clear feeling",
            "negative": "unhappy or angry",
        },
    },
    "urgency": {
        "type": "score",
        "instructions": "How urgently does this need a reply?",
        "criteria": ["not urgent", "within a week", "within a day", "immediately"],
    },
    "refund": {
        "type": "probability",
        "instructions": "The customer is asking for a refund.",
    },
}


# ---------------------------------------------------------------------------
# Helpers live at module level so OpenWebUI does not expose them as tools.
# (OpenWebUI treats every public-ish method on Tools as a callable tool.)
# ---------------------------------------------------------------------------


def _error(message: str) -> str:
    return json.dumps(
        {
            "error": message,
            "what_to_do": (
                "Fix the call and call laya_decide again. Do not answer the "
                "decision yourself."
            ),
            "example_questions": EXAMPLE_QUESTIONS,
        },
        ensure_ascii=False,
    )


def _service_error(message: str) -> str:
    return json.dumps(
        {
            "error": f"Laya decision service error: {message}",
            "what_to_do": (
                "Tell the user the decision service failed. Do not make the "
                "decision yourself."
            ),
        },
        ensure_ascii=False,
    )


def _parse_questions(questions: Any) -> tuple[dict | None, str | None]:
    # Models often send nested objects as JSON strings.
    if isinstance(questions, str):
        try:
            questions = json.loads(questions)
        except json.JSONDecodeError as e:
            return None, f"questions is not valid JSON: {e}"

    # Accept a list of question objects with a "name" field.
    if isinstance(questions, list):
        converted: dict[str, Any] = {}
        for i, q in enumerate(questions):
            if not isinstance(q, dict):
                return None, f"questions[{i}] must be an object"
            name = str(q.get("name") or f"q{i + 1}")
            converted[name] = {k: v for k, v in q.items() if k != "name"}
        questions = converted

    if not isinstance(questions, dict) or not questions:
        return None, (
            "questions must be a non-empty object mapping a short name for "
            "each question to its definition"
        )

    return questions, None


def _normalize_question(
    name: str, q: Any, max_options: int = 20
) -> tuple[dict | None, str | None]:
    if not isinstance(q, dict):
        return None, f'"{name}" must be an object with type, instructions and criteria'

    raw_type = str(q.get("type", "")).strip().lower()
    qtype = TYPE_ALIASES.get(raw_type)
    if qtype is None:
        return None, (
            f'"{name}": type must be "choice", "score" or "probability" '
            f"(got {q.get('type')!r})"
        )

    instructions = q.get("instructions") or q.get("question") or q.get("prompt")
    if not isinstance(instructions, str) or not instructions.strip():
        return None, f'"{name}": instructions must be a non-empty string'
    instructions = instructions.strip()

    criteria = q.get("criteria")
    for alt in ("labels", "options", "levels"):
        if criteria is None:
            criteria = q.get(alt)

    if qtype in ("choice", "score") and isinstance(criteria, str):
        try:
            criteria = json.loads(criteria)
        except json.JSONDecodeError:
            pass

    if qtype == "choice":
        if isinstance(criteria, list) and all(isinstance(c, str) for c in criteria):
            criteria = {c: c for c in criteria}
        if not isinstance(criteria, dict) or len(criteria) < 2:
            return None, (
                f'"{name}": a choice question needs criteria as an object '
                "mapping at least two labels to descriptions"
            )
        if len(criteria) > max_options:
            return None, (
                f'"{name}": {len(criteria)} options is too many for one choice '
                f"question (limit {max_options}); accuracy drops sharply beyond "
                "that. Split it into two steps: a coarse choice between groups, "
                "then a separate call with the options inside the chosen group"
            )
        criteria = {str(k): str(v) for k, v in criteria.items()}

    elif qtype == "score":
        if not isinstance(criteria, list) or len(criteria) < 2:
            return None, (
                f'"{name}": a score question needs criteria as a list of at '
                "least two levels, lowest first"
            )
        criteria = [str(c) for c in criteria]

    else:  # noul / probability: criteria is optional extra guidance
        normalized = {"type": "noul", "instructions": instructions}
        if criteria not in (None, "", {}, []):
            normalized["criteria"] = criteria
        return normalized, None

    return {"type": qtype, "instructions": instructions, "criteria": criteria}, None


def _certainty(probabilities: list[float]) -> float | None:
    """1 - normalized Shannon entropy: 0 = uniform, 1 = all mass on one outcome.

    Normalized by ln(K) over all K options, so it is comparable across
    questions with different numbers of options. Matches Laya's own
    `confidence` for choice and score answers.
    """
    ps = [max(0.0, p) for p in probabilities]
    total = sum(ps)
    if len(ps) < 2 or total <= 0:
        return None
    entropy = -sum((p / total) * math.log(p / total) for p in ps if p > 0)
    return max(0.0, min(1.0, 1.0 - entropy / math.log(len(ps))))


def _summarize(
    answer: dict[str, Any], min_certainty: float, include_raw: bool
) -> dict[str, Any]:
    """Reduce a Laya answer to the fields the assistant should act on."""
    qtype = answer.get("type")
    result: dict[str, Any] = {}
    certainty: float | None = None

    probabilities = answer.get("probabilities")
    p_true = answer.get("noul", answer.get("probability"))

    if isinstance(probabilities, dict) and probabilities:
        numeric: dict[str, float] = {}
        for label, p in probabilities.items():
            try:
                numeric[str(label)] = float(p)
            except (TypeError, ValueError):
                continue
        if numeric:
            top = max(numeric, key=numeric.get)
            total = sum(numeric.values()) or 1.0
            if qtype == "score":
                result["type"] = "score"
                result["decision"] = int(top) if top.isdigit() else top
                legend = answer.get("legend")
                if isinstance(legend, dict) and top in legend:
                    result["decision_label"] = legend[top]
                # Laya's own expected level (probability-weighted mean).
                expected = answer.get("score")
                if expected is None and all(k.isdigit() for k in numeric):
                    total = sum(numeric.values()) or 1.0
                    expected = sum(int(k) * v for k, v in numeric.items()) / total
                try:
                    result["expected_level"] = round(float(expected), 4)
                except (TypeError, ValueError):
                    pass
            else:
                result["type"] = "choice"
                # Prefer the server's own pick; fall back to the top probability.
                result["decision"] = answer.get("choice", top)
            chosen = str(result["decision"])
            if chosen in numeric:
                result["decision_probability"] = round(numeric[chosen] / total, 4)
            result["probabilities"] = {k: round(v, 4) for k, v in numeric.items()}
            certainty = _certainty(list(numeric.values()))

    elif p_true is not None:
        try:
            p = max(0.0, min(1.0, float(p_true)))
            result["type"] = "probability"
            # Same shape as choice/score: the full distribution is only
            # kept for flagged questions (trimmed later otherwise).
            result["probabilities"] = {"yes": round(p, 4), "no": round(1.0 - p, 4)}
            result["decision"] = "yes" if p >= 0.5 else "no"
            result["decision_probability"] = round(p if p >= 0.5 else 1.0 - p, 4)
            # Binary entropy, on the same scale as choice/score answers.
            certainty = _certainty([p, 1.0 - p])
        except (TypeError, ValueError):
            pass

    if certainty is None:
        result["needs_review"] = True
        result["review_reason"] = "no usable probabilities were returned"
    else:
        result["certainty"] = round(certainty, 4)
        result["needs_review"] = certainty < min_certainty

    if include_raw:
        result["raw"] = answer

    return result


THRESHOLD_PRESETS = {"lenient": 0.05, "normal": 0.15, "strict": 0.30}


def _resolve_threshold(
    call_value: Any, user: Any, default: float
) -> tuple[float | None, str, str | None]:
    """Pick the review threshold: per-call > per-user valve > admin valve.

    Returns (threshold, source, error).
    """
    if call_value not in (None, ""):
        if isinstance(call_value, str):
            key = call_value.strip().lower()
            if key in THRESHOLD_PRESETS:
                return THRESHOLD_PRESETS[key], f"request ({key})", None
            try:
                call_value = float(key)
            except ValueError:
                return (
                    None,
                    "",
                    (
                        "min_certainty must be a number between 0 and 1 or one of "
                        "lenient, normal, strict"
                    ),
                )
        try:
            value = float(call_value)
        except (TypeError, ValueError):
            return None, "", "min_certainty must be a number between 0 and 1"
        if not 0.0 <= value <= 1.0:
            return None, "", "min_certainty must be between 0 and 1"
        return value, "request", None

    user_valves = (user or {}).get("valves") if isinstance(user, dict) else None
    if user_valves is not None:
        raw = (
            user_valves.get("MIN_CERTAINTY")
            if isinstance(user_valves, dict)
            else getattr(user_valves, "MIN_CERTAINTY", None)
        )
        try:
            value = float(raw)
            if 0.0 <= value <= 1.0:
                return value, "user setting", None
        except (TypeError, ValueError):
            pass

    return default, "default", None


def _parse_items(items: Any, max_items: int) -> tuple[dict | None, str | None]:
    """Normalize items to {id: text}. Accepts an object, a JSON string,
    a single plain string, or a list of strings / {"id", "text"} objects."""
    if isinstance(items, str):
        stripped = items.strip()
        if stripped.startswith(("{", "[")):
            try:
                items = json.loads(stripped)
            except json.JSONDecodeError as e:
                return None, f"items is not valid JSON: {e}"
        else:
            items = {"text": items}

    if isinstance(items, list):
        converted: dict[str, Any] = {}
        for i, entry in enumerate(items, start=1):
            if isinstance(entry, dict):
                item_id = str(entry.get("id") or i)
                converted[item_id] = entry.get("text")
            else:
                converted[str(i)] = entry
        items = converted

    if not isinstance(items, dict) or not items:
        return None, (
            "items must be a non-empty object mapping an ID for each text to "
            "the exact text"
        )
    if len(items) > max_items:
        return None, f"too many items ({len(items)}); the limit is {max_items} per call"

    clean: dict[str, Any] = {}
    for item_id, state in items.items():
        if isinstance(state, str):
            if not state.strip():
                return None, f'item "{item_id}" is empty'
        elif isinstance(state, (dict, list)):
            if not state:
                return None, f'item "{item_id}" is empty'
        else:
            return None, (f'item "{item_id}" must be text or a JSON object of fields')
        clean[str(item_id)] = state
    return clean, None


def _routed_model(data: dict[str, Any]) -> str | None:
    """The checkpoint the server says it used, if it reports one."""
    routing = data.get("routing")
    if isinstance(routing, dict):
        for key in ("model", "family", "checkpoint"):
            if isinstance(routing.get(key), str):
                return routing[key]
    for key in ("family", "model"):
        if isinstance(data.get(key), str):
            return data[key]
    return None


async def _call_laya(
    session: aiohttp.ClientSession,
    valves: Any,
    state: Any,
    questions: dict[str, Any],
) -> tuple[dict | None, str | None, str | None]:
    """One request to Laya. Returns (answers, routed_model, error)."""
    # No "model" field: the server routes each item to a checkpoint itself.
    body: dict[str, Any] = {"state": state, "questions": questions}
    try:
        async with session.post(valves.LAYA_URL, json=body) as response:
            text = await response.text()
            if response.status >= 400:
                return None, None, f"HTTP {response.status}. Server said: {text[:500]}"
        data = json.loads(text)
    except asyncio.TimeoutError:
        return None, None, f"request timed out after {valves.TIMEOUT} seconds"
    except aiohttp.ClientError as e:
        return None, None, f"request failed: {e}"
    except ValueError as e:
        return None, None, f"invalid JSON response: {e}"

    answers = data.get("answers") if isinstance(data, dict) else None
    if not isinstance(answers, dict):
        return None, None, "response did not contain an answers object"
    return answers, _routed_model(data), None


class Tools:
    class Valves(BaseModel):
        LAYA_URL: str = Field(
            default=os.environ.get("LAYA_URL", "http://llama.lan:8081/v1/systemone"),
            description="Laya endpoint URL",
        )
        LAYA_API_KEY: str = Field(
            default=os.environ.get("LAYA_API_KEY", ""),
            description="Bearer token (optional)",
        )
        TIMEOUT: int = Field(
            default=int(os.environ.get("LAYA_TIMEOUT", "120")),
            description="Request timeout in seconds",
        )
        MIN_CERTAINTY: float = Field(
            default=float(os.environ.get("LAYA_MIN_CERTAINTY", "0.15")),
            description=(
                "Flag results whose certainty (1 - normalized entropy) is below "
                "this. 0.15 is roughly a 72/28 yes/no split, or a top option "
                "near 50% in a four-way choice."
            ),
        )
        MAX_CHOICE_OPTIONS: int = Field(
            default=int(os.environ.get("LAYA_MAX_CHOICE_OPTIONS", "20")),
            description="Reject choice questions with more options than this",
        )
        MAX_ITEMS: int = Field(
            default=int(os.environ.get("LAYA_MAX_ITEMS", "100")),
            description="Maximum items per call",
        )
        SHOW_ALL_PROBABILITIES: bool = Field(
            default=False,
            description=(
                "Return probabilities for every question, not only those "
                "needing review"
            ),
        )
        INCLUDE_RAW: bool = Field(
            default=False,
            description="Also return the raw Laya answer (debugging; can confuse the model)",
        )

    class UserValves(BaseModel):
        MIN_CERTAINTY: float = Field(
            default=-1.0,
            description=(
                "Your review threshold (0-1): results with lower certainty are "
                "flagged. 0.05 lenient, 0.15 normal, 0.3 strict. -1 uses the "
                "admin default."
            ),
        )

    def __init__(self):
        self.valves = self.Valves()

    async def laya_decide(
        self,
        items: dict,
        questions: dict,
        min_certainty: Optional[float] = None,
        __user__: Optional[dict] = None,
    ) -> str:
        """
        Get structured judgments about one or more texts or JSON records from the Laya decision model: classify each into one label, rate it on an ordered scale, or estimate how likely a statement about it is true. Use this whenever the user wants text classified, routed, rated, scored, or judged yes/no. Put every text that needs the same questions into a single call; the tool evaluates them in parallel and returns them in the same order. Write the questions from the categories and scales the user gave, then call the tool. Do not work out the expected answers yourself, not even to sanity-check the results: the returned decisions are final and your own predictions are never used. When a text could fit more than one label, leave that to the model; do not resolve edge cases while writing criteria. Pass the texts straight into items without restating them first.

        Each question has a "type", "instructions" and "criteria":
        - "choice": pick exactly one label (at most 20; for more, choose a group first, then choose within it). criteria = object mapping each label to a short description of what belongs there. Describe every label by its own content, including catch-all labels such as "other" (for example "general questions, feedback and complaints"); never describe a label as not fitting the others.
        - "score": pick one level on an ordered scale. criteria = list of levels, lowest first. Returns the most likely level and the expected (average) level.
        - "probability": how likely the statement in instructions is true. Use this for every yes/no judgment. Make the statement one short claim; naming concrete examples in parentheses helps (e.g. "The customer threatens to churn (cancel, leave, or switch to a competitor)."), but do not add exceptions, and put separate conditions in separate questions. It takes no criteria.
        Keep every description short (a few words): Laya reads all of a question's options within a small token budget, so long descriptions get cut off.

        Example value for questions:
        {"sentiment": {"type": "choice", "instructions": "What is the customer's overall sentiment?", "criteria": {"positive": "satisfied or happy", "neutral": "no clear feeling", "negative": "unhappy or angry"}}, "urgency": {"type": "score", "instructions": "How urgently does this need a reply?", "criteria": ["not urgent", "within a week", "within a day", "immediately"]}, "refund": {"type": "probability", "instructions": "The customer is asking for a refund."}}

        :param items: Object mapping an ID for each item (such as a ticket number, or "text" if there is only one) to either the exact text, copied in full without the ID, or a JSON object of the item's fields.
        :param questions: Object mapping a short name for each question to its definition (type, instructions, criteria), as in the example. The same questions are applied to every item.
        :param min_certainty: Leave empty unless the user asks for a review threshold or for stricter/looser review. Then use their number (0-1), or "lenient" (0.05), "normal" (0.15) or "strict" (0.3).
        :return: JSON with the items in the order given, each with its decisions, certainty (0-1) and any questions needing human review, plus a list of the item IDs needing review, least certain first.
        """
        threshold, threshold_source, err = _resolve_threshold(
            min_certainty, __user__, self.valves.MIN_CERTAINTY
        )
        if err:
            return _error(err)

        parsed_items, err = _parse_items(items, self.valves.MAX_ITEMS)
        if err:
            return _error(err)

        parsed, err = _parse_questions(questions)
        if err:
            return _error(err)

        normalized: dict[str, Any] = {}
        for name, q in parsed.items():
            nq, err = _normalize_question(str(name), q, self.valves.MAX_CHOICE_OPTIONS)
            if err:
                return _error(err)
            normalized[str(name)] = nq

        headers = {"Content-Type": "application/json"}
        if self.valves.LAYA_API_KEY:
            headers["Authorization"] = f"Bearer {self.valves.LAYA_API_KEY}"

        async def evaluate(
            session: aiohttp.ClientSession, item_id: str, state: Any
        ) -> dict[str, Any]:
            answers, routed, call_err = await _call_laya(
                session, self.valves, state, normalized
            )
            if call_err:
                return {"id": item_id, "error": call_err}

            results: dict[str, Any] = {}
            review: list[str] = []
            certainties: list[float] = []
            for name, answer in answers.items():
                if isinstance(answer, dict):
                    summarized = _summarize(answer, threshold, self.valves.INCLUDE_RAW)
                else:
                    summarized = {"raw": answer, "needs_review": True}
                flagged = bool(summarized.pop("needs_review", False))
                if flagged:
                    review.append(name)
                    summarized["needs_review"] = True
                elif not self.valves.SHOW_ALL_PROBABILITIES:
                    summarized.pop("probabilities", None)
                results[name] = summarized
                certainties.append(summarized.get("certainty", 0.0))

            item: dict[str, Any] = {
                "id": item_id,
                "lowest_certainty": round(min(certainties), 4) if certainties else 0.0,
                "needs_review": review,
                "results": results,
            }
            if routed:
                item["model"] = routed
            return item

        # All items are sent at once; the Laya server queues them.
        # limit=0 removes aiohttp's default cap of 100 open connections.
        async with aiohttp.ClientSession(
            headers=headers,
            timeout=aiohttp.ClientTimeout(total=self.valves.TIMEOUT),
            connector=aiohttp.TCPConnector(limit=0),
        ) as session:
            evaluated = await asyncio.gather(
                *(
                    evaluate(session, item_id, state)
                    for item_id, state in parsed_items.items()
                )
            )

        if all("error" in e for e in evaluated):
            return _service_error(evaluated[0]["error"])

        # Items stay in the order given. Summaries point at the ones needing
        # attention without reordering anything.
        failed = [e["id"] for e in evaluated if "error" in e]
        review_order = [
            e["id"]
            for e in sorted(
                (e for e in evaluated if "error" not in e and e["needs_review"]),
                key=lambda e: e["lowest_certainty"],
            )
        ]

        output: dict[str, Any] = {
            "items": evaluated,
            "needs_review": review_order,
        }
        if failed:
            output["failed"] = failed
        output["review_threshold"] = {
            "min_certainty": threshold,
            "source": threshold_source,
        }
        output["how_to_report"] = (
            "These are the decision model's answers, with items in the order "
            "given. needs_review lists the IDs of items with at least one "
            "uncertain decision, least certain first; lowest_certainty on each "
            "item ranks all items if the user wants a confidence ordering. "
            "Report every decision as given, including ones that need review, "
            "and never substitute your own judgment. certainty is a 0-1 score "
            "of how decided the model is (0 = no idea, 1 = certain); it is not "
            "a probability, so never present it as a percentage. To give a "
            "percentage, use decision_probability (the probability of the "
            "decision shown). For each question marked needs_review, mark that "
            "decision as needing human review and show its probabilities (they "
            "are included only for those questions). Treat every other decision "
            "as settled, even if its probability looks low. If an item has an error, "
            "say it could not be evaluated."
        )

        return json.dumps(
            output,
            ensure_ascii=False,
        )
