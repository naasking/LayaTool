# Laya Decide — OpenWebUI tool

An [OpenWebUI](https://github.com/open-webui/open-webui) tool that lets a chat model hand structured decisions to a [Laya](https://laya.convaiinnovations.com/) decision model: classify text into one label, rate it on an ordered scale, or estimate how likely a statement about it is true. Laya is a Jev-compatible decision model.

The OpenWebUI chat model writes the questions once and passes in any number of texts. This tool sends the questions to your Laya server in parallel and returns calibrated probabilities, flagging the decisions that are too uncertain to state as fact.

Laya was run locally with all 3 model checkpoints, and it was tested against Qwen3.8-27B served locally as well, so you can have a fully local experience.

## Features

- **Three question types**: `choice` (pick one label), `score` (ordered scale), and `probability` (yes/no).
- **Batching**: many texts or JSON records in one call, evaluated concurrently, returned in input order.
- **Review flags**: every decision gets a `certainty` score (1 − normalized entropy, 0 to 1). Decisions below a threshold are marked `needs_review`, and a top-level list names the items to check, least certain first.
- **Readable output for the chat model**: each decision reports the probability of the answer shown. Full distributions appear only for flagged decisions, which keeps the response small.
- **Tolerant input**: accepts JSON-encoded strings, lists, and common synonyms (`rating`, `yes_no`, …) as well as the documented form, and returns errors that tell the model how to fix its call.
- **Automatic checkpoint routing**: no `model` field is sent, so a routing server picks the checkpoint per item (for example multilingual for non-English text). If the server reports its choice, each item shows it as `model`.

## Requirements

- OpenWebUI with tool support. Native function calling is recommended: set *Function Calling* to **Native** in the model's advanced parameters.
- A Laya server exposing the TypeSafe/Jev-compatible `POST /v1/systemone` endpoint, such as `laya serve` from [ggmlc](https://github.com/monatis/ggmlc) or `laya-serve` from the [Laya](https://github.com/NandhaKishorM/laya) package.
- `aiohttp`, which OpenWebUI already includes.

## Installation

1. In OpenWebUI, open **Workspace → Tools** and create a new tool.
2. Paste the contents of `laya.py` and save.
3. Open the tool's settings (valves) and set `LAYA_URL` to your server's `/v1/systemone` endpoint.
4. Enable the tool for the models that should use it.

## Configuration

Admin valves. Each one also reads the environment variable shown, if set.

| Valve | Env var | Default | Purpose |
|---|---|---|---|
| `LAYA_URL` | `LAYA_URL` | `http://llama.lan:8081/v1/systemone` | Laya endpoint |
| `LAYA_API_KEY` | `LAYA_API_KEY` | *(empty)* | Bearer token, if the server requires one |
| `TIMEOUT` | `LAYA_TIMEOUT` | `120` | Per-request timeout in seconds |
| `MIN_CERTAINTY` | `LAYA_MIN_CERTAINTY` | `0.15` | Review threshold (see below) |
| `MAX_CHOICE_OPTIONS` | `LAYA_MAX_CHOICE_OPTIONS` | `20` | Reject choice questions with more options |
| `MAX_ITEMS` | `LAYA_MAX_ITEMS` | `100` | Maximum items per call |
| `SHOW_ALL_PROBABILITIES` | | `false` | Return distributions for every decision, not just flagged ones |
| `INCLUDE_RAW` | | `false` | Also return the raw server answer (for debugging) |

Users can set their own `MIN_CERTAINTY` in the tool's user settings; `-1` means "use the admin default".

### Review threshold

`certainty` is 1 minus the normalized entropy of a decision's probability distribution: 0 means the model has no idea, and 1 means all probability is on one answer. It is comparable across questions with different numbers of options. A decision is flagged when its certainty is below the threshold.

The threshold is taken from the first of these that is set:

1. `min_certainty` in the call. The chat model sets this only when the user asks, using a number or the presets `lenient` (0.05), `normal` (0.15) or `strict` (0.3).
2. The user's `MIN_CERTAINTY` setting.
3. The admin `MIN_CERTAINTY` valve.

For a sense of scale, 0.15 corresponds roughly to a 72/28 yes/no split, or to a four-way choice where the winner has about half the probability.

## Usage

The tool exposes one function, `laya_decide(items, questions, min_certainty=None)`.

- `items`: an object mapping an ID to each text (or JSON record).
- `questions`: an object mapping a short name to each question. The same questions apply to every item.

Example call:

```json
{
  "items": {
    "T-1042": "I was charged $49 twice this month. Please refund the duplicate.",
    "T-1048": "How do I add my team members to the workspace?"
  },
  "questions": {
    "route": {
      "type": "choice",
      "instructions": "Route this ticket to exactly one team.",
      "criteria": {
        "billing": "charges, invoices, refunds, payments",
        "technical": "bugs, crashes, login, sync",
        "sales": "pricing, discounts, new features",
        "other": "how-to questions, feedback, general complaints"
      }
    },
    "urgency": {
      "type": "score",
      "instructions": "How urgently does this ticket need a response?",
      "criteria": ["0 = can wait", "1 = handle soon", "2 = blocking"]
    },
    "churn": {
      "type": "probability",
      "instructions": "The customer threatens to churn (cancel, leave, or switch to a competitor)."
    }
  }
}
```

Example output (one item shown):

```json
{
  "items": [
    {
      "id": "T-1048",
      "lowest_certainty": 0.1978,
      "needs_review": [],
      "results": {
        "route":   {"type": "choice", "decision": "other", "decision_probability": 0.5451, "certainty": 0.1978},
        "urgency": {"type": "score", "decision": 1, "decision_label": "1 = handle soon",
                    "expected_level": 1.3337, "decision_probability": 0.6357, "certainty": 0.3452},
        "churn":   {"type": "probability", "decision": "no", "decision_probability": 0.976, "certainty": 0.8367}
      },
      "model": "english"
    }
  ],
  "needs_review": [],
  "review_threshold": {"min_certainty": 0.15, "source": "default"},
  "how_to_report": "..."
}
```

Flagged decisions also include `"needs_review": true` and their full `probabilities`. Items that fail are listed in a top-level `failed` array and carry an `error` message instead of results.

## Writing good questions

These points are in the tool's description so the chat model follows them. They also apply if you write calls yourself.

- **Keep descriptions short.** Each option's description should be a few words. Laya reads all of a question's options within a small token budget, so long descriptions get cut off.
- **Describe every label by its own content**, including catch-alls such as `other`. A description like "anything that doesn't fit the others" pulls results toward whichever label is listed first.
- **Use at most 20 options per choice question.** For more, choose a group first, then choose within it.
- **Make each probability statement one short claim.** Concrete examples in parentheses help. Exceptions and extra conditions hurt, so put separate conditions in separate questions. Probability questions take no `criteria`, since Laya ignores them.
- **Use the same questions for the whole batch**, so that results can be compared across items.

It also helps to add a line to the chat model's system prompt such as *"When using laya_decide, write the questions from the user's categories and call it immediately."* Without it, some models spend many tokens predicting the answers before calling the tool.

## Limitations

- Laya base checkpoints are zero-shot and uncalibrated. Treat `certainty` as a relative ranking within a batch rather than an absolute guarantee. For reliable absolute levels, calibrate or fine-tune Laya on your own labeled data. Items with a low `lowest_certainty` are good candidates to label first.
- Results depend on question wording. Laya is deterministic, so the same questions always produce the same answers, but small changes to label descriptions can shift results noticeably.
- The typed-decisions checkpoint is tuned for specific workflows. With free-form questions, the general English checkpoint usually does better.
