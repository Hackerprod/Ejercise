"""Deterministically build the pre-registered L0/L1 instrument bank."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import random
from typing import Any


FIXED_GENERATOR_SEED = 20260929
L0_TEMPLATE_VERSION = "L0_LOCAL_LANGUAGE_SANITY_v1"
L1_TEMPLATE_VERSION = "L1_CONTEXTUAL_TWIN_PAIRS_v1"
LEXICON_VERSION = "EVERYDAY_ENGLISH_LEXICONS_v1"
CATEGORY_ORDER = (
    "syntax_agreement",
    "semantic_selection",
    "local_reference",
    "immediate_plausibility_completion",
)
L1_CATEGORY_ORDER = (
    "entity_attribute_binding",
    "temporal_ordering",
    "simple_causal_consequence",
    "discourse_situation_consistency",
)


def _canonical_bytes(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8")


def canonical_hash(value: Any) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _prompt_candidate(tokenizer: Any, prompt_text: str, continuation_text: str) -> tuple[list[int], list[int], list[int]]:
    prompt_ids = list(tokenizer.encode(prompt_text, add_special_tokens=False))
    joined_ids = list(tokenizer.encode(prompt_text + continuation_text, add_special_tokens=False))
    if joined_ids[: len(prompt_ids)] != prompt_ids:
        raise ValueError("prompt/candidate BPE boundary is not stable; revise the versioned template")
    continuation_ids = joined_ids[len(prompt_ids) :]
    if not continuation_ids:
        raise ValueError("instrument continuation tokenizes to an empty sequence")
    return prompt_ids, continuation_ids, joined_ids


def _l0_specs(seed: int, attempt: int) -> list[dict[str, Any]]:
    rng = random.Random(seed + attempt * 104729)
    names = ["Mara", "Nina", "Lena", "Maya", "Rosa", "Tara", "Iris", "June", "Omar", "Theo", "Liam", "Noah", "Evan", "Milo", "Ada", "Leah"]
    objects = ["mug", "book", "basket", "coat", "lamp", "bottle", "towel", "cup", "scarf", "box", "blanket", "key", "plate", "bag", "pencil", "chair"]
    foods = ["toast", "rice", "bread", "soup", "fruit", "pasta", "cheese", "salad", "oatmeal", "beans", "yogurt", "potatoes", "crackers", "noodles", "corn", "apples"]
    places = ["porch", "garden", "kitchen", "hallway", "yard", "bedroom", "garage", "office", "stairs", "patio", "shop", "classroom", "park", "dining room", "workshop", "entryway"]
    nouns = ["dog", "bird", "cat", "horse", "rabbit", "neighbor", "baker", "driver", "teacher", "sister", "brother", "worker", "gardener", "friend", "visitor", "duck"]
    rng.shuffle(names)
    rng.shuffle(objects)
    rng.shuffle(foods)
    rng.shuffle(places)
    rng.shuffle(nouns)

    rows: list[dict[str, Any]] = []
    for index in range(16):
        is_plural = index >= 8
        noun = nouns[index % 8]
        subject = f"the {['small', 'quiet', 'brown', 'young'][index % 4]} {noun}{'s' if is_plural else ''}"
        correct, foil = ("wait.", "waits.") if is_plural else ("waits.", "wait.")
        prompt = f"At the end of the day, {names[index]} noticed that {subject} by the gate"
        rows.append({"category": "syntax_agreement", "prompt_text": prompt, "preferred_text": " " + correct, "foil_text": " " + foil, "decisive_text": subject})

    for index in range(16):
        food = foods[index]
        name = names[(index + 4) % len(names)]
        if index % 4 == 0:
            prompt = f"At breakfast, {name} spread butter on the"
            preferred, foil = " toast.", " pillow."
        elif index % 4 == 1:
            prompt = f"At lunch, {name} stirred the {food} with a"
            preferred, foil = " spoon.", " sock."
        elif index % 4 == 2:
            prompt = f"Before eating, {name} put the warm {food} on a"
            preferred, foil = " plate.", " window."
        else:
            prompt = f"At the table, {name} cut the {food} with a"
            preferred, foil = " knife.", " blanket."
        rows.append({"category": "semantic_selection", "prompt_text": prompt, "preferred_text": preferred, "foil_text": foil, "decisive_text": prompt})

    for index in range(16):
        name = names[index]
        object_name = objects[index]
        prompt = f"{name} placed the {object_name} on the table. Later, {name} picked"
        rows.append({"category": "local_reference", "prompt_text": prompt, "preferred_text": " it up.", "foil_text": " them up.", "decisive_text": f"the {object_name}"})

    for index in range(16):
        name = names[(index + 8) % len(names)]
        place = places[index]
        prompt = f"Rain started while {name} was leaving the {place}. {name} opened the"
        rows.append({"category": "immediate_plausibility_completion", "prompt_text": prompt, "preferred_text": " umbrella.", "foil_text": " sandwich.", "decisive_text": "Rain started"})

    if len(rows) != 64 or [row["category"] for row in rows].count("syntax_agreement") != 16:
        raise AssertionError("L0 generator did not produce exactly 64 minimal pairs/16 per category")
    for index, row in enumerate(rows):
        row["pair_id"] = f"L0-{row['category']}-{index % 16 + 1:02d}"
    return rows


def _l1_specs(seed: int, attempt: int) -> list[dict[str, Any]]:
    rng = random.Random(seed + 7919 + attempt * 65537)
    names = ["Mara", "Nina", "Lena", "Maya", "Rosa", "Tara", "Iris", "June", "Omar", "Theo", "Liam", "Noah", "Evan", "Milo", "Ada", "Leah"]
    objects = ["scarf", "mug", "notebook", "hat", "towel", "bag", "cup", "shirt", "lamp", "key", "book", "box", "coat", "bottle", "blanket", "basket"]
    color_pairs = [("red", "blue"), ("green", "yellow"), ("black", "white"), ("pink", "orange"), ("purple", "brown"), ("gray", "gold"), ("silver", "teal"), ("navy", "cream")]
    action_pairs = [
        ("washed the cups", "swept the floor", "washing the cups", "sweeping the floor"),
        ("folded the towels", "watered the plants", "folding the towels", "watering the plants"),
        ("packed the lunch", "closed the window", "packing the lunch", "closing the window"),
        ("shelved the books", "wiped the table", "shelving the books", "wiping the table"),
        ("fed the cat", "filled the kettle", "feeding the cat", "filling the kettle"),
        ("brushed the dog", "locked the door", "brushing the dog", "locking the door"),
        ("set the plates", "chopped the vegetables", "setting the plates", "chopping the vegetables"),
        ("hung the coat", "put away the shoes", "hanging the coat", "putting away the shoes"),
    ]
    causal_pairs = [
        ("freezer", "sunny windowsill", "cold", "warm"),
        ("ice bucket", "warm shelf", "cold", "warm"),
        ("cool cellar", "heater", "cold", "warm"),
        ("snowy porch", "sunlit room", "cold", "warm"),
    ]
    rng.shuffle(names)
    rng.shuffle(objects)
    rng.shuffle(color_pairs)
    rng.shuffle(action_pairs)
    rng.shuffle(causal_pairs)

    rows: list[dict[str, Any]] = []
    # Entity/attribute: the twin swaps which person packed which color.
    for index in range(16):
        name_a = names[index]
        name_b = names[(index + 8) % len(names)]
        object_name = objects[index]
        color_a, color_b = color_pairs[index % len(color_pairs)]
        prefix_a = f"{name_a} packed the {color_a} {object_name}. {name_b} packed the {color_b} {object_name}. At the station, {name_a} searched the bag for the item {name_a} packed."
        prefix_b = f"{name_a} packed the {color_b} {object_name}. {name_b} packed the {color_a} {object_name}. At the station, {name_a} searched the bag for the item {name_a} packed."
        suffix = f"The color of {name_a}'s {object_name} was:"
        rows.append({
            "category": "entity_attribute_binding",
            "prompt_a_text": prefix_a + " " + suffix,
            "prompt_b_text": prefix_b + " " + suffix,
            "preferred_a_text": f" {color_a}.",
            "foil_a_text": f" {color_b}.",
            "decisive_a_text": prefix_a,
            "decisive_b_text": prefix_b,
        })

    # Temporal ordering: the second explicitly ordered task is last.
    for index in range(16):
        name_a = names[(index + 1) % len(names)]
        name_b = names[(index + 9) % len(names)]
        first, second, first_ing, second_ing = action_pairs[index % len(action_pairs)]
        prefix_a = f"{name_a} {first} before {name_b} {second}. They finished both tasks before dinner."
        prefix_b = f"{name_a} {second} before {name_b} {first}. They finished both tasks before dinner."
        suffix = "The last task completed was:"
        rows.append({
            "category": "temporal_ordering",
            "prompt_a_text": prefix_a + " " + suffix,
            "prompt_b_text": prefix_b + " " + suffix,
            "preferred_a_text": f" {second_ing}.",
            "foil_a_text": f" {first_ing}.",
            "decisive_a_text": prefix_a,
            "decisive_b_text": prefix_b,
        })

    # Simple causal consequence: opposite temperature exposure gives opposite state.
    for index in range(16):
        object_name = objects[(index + 3) % len(objects)]
        cold_place, warm_place, cold_word, warm_word = causal_pairs[index % len(causal_pairs)]
        prefix_a = f"The {object_name} was left in the {cold_place} for an hour. Then it was moved to the table."
        prefix_b = f"The {object_name} was left near the {warm_place} for an hour. Then it was moved to the table."
        suffix = f"At the end, the {object_name} felt:"
        rows.append({
            "category": "simple_causal_consequence",
            "prompt_a_text": prefix_a + " " + suffix,
            "prompt_b_text": prefix_b + " " + suffix,
            "preferred_a_text": f" {cold_word}.",
            "foil_a_text": f" {warm_word}.",
            "decisive_a_text": prefix_a,
            "decisive_b_text": prefix_b,
        })

    # Discourse/situation: track transfer and the resulting holder.
    for index in range(16):
        giver = names[(index + 2) % len(names)]
        receiver = names[(index + 10) % len(names)]
        prefix_a = f"{giver} handed an umbrella to {receiver} before walking to the car. {receiver} stayed by the door holding it while {giver} left."
        prefix_b = f"{receiver} handed an umbrella to {giver} before walking to the car. {giver} stayed by the door holding it while {receiver} left."
        suffix = "The person holding the umbrella was:"
        rows.append({
            "category": "discourse_situation_consistency",
            "prompt_a_text": prefix_a + " " + suffix,
            "prompt_b_text": prefix_b + " " + suffix,
            "preferred_a_text": f" {receiver}.",
            "foil_a_text": f" {giver}.",
            "decisive_a_text": prefix_a,
            "decisive_b_text": prefix_b,
        })

    if len(rows) != 64:
        raise AssertionError("L1 generator did not produce exactly 64 twin pairs")
    for category in L1_CATEGORY_ORDER:
        if sum(row["category"] == category for row in rows) != 16:
            raise AssertionError(f"L1 generator category count is not 16: {category}")
    for index, row in enumerate(rows):
        row["twin_id"] = f"L1-{row['category']}-{index % 16 + 1:02d}"
    return rows


def build_candidate_bank(tokenizer: Any, *, attempt: int = 0, seed: int = FIXED_GENERATOR_SEED) -> dict[str, Any]:
    if seed != FIXED_GENERATOR_SEED:
        raise ValueError("instrument generator seed is frozen to 20260929")
    l0: list[dict[str, Any]] = []
    for spec in _l0_specs(seed, attempt):
        prompt_ids, preferred_ids, preferred_joined = _prompt_candidate(tokenizer, spec["prompt_text"], spec["preferred_text"])
        foil_prompt_ids, foil_ids, foil_joined = _prompt_candidate(tokenizer, spec["prompt_text"], spec["foil_text"])
        if prompt_ids != foil_prompt_ids:
            raise ValueError("L0 candidate tokenization changed its prompt prefix")
        l0.append(
            {
                **spec,
                "prompt_token_ids": prompt_ids,
                "preferred_token_ids": preferred_ids,
                "foil_token_ids": foil_ids,
                "preferred_prompt_continuation_token_ids": preferred_joined,
                "foil_prompt_continuation_token_ids": foil_joined,
            }
        )

    l1: list[dict[str, Any]] = []
    for spec in _l1_specs(seed, attempt):
        prompt_a_ids, preferred_a_ids, preferred_a_joined = _prompt_candidate(tokenizer, spec["prompt_a_text"], spec["preferred_a_text"])
        prompt_a_foil_ids, foil_a_ids, foil_a_joined = _prompt_candidate(tokenizer, spec["prompt_a_text"], spec["foil_a_text"])
        prompt_b_ids, preferred_b_ids, preferred_b_joined = _prompt_candidate(tokenizer, spec["prompt_b_text"], spec["foil_a_text"])
        prompt_b_foil_ids, foil_b_ids, foil_b_joined = _prompt_candidate(tokenizer, spec["prompt_b_text"], spec["preferred_a_text"])
        if prompt_a_ids != prompt_a_foil_ids or prompt_b_ids != prompt_b_foil_ids:
            raise ValueError(f"L1 twin candidate boundary tokenization mismatch: {spec['twin_id']}")
        decisive_a_ids = list(tokenizer.encode(spec["decisive_a_text"], add_special_tokens=False))
        decisive_b_ids = list(tokenizer.encode(spec["decisive_b_text"], add_special_tokens=False))
        if prompt_a_ids[: len(decisive_a_ids)] != decisive_a_ids or prompt_b_ids[: len(decisive_b_ids)] != decisive_b_ids:
            raise ValueError(f"L1 decisive-information span is not a stable prompt prefix: {spec['twin_id']}")
        l1.append(
            {
                **spec,
                "prompt_a_token_ids": prompt_a_ids,
                "prompt_b_token_ids": prompt_b_ids,
                "decisive_a_token_ids": decisive_a_ids,
                "decisive_b_token_ids": decisive_b_ids,
                "prompt_a_last5_token_ids": prompt_a_ids[-5:],
                "prompt_b_last5_token_ids": prompt_b_ids[-5:],
                "preferred_a_token_ids": preferred_a_ids,
                "foil_a_token_ids": foil_a_ids,
                "preferred_b_token_ids": preferred_b_ids,
                "foil_b_token_ids": foil_b_ids,
                "preferred_a_prompt_continuation_token_ids": preferred_a_joined,
                "foil_a_prompt_continuation_token_ids": foil_a_joined,
                "preferred_b_prompt_continuation_token_ids": preferred_b_joined,
                "foil_b_prompt_continuation_token_ids": foil_b_joined,
            }
        )
    bank = {
        "schema": "omega-minimum-language-competence-l0-l1-instrument-bank-v1",
        "status": "CANDIDATE_UNSEALED",
        "fixed_generator_seed": seed,
        "deterministic_generation_attempt": attempt,
        "templates_version": {"L0": L0_TEMPLATE_VERSION, "L1": L1_TEMPLATE_VERSION},
        "lexicons_version": LEXICON_VERSION,
        "canonical_order": {"L0": list(CATEGORY_ORDER), "L1": list(L1_CATEGORY_ORDER), "within_category": "pair_id ascending"},
        "L0_minimal_pairs": l0,
        "L1_twin_pairs": l1,
    }
    return bank


def validate_l1_twins(bank: dict[str, Any]) -> list[dict[str, Any]]:
    checks: list[dict[str, Any]] = []
    for row in bank["L1_twin_pairs"]:
        prompt_a = row["prompt_a_token_ids"]
        prompt_b = row["prompt_b_token_ids"]
        decisive_a_end = len(row["decisive_a_token_ids"])
        decisive_b_end = len(row["decisive_b_token_ids"])
        check = {
            "twin_id": row["twin_id"],
            "preferred_a_equals_foil_b": row["preferred_a_token_ids"] == row["foil_b_token_ids"],
            "foil_a_equals_preferred_b": row["foil_a_token_ids"] == row["preferred_b_token_ids"],
            "last5_prompts_identical": prompt_a[-5:] == prompt_b[-5:],
            "decisive_a_outside_last5": decisive_a_end <= len(prompt_a) - 5,
            "decisive_b_outside_last5": decisive_b_end <= len(prompt_b) - 5,
            "preferred_differs_from_foil_a": row["preferred_a_token_ids"] != row["foil_a_token_ids"],
            "both_continuations_nonempty": bool(row["preferred_a_token_ids"]) and bool(row["foil_a_token_ids"]),
            "all_token_ids_finite_ints": all(isinstance(value, int) and value >= 0 for value in prompt_a + prompt_b + row["preferred_a_token_ids"] + row["foil_a_token_ids"]),
        }
        check["pass"] = all(value for key, value in check.items() if key != "twin_id")
        checks.append(check)
    return checks


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=FIXED_GENERATOR_SEED)
    parser.add_argument("--attempt", type=int, default=0)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.seed != FIXED_GENERATOR_SEED:
        parser.error("fixed_generator_seed is frozen to 20260929")
    from transformers import AutoTokenizer  # noqa: PLC0415

    tokenizer = AutoTokenizer.from_pretrained(
        "distilbert/distilgpt2",
        revision="2290a62682d06624634c1f46a6ad5be0f47f38aa",
        use_fast=True,
        local_files_only=True,
    )
    if len(tokenizer) != 50257:
        parser.error("local GPT-2 tokenizer vocabulary differs from the frozen 50257-token vocabulary")
    bank = build_candidate_bank(tokenizer, attempt=args.attempt, seed=args.seed)
    bank["generation_self_sha256"] = canonical_hash(bank)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(bank, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps({"status": bank["status"], "L0_pairs": len(bank["L0_minimal_pairs"]), "L1_twin_pairs": len(bank["L1_twin_pairs"]), "output": str(args.output), "generation_self_sha256": bank["generation_self_sha256"]}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
