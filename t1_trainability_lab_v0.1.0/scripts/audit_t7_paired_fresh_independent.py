import json, sys

base_dir = r"D:\Install\Dev\Projects\IA\Ejercise\t1_trainability_lab_v0.1.0\campaign\t7_noop_none_lexical_paired_fresh"
seeds = [7801, 7802, 7803, 7804, 7805]

total_base = 0
total_aug = 0
total_base_correct = 0
total_base_margins = 0
total_aug_correct = 0
total_aug_margins = 0
total_pairs_checked = 0
total_pairs_equiv = 0
total_pairs_target_match = 0
fp_count = 0
fn_count = 0
mismatches = []

for seed in seeds:
    base_by_pair = {}
    aug_records = []
    path = f"{base_dir}\\seed_{seed}\\cases.jsonl"
    with open(path, encoding="utf-8") as f:
        for line in f:
            c = json.loads(line)
            if c["kind"] == "base":
                total_base += 1
                if c["structured_correct"]:
                    total_base_correct += 1
                else:
                    mismatches.append(("base_incorrect", seed, c["case_id"]))
                if c["margins_strict"]:
                    total_base_margins += 1
                base_by_pair[c["pair_id"]] = c
                # check FP/FN
                for r in c["roles"]:
                    if r["expected_present"] and r["presence"] != "present":
                        fn_count += 1
                    if not r["expected_present"] and r["presence"] == "present":
                        fp_count += 1
            else:
                total_aug += 1
                if c["structured_correct"]:
                    total_aug_correct += 1
                else:
                    mismatches.append(("aug_incorrect", seed, c["case_id"]))
                if c["margins_strict"]:
                    total_aug_margins += 1
                aug_records.append(c)
                for r in c["roles"]:
                    if r["expected_present"] and r["presence"] != "present":
                        fn_count += 1
                    if not r["expected_present"] and r["presence"] == "present":
                        fp_count += 1

    # now check pairing equivalence
    for c in aug_records:
        pid = c["pair_id"]
        base = base_by_pair.get(pid)
        if base is None:
            mismatches.append(("missing_base_pair", seed, pid))
            continue
        total_pairs_checked += 1
        if base["structured_correct"] and c["structured_correct"]:
            total_pairs_target_match += 1
        if c["semantic_signature"] == base["semantic_signature"]:
            total_pairs_equiv += 1
        else:
            mismatches.append(("semantic_mismatch", seed, pid))

    print(f"seed {seed} done: base_so_far={total_base} aug_so_far={total_aug}", file=sys.stderr)

print()
print("TOTAL base:", total_base, "correct:", total_base_correct, "margins_strict:", total_base_margins)
print("TOTAL augmented:", total_aug, "correct:", total_aug_correct, "margins_strict:", total_aug_margins)
print("TOTAL pairs checked:", total_pairs_checked, "semantic_equiv:", total_pairs_equiv, "target_match(both structured_correct):", total_pairs_target_match)
print("FP:", fp_count, "FN:", fn_count)
print("mismatches found:", len(mismatches))
for m in mismatches[:20]:
    print(m)
