"""
Compares scansion_v2.py (current best) against new candidate-fix variants in
variants2/, isolating the medial-e-elision restriction (D1) and the
diphthong+final-e syllable flex (D2) to see whether each helps versus churns.

Usage: python3 compare_variants2.py
"""

import csv
import os
import re
import sys
import glob
import importlib.util

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VARIANTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'variants2')
sys.path.insert(0, ROOT)
import scansion_v2 as baseline

SCANSION_TOOL_DIR = os.path.join(ROOT, 'scansion_tool')
FOLDERS = ['to_complete', 'in_progress', 'completed_unvetted', 'gold']
DEFAULT_TARGET = 10
BD_HF_PREFIXES = ('BD_', 'HF_')


def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def target_for_filename(filename):
    return 8 if filename.startswith(BD_HF_PREFIXES) else DEFAULT_TARGET


def regenerate(module, text, target):
    if not text or not str(text).strip():
        return ''
    stresses, _ = module.scan(text, target)
    return ' '.join(stresses)


def normalize(s):
    s = re.sub(r'[xX]', '', s)
    return re.sub(r'\s+', ' ', s).strip()


def main():
    variants = {
        'baseline (v2)': baseline,
        'D1 (medial-e final-cluster only)': load_module(os.path.join(VARIANTS_DIR, 'variant_d1_final_cluster_only.py'), 'variant_d1'),
        'D2 (diphthong+e unfollowed flex)': load_module(os.path.join(VARIANTS_DIR, 'variant_d2_diphthong_e_unfollowed.py'), 'variant_d2'),
        'D1+D2 combined': load_module(os.path.join(VARIANTS_DIR, 'variant_d1d2_combined.py'), 'variant_d1d2'),
        'F (funcword tiebreak)': load_module(os.path.join(VARIANTS_DIR, 'variant_f_funcword_tiebreak.py'), 'variant_f'),
        'G (initial "ye" glide fix)': load_module(os.path.join(VARIANTS_DIR, 'variant_g_initial_ye.py'), 'variant_g'),
        'H (G + Proigne)': load_module(os.path.join(VARIANTS_DIR, 'variant_h_g_plus_proigne.py'), 'variant_h'),
        'G2 (ye always 1, pos-independent)': load_module(os.path.join(VARIANTS_DIR, 'variant_g2_ye_always_one.py'), 'variant_g2'),
    }

    totals = {name: {'match': 0, 'fixed': 0, 'regressed': 0} for name in variants}
    fixed_examples = {name: [] for name in variants}
    regressed_examples = {name: [] for name in variants}
    total_green = 0

    for folder in FOLDERS:
        for path in sorted(glob.glob(os.path.join(SCANSION_TOOL_DIR, folder, '*.csv'))):
            filename = os.path.basename(path)
            target = target_for_filename(filename)

            with open(path, newline='', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    flag = (row.get('SCANSION_FLAG_COLOR') or '').strip().lower()
                    if flag != 'green':
                        continue
                    total_green += 1

                    text = row.get('OXFORD_TEXT', '')
                    green_norm = normalize(row.get('OXFORD_SCANSION', ''))

                    baseline_scansion = regenerate(baseline, text, target)
                    baseline_ok = normalize(baseline_scansion) == green_norm

                    for name, module in variants.items():
                        if name == 'baseline (v2)':
                            if baseline_ok:
                                totals[name]['match'] += 1
                            continue
                        v_scansion = regenerate(module, text, target)
                        ok = normalize(v_scansion) == green_norm
                        if ok:
                            totals[name]['match'] += 1
                        if ok and not baseline_ok:
                            totals[name]['fixed'] += 1
                            fixed_examples[name].append((text, green_norm, baseline_scansion, v_scansion))
                        elif baseline_ok and not ok:
                            totals[name]['regressed'] += 1
                            regressed_examples[name].append((text, green_norm, baseline_scansion, v_scansion))

    print(f"Total green-flagged rows: {total_green}\n")
    print(f"{'variant':36s} {'match':>8s} {'match%':>8s} {'fixed':>8s} {'regressed':>10s} {'net':>6s}")
    for name in variants:
        m = totals[name]['match']
        f = totals[name]['fixed']
        r = totals[name]['regressed']
        print(f"{name:36s} {m:8d} {m/total_green*100:7.2f}% {f:8d} {r:10d} {f-r:6d}")

    for name in variants:
        if name == 'baseline (v2)':
            continue
        print(f"\n--- {name}: regressed examples (up to 10) ---")
        for text, g, b, v in regressed_examples[name][:10]:
            print(f"  {text}\n    green: {g}\n    v2   : {b}\n    var  : {v}")


if __name__ == '__main__':
    main()
