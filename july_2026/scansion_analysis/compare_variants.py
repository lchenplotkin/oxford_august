"""
Compares scansion.py (baseline) plus several isolated candidate-fix variants
against every green-flagged row, to see which individual change actually
helps versus which ones churn/regress.

Usage: python3 compare_variants.py
"""

import csv
import os
import re
import sys
import glob
import importlib.util

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VARIANTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'variants')
sys.path.insert(0, ROOT)
import scansion as baseline

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
        'baseline': baseline,
        'A (feminine relax)': load_module(os.path.join(VARIANTS_DIR, 'scansion_variant_a.py'), 'variant_a'),
        'B (modal preterite -e)': load_module(os.path.join(VARIANTS_DIR, 'scansion_variant_b.py'), 'variant_b'),
        'C (every/eyen)': load_module(os.path.join(VARIANTS_DIR, 'scansion_variant_c.py'), 'variant_c'),
        'D (B+C combined)': load_module(os.path.join(VARIANTS_DIR, 'scansion_variant_d.py'), 'variant_d'),
        'E (A, prefer-exact)': load_module(os.path.join(VARIANTS_DIR, 'scansion_variant_e.py'), 'variant_e'),
        'A+C (recommended)': load_module(os.path.join(VARIANTS_DIR, 'scansion_variant_ac.py'), 'variant_ac'),
    }

    totals = {name: {'match': 0, 'fixed': 0, 'regressed': 0} for name in variants}
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

                    baseline_ok = normalize(regenerate(baseline, text, target)) == green_norm

                    for name, module in variants.items():
                        if name == 'baseline':
                            if baseline_ok:
                                totals[name]['match'] += 1
                            continue
                        ok = normalize(regenerate(module, text, target)) == green_norm
                        if ok:
                            totals[name]['match'] += 1
                        if ok and not baseline_ok:
                            totals[name]['fixed'] += 1
                        elif baseline_ok and not ok:
                            totals[name]['regressed'] += 1

    print(f"Total green-flagged rows: {total_green}\n")
    print(f"{'variant':26s} {'match':>8s} {'match%':>8s} {'fixed':>8s} {'regressed':>10s} {'net':>6s}")
    for name in variants:
        m = totals[name]['match']
        f = totals[name]['fixed']
        r = totals[name]['regressed']
        print(f"{name:26s} {m:8d} {m/total_green*100:7.2f}% {f:8d} {r:10d} {f-r:6d}")


if __name__ == '__main__':
    main()
