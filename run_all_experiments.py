# -*- coding: utf-8 -*-
"""
Chạy toàn bộ pipeline "new feature" theo giao thức KHÔNG RÒ RỈ:
  0. Sinh Dataset_22features*.csv từ MalMem2022.csv (nếu chưa có).
  1. Kiểm định rò rỉ dữ liệu + trần độ chính xác lý thuyết.
  2. So sánh Naive split vs Group split (định lượng độ lạc quan giả).
  3. Benchmark tối ưu độ chính xác (22 / 55 / 55+eng đặc trưng, nhiều mô hình, ensemble,
     suy luận theo mẫu) – leakfree_benchmark.py.
  4. GAN augmentation (CWGAN-GP) trên Kịch bản 2 & 3 với kiểm định ý nghĩa thống kê.
  5. Few-shot (mọi phương pháp chỉ thấy K mẫu) + Zero-day class-disjoint.

Dùng:  python3 run_all_experiments.py [--quick]
"""

import os
import sys
import time
import argparse
import warnings

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import ensure_22feature_datasets, RESULTS_DIR, set_seed

try:
    sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass
warnings.filterwarnings('ignore')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--quick', action='store_true', help='Ít epoch/lượt hơn để chạy thử nhanh')
    ap.add_argument('--skip', nargs='*', default=[],
                    choices=['audit', 'compare', 'benchmark', 'gan', 'fewshot', 'zeroday'])
    args = ap.parse_args()
    set_seed(42)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    t0 = time.time()
    ensure_22feature_datasets()

    if 'audit' not in args.skip:
        from leakage_analysis import audit_dataset_leakage
        audit_dataset_leakage()
    if 'compare' not in args.skip:
        from compare_leak_vs_leakfree import run_leak_comparison
        run_leak_comparison()
    if 'benchmark' not in args.skip:
        import subprocess
        cmd = [sys.executable, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'leakfree_benchmark.py')]
        if args.quick:
            cmd.append('--quick')
        subprocess.run(cmd, check=True)

    gans = {}
    if 'gan' not in args.skip:
        from train_with_gan import run_gan_enrichment_experiment
        for sc in (2, 3):
            _, gans[sc] = run_gan_enrichment_experiment(
                scenario=sc, gan_levels=[200, 500, 1000, 2000],
                epochs_gan=20 if args.quick else 40, output_dir=RESULTS_DIR)

    if 'fewshot' not in args.skip:
        from train_with_gan import load_scenario_data
        from fewshot_learning import benchmark_fewshot_with_gan
        for sc, ks in [(2, [5, 10, 20, 50]), (3, [5, 10, 20])]:
            Xtr, Xte, ytr, yte, _, _, gtr, _ = load_scenario_data(None, sc, return_groups=True)
            benchmark_fewshot_with_gan(Xtr, ytr, Xte, yte, gan_model=True, k_shots=ks,
                                       n_gan_augment_per_class=100, n_runs=2 if args.quick else 3,
                                       groups_train=gtr, gan_epochs=100 if args.quick else 200,
                                       output_dir=RESULTS_DIR, tag=f"scenario_{sc}")

    if 'zeroday' not in args.skip:
        from eval_zeroday_fewshot_leakfree import run_zeroday_fewshot
        run_zeroday_fewshot(n_episodes=5 if args.quick else 20)

    print(f"\n🎉 HOÀN THÀNH TRONG {(time.time() - t0)/60:.1f} PHÚT – kết quả tại: {RESULTS_DIR}")


if __name__ == '__main__':
    main()
