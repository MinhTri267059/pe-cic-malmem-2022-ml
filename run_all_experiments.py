# -*- coding: utf-8 -*-
"""
Chương trình thực thi toàn bộ quy trình:
1. Huấn luyện lại các phương pháp mới trên 22 đặc trưng (new feature).
2. Làm phong phú bộ dữ liệu bằng Conditional WGAN-GP với các mức 200 - 2000 mẫu.
3. Huấn luyện và đánh giá Học vài mẫu (Few-Shot Learning): Prototypical Network, Siamese Network, GAN-Assisted Few-Shot.
4. Kiểm định rò rỉ dữ liệu (Data Leakage Audit & Fix) và đánh giá Zero-Day Class-Disjoint Few-Shot.
5. Xuất toàn bộ bảng chỉ số, biểu đồ trực quan hóa và báo cáo tổng hợp.
"""

import os
import sys
sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import time
import json
import warnings
import numpy as np
import pandas as pd

from train_with_gan import run_gan_enrichment_experiment, load_scenario_data
from fewshot_learning import benchmark_fewshot_with_gan
from leakage_analysis import audit_dataset_leakage
from compare_leak_vs_leakfree import run_leak_comparison
from eval_zeroday_fewshot_leakfree import run_zeroday_fewshot

warnings.filterwarnings('ignore')

BASE_DIR = r"D:\Tai Lieu Hoc Tap Cao Hoc\Chuyên đề nghiên cứu về Phân loại mã độc\pe-cic-malmem-2022-ml"
NEW_FEAT_DIR = os.path.join(BASE_DIR, "new feature")
RESULTS_DIR = os.path.join(NEW_FEAT_DIR, "results")


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    t_global_start = time.time()

    print("=" * 80)
    print("🌟 CHƯƠNG TRÌNH HUẤN LUYỆN NÂNG CAO VỚI PHƯƠNG PHÁP MỚI & TĂNG CƯỜNG DỮ LIỆU GAN")
    print("   Bộ dữ liệu: CIC-MalMem-2022 với 22 đặc trưng chuyên sâu về cơ chế ẩn giấu")
    print("=" * 80)

    # -------------------------------------------------------------------------
    # BƯỚC 1: KỊCH BẢN 2 (4 LỚP: BENIGN, RANSOMWARE, SPYWARE, TROJAN)
    # -------------------------------------------------------------------------
    print("\n" + "#" * 80)
    print("📍 [BƯỚC 1/3] THỰC THI KỊCH BẢN 2 (4 LỚP PHÂN LOẠI HÀNH VI)")
    print("#" * 80)

    res_df_s2, gan_s2 = run_gan_enrichment_experiment(
        scenario=2,
        gan_levels=[200, 500, 1000, 2000],
        epochs_gan=40,
        output_dir=RESULTS_DIR
    )

    # Đánh giá Few-Shot Learning trên Kịch bản 2
    csv_s2 = os.path.join(NEW_FEAT_DIR, "Dataset_22features_Scenario_2.csv")
    X_train_s2, X_test_s2, y_train_s2, y_test_s2, mapping_s2, feat_cols = load_scenario_data(csv_s2, scenario=2)

    res_fewshot_s2 = benchmark_fewshot_with_gan(
        X_train_s2, y_train_s2, X_test_s2, y_test_s2,
        gan_model=gan_s2,
        k_shots=[5, 10, 20, 50],
        n_gan_augment_per_class=100,
        n_runs=3,
        output_dir=RESULTS_DIR
    )
    res_fewshot_s2.to_csv(os.path.join(RESULTS_DIR, "fewshot_benchmark_results_scenario_2.csv"), index=False)

    # -------------------------------------------------------------------------
    # BƯỚC 2: KỊCH BẢN 3 (16 LỚP: BENIGN + 15 HỌ MÃ ĐỘC CHI TIẾT)
    # -------------------------------------------------------------------------
    print("\n" + "#" * 80, flush=True)
    print("📍 [BƯỚC 2/3] THỰC THI KỊCH BẢN 3 (16 LỚP PHÂN LOẠI HỌ MÃ ĐỘC CHI TIẾT)", flush=True)
    print("#" * 80, flush=True)

    res_df_s3, gan_s3 = run_gan_enrichment_experiment(
        scenario=3,
        gan_levels=[200, 500, 1000, 2000],
        epochs_gan=45,
        output_dir=RESULTS_DIR
    )

    # Đánh giá Few-Shot Learning trên Kịch bản 3
    csv_s3 = os.path.join(NEW_FEAT_DIR, "Dataset_22features_Scenario_3.csv")
    X_train_s3, X_test_s3, y_train_s3, y_test_s3, mapping_s3, feat_cols = load_scenario_data(csv_s3, scenario=3)

    res_fewshot_s3 = benchmark_fewshot_with_gan(
        X_train_s3, y_train_s3, X_test_s3, y_test_s3,
        gan_model=gan_s3,
        k_shots=[5, 10, 20],
        n_gan_augment_per_class=50,
        n_runs=3,
        output_dir=RESULTS_DIR
    )
    res_fewshot_s3.to_csv(os.path.join(RESULTS_DIR, "fewshot_benchmark_results_scenario_3.csv"), index=False)

    # -------------------------------------------------------------------------
    # BƯỚC 3: KIỂM ĐỊNH DATA LEAKAGE & ĐỐI SÁNH LEAK-FREE GROUP SPLIT & ZERO-DAY
    # -------------------------------------------------------------------------
    print("\n" + "#" * 80, flush=True)
    print("📍 [BƯỚC 3/3] KIỂM ĐỊNH DATA LEAKAGE & THỰC NGHIỆM ĐỐI SÁNH KHÔNG RÒ RỈ", flush=True)
    print("#" * 80, flush=True)
    audit_dataset_leakage()
    run_leak_comparison()
    run_zeroday_fewshot()

    # -------------------------------------------------------------------------
    # BƯỚC 4: TỔNG KẾT TOÀN DIỆN
    # -------------------------------------------------------------------------
    total_elapsed = time.time() - t_global_start
    print("\n" + "=" * 80)
    print(f"🎉 HOÀN THÀNH TOÀN BỘ CÁC THỰC NGHIỆM TRONG: {total_elapsed/60:.2f} PHÚT!")
    print(f"📁 TẤT CẢ FILE KẾT QUẢ, DATASET MỚI VÀ BIỂU ĐỒ ĐÃ ĐƯỢC LƯU TẠI: {RESULTS_DIR}")
    print("=" * 80)


if __name__ == '__main__':
    main()
