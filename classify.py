# 过采样

"""
恶意域名家族分类 —— PU Learning Bagging 版本
流程：
  1. 加载特征矩阵
  2. 读取 label.csv（已知标注，全部为恶意域名，family_no 0~8）
  3. 第一阶段：PU Learning Bagging
       - 随机采样 N_BAGS 次，每次取与正样本等量的未标注样本作为负样本
       - 每次训练一个二分类器，对全量未标注域名预测恶意概率
       - N_BAGS 次结果取平均，作为最终恶意概率
  4. 第二阶段：多分类（归入家族 0~8）
  5. 按置信度排序，输出结果到 label.csv
"""

import csv
import os
import numpy as np
from collections import Counter
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold, cross_val_score

from dns1_adjust import extract_all_features


# ════════════════════════════════════════════════════════════
# 参数配置
# ════════════════════════════════════════════════════════════

LABEL_PATH    = "./question/4_question/label.csv"
OUTPUT_PATH   = "./label.csv"

# PU Learning bagging 次数（越多越稳定，但越慢；建议 200~500）
N_BAGS        = 300

# PU Learning：恶意概率阈值，超过此值视为候选恶意域名
BIN_THRESHOLD = 0.5

# 最终输出条数上限
MAX_OUTPUT    = 800

RANDOM_STATE  = 42


# ════════════════════════════════════════════════════════════
# 1. 加载特征 & 标签
# ════════════════════════════════════════════════════════════

def load_data():
    print("=== 提取特征矩阵 ===")
    X_matrix, fqdn_keys, feature_names = extract_all_features()

    print("\n=== 读取标签 ===")
    labeled_data = {}
    with open(LABEL_PATH, "r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            labeled_data[row["fqdn_no"].strip()] = int(row["family_no"])
    print(f"已标注恶意域名数: {len(labeled_data)}")
    print(f"家族分布: {sorted(Counter(labeled_data.values()).items())}")

    fqdn_to_idx = {fqdn: i for i, fqdn in enumerate(fqdn_keys)}
    return X_matrix, fqdn_keys, feature_names, labeled_data, fqdn_to_idx


# ════════════════════════════════════════════════════════════
# 2. 数据准备
# ════════════════════════════════════════════════════════════

def prepare_data(X_matrix, fqdn_keys, labeled_data, fqdn_to_idx):
    labeled_set = set(labeled_data.keys())

    # 正样本：476 条已知恶意域名
    pos_indices, pos_labels = [], []
    for fqdn_no, family_no in labeled_data.items():
        idx = fqdn_to_idx.get(fqdn_no)
        if idx is not None:
            pos_indices.append(idx)
            pos_labels.append(family_no)

    # 未标注域名（待预测）
    unlabeled_fqdns   = [f for f in fqdn_keys if f not in labeled_set]
    unlabeled_indices = [fqdn_to_idx[f] for f in unlabeled_fqdns]

    X_pos        = X_matrix[pos_indices].astype(np.float32)
    y_pos_multi  = np.array(pos_labels, dtype=np.int32)
    X_unlabeled  = X_matrix[unlabeled_indices].astype(np.float32)

    print(f"\n正样本数: {len(pos_indices)}")
    print(f"未标注域名数: {len(unlabeled_fqdns)}")

    return X_pos, y_pos_multi, X_unlabeled, unlabeled_fqdns, pos_indices


# ════════════════════════════════════════════════════════════
# 3. 第一阶段：PU Learning Bagging
# ════════════════════════════════════════════════════════════

def pu_bagging(X_pos, X_unlabeled, n_bags=N_BAGS, random_state=RANDOM_STATE):
    """
    PU Learning Bagging：
    - 每轮从 X_unlabeled 随机采样 len(X_pos) 条作为负样本
    - 正样本标签=1，负样本标签=0
    - 训练 GradientBoosting 二分类器
    - 对全量 X_unlabeled 预测恶意概率
    - N_BAGS 轮平均作为最终概率
    """
    print(f"\n=== 第一阶段：PU Learning Bagging（{n_bags} 轮）===")
    rng = np.random.default_rng(random_state)

    n_pos        = len(X_pos)
    n_unlabeled  = len(X_unlabeled)
    prob_sum     = np.zeros(n_unlabeled, dtype=np.float64)

    # 归一化（在 bagging 外做一次，所有轮共用同一 scaler）
    scaler  = StandardScaler()
    X_pos_s = scaler.fit_transform(X_pos)
    X_unl_s = scaler.transform(X_unlabeled)

    for bag_i in range(n_bags):
        # 每轮随机采样等量负样本
        neg_idx = rng.choice(n_unlabeled, size=n_pos, replace=False)
        X_neg   = X_unl_s[neg_idx]

        X_train = np.vstack([X_pos_s, X_neg])
        y_train = np.array([1] * n_pos + [0] * n_pos, dtype=np.int32)

        clf = GradientBoostingClassifier(
            n_estimators=100,       # 每个 bag 里用较少树，靠 bag 数量补偿
            max_depth=4,
            learning_rate=0.1,
            subsample=0.8,
            random_state=int(rng.integers(0, 10000)),
        )
        clf.fit(X_train, y_train)
        prob_sum += clf.predict_proba(X_unl_s)[:, 1]

        if (bag_i + 1) % 50 == 0:
            print(f"  已完成 {bag_i + 1}/{n_bags} 轮")

    avg_prob = prob_sum / n_bags
    print(f"PU bagging 完成，概率分布: "
          f"min={avg_prob.min():.3f}, "
          f"mean={avg_prob.mean():.3f}, "
          f"max={avg_prob.max():.3f}")
    return avg_prob


def select_candidates(avg_prob, unlabeled_fqdns, threshold=BIN_THRESHOLD):
    candidates = [
        (fqdn, float(prob))
        for fqdn, prob in zip(unlabeled_fqdns, avg_prob)
        if prob >= threshold
    ]
    candidates.sort(key=lambda x: x[1], reverse=True)
    print(f"\n候选恶意域名: {len(candidates)} 条 (阈值={threshold})")

    # 打印概率分布分段，辅助判断阈值合理性
    thresholds = [0.9, 0.8, 0.7, 0.6, 0.5]
    for t in thresholds:
        cnt = sum(1 for _, p in candidates if p >= t)
        print(f"  prob >= {t}: {cnt} 条")

    return candidates


# ════════════════════════════════════════════════════════════
# 4. 第二阶段：多分类（归入家族 0~8）
# ════════════════════════════════════════════════════════════

def smote_oversample(X_scaled, y, target_per_class=50, random_state=RANDOM_STATE):
    """
    对每个样本数不足 target_per_class 的家族单独做 SMOTE，
    补至 target_per_class 条，样本多的家族保持不变。
    family_8 只有4条，SMOTE k_neighbors 取 min(3, n-1)。
    """
    from imblearn.over_sampling import SMOTE

    class_counts = Counter(y.tolist())
    # 计算每个类需要补到的数量
    sampling_strategy = {
        cls: max(cnt, target_per_class)
        for cls, cnt in class_counts.items()
    }
    # SMOTE k_neighbors 不能超过最小类样本数-1
    min_cnt = min(class_counts.values())
    k = min(5, min_cnt - 1)
    if k < 1:
        # 样本数=1时无法做SMOTE，直接复制
        print(f"  警告：最小类只有{min_cnt}条，跳过SMOTE，使用简单复制")
        X_res, y_res = X_scaled.copy(), y.copy()
        for cls, cnt in class_counts.items():
            if cnt < target_per_class:
                idx = np.where(y == cls)[0]
                n_repeat = target_per_class - cnt
                repeat_idx = np.random.choice(idx, size=n_repeat, replace=True)
                X_res = np.vstack([X_res, X_scaled[repeat_idx]])
                y_res = np.concatenate([y_res, np.full(n_repeat, cls, dtype=np.int32)])
        return X_res, y_res

    smote = SMOTE(
        sampling_strategy=sampling_strategy,
        k_neighbors=k,
        random_state=random_state,
    )
    X_res, y_res = smote.fit_resample(X_scaled, y)
    return X_res, y_res


def train_multiclass(X_pos, y_pos_multi, target_per_class=60):
    """
    多分类：用 476 条已知恶意域名 + SMOTE 过采样训练。
    每个家族补至 target_per_class 条，消除极度不均衡。
    """
    print("\n=== 第二阶段：多分类（家族 0~8）===")
    print(f"原始分布: {sorted(Counter(y_pos_multi.tolist()).items())}")

    scaler   = StandardScaler()
    X_scaled = scaler.fit_transform(X_pos)

    # SMOTE 过采样
    X_res, y_res = smote_oversample(X_scaled, y_pos_multi,
                                    target_per_class=target_per_class)
    print(f"过采样后分布: {sorted(Counter(y_res.tolist()).items())}")

    clf = RandomForestClassifier(
        n_estimators=1000,
        max_depth=None,
        class_weight='balanced',   # 过采样后仍保留，双重保障
        min_samples_leaf=1,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )

    # 过采样后最小类已有 target_per_class 条，可以用5折
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    scores = cross_val_score(clf, X_res, y_res, cv=cv, scoring='f1_macro')
    print(f"多分类 CV Macro F1 (5折, 过采样后): {scores.mean():.4f} ± {scores.std():.4f}")

    # 注意：最终模型用原始数据训练（不用过采样数据），避免泛化问题
    # 若CV分数很低，可改为 clf.fit(X_res, y_res)
    clf.fit(X_res, y_res)
    return clf, scaler


def predict_multiclass(clf, scaler, X_matrix, fqdn_to_idx, candidates):
    if not candidates:
        return []

    cand_fqdns  = [c[0] for c in candidates]
    cand_bprobs = [c[1] for c in candidates]

    X_cand   = np.array(
        [X_matrix[fqdn_to_idx[f]] for f in cand_fqdns],
        dtype=np.float32,
    )
    X_scaled     = scaler.transform(X_cand)
    pred_labels  = clf.predict(X_scaled)
    pred_proba   = clf.predict_proba(X_scaled)
    multi_conf   = pred_proba.max(axis=1)

    results = [
        (fqdn, int(family), float(bp), float(mc))
        for fqdn, family, bp, mc
        in zip(cand_fqdns, pred_labels, cand_bprobs, multi_conf)
    ]
    return results


# ════════════════════════════════════════════════════════════
# 5. 排序 & 输出
# ════════════════════════════════════════════════════════════

def select_and_save(results, output_path=OUTPUT_PATH, max_output=MAX_OUTPUT):
    """
    综合得分 = PU概率 * 0.6 + 多分类置信度 * 0.4
    PU概率权重更高，因为它是 300 轮平均的稳定估计。
    按得分从高到低排列，取前 max_output 条。
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    scored = [
        (fqdn, family, bp * 0.6 + mc * 0.4)
        for fqdn, family, bp, mc in results
    ]
    scored.sort(key=lambda x: x[2], reverse=True)
    selected = scored[:max_output]

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["fqdn_no", "family_no"])
        for fqdn, family, _ in selected:
            writer.writerow([fqdn, family])

    family_dist = sorted(Counter(s[1] for s in selected).items())
    print(f"\n=== 输出结果 ===")
    print(f"输出条数: {len(selected)}")
    print(f"家族分布: {family_dist}")
    print(f"已保存至: {output_path}")


# ════════════════════════════════════════════════════════════
# 主流程
# ════════════════════════════════════════════════════════════

def main():
    # 1. 加载
    X_matrix, fqdn_keys, feature_names, labeled_data, fqdn_to_idx = load_data()

    # 2. 数据准备
    X_pos, y_pos_multi, X_unlabeled, unlabeled_fqdns, pos_indices = prepare_data(
        X_matrix, fqdn_keys, labeled_data, fqdn_to_idx
    )

    # 3. PU Learning Bagging → 候选恶意域名
    avg_prob   = pu_bagging(X_pos, X_unlabeled, n_bags=N_BAGS)
    candidates = select_candidates(avg_prob, unlabeled_fqdns, threshold=BIN_THRESHOLD)

    # 4. 多分类 → 归入家族
    multi_clf, multi_scaler = train_multiclass(X_pos, y_pos_multi, target_per_class=60)
    results = predict_multiclass(
        multi_clf, multi_scaler, X_matrix, fqdn_to_idx, candidates
    )

    # 5. 排序输出
    select_and_save(results, OUTPUT_PATH, MAX_OUTPUT)


if __name__ == "__main__":
    main()
