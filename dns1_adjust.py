import csv
import os
import re
from collections import Counter, defaultdict
from scipy.sparse import lil_matrix, save_npz, load_npz
from sklearn.manifold import SpectralEmbedding
from sklearn.preprocessing import normalize
import numpy as np
import json


def read_fqdn_csv(csv_file_path = "./question/4_question/fqdn.csv"):
    fqdn_data = {}

    if not os.path.exists(csv_file_path):
        print(f"错误：文件 {csv_file_path} 不存在！")
        return fqdn_data

    with open(csv_file_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter=",")

        for row_num, row in enumerate(reader, start=2):
            encoded_fqdn = row["encoded_fqdn"].strip()
            fqdn_no = row["fqdn_no"].strip()

            fqdn_data[fqdn_no] = encoded_fqdn

        print(f"共加载 {len(fqdn_data)} 条域名数据")

    return fqdn_data


def read_access_csv(csv_file_path = "./question/4_question/access.csv"):
    access_data = {}
    access_ip_map = defaultdict(lambda: defaultdict(list))

    if not os.path.exists(csv_file_path):
        print(f"错误：文件 {csv_file_path} 不存在！")
        return access_data, dict(access_ip_map)

    with open(csv_file_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter=",")

        for row_num, row in enumerate(reader, start=2):
            count = int(row["count"])
            time = row["time"].strip()
            date = time[:8]
            hour = time[8:10]
            fqdn_no = row["fqdn_no"].strip()

            if fqdn_no not in access_data:
                access_data[fqdn_no] = []

            access_data[fqdn_no].append({"request_cnt":count, "date":date, "hour":hour})

            encoded_ip = row["encoded_ip"].strip()
            if encoded_ip:
                access_ip_map[fqdn_no][encoded_ip].append({
                    "date": date,
                    "hour": int(hour),
                    "request_cnt": count,
                })

        print(f"共加载 {len(access_data)} 条域名数据")

    return access_data, dict(access_ip_map)


def read_flint_csv(csv_file_path = "./question/4_question/flint.csv"):
    flint_data = {}

    if not os.path.exists(csv_file_path):
        print(f"错误：文件 {csv_file_path} 不存在！")
        return flint_data

    with open(csv_file_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter=",")

        for row_num, row in enumerate(reader, start=2):
            flintType = int(row["flintType"])
            encoded_value = row["encoded_value"].strip()
            requestCnt = int(row["requestCnt"])
            date = row["date"].strip()
            fqdn_no = row["fqdn_no_x"].strip()

            if fqdn_no not in flint_data:
                flint_data[fqdn_no] = []

            flint_data[fqdn_no].append({"flintType":flintType, "encoded_value":encoded_value, "requestCnt":requestCnt, "date":date})

        print(f"共加载 {len(flint_data)} 条域名数据")

    return flint_data


def read_ip_csv(csv_file_path = "./question/4_question/ip.csv"):
    ip_data = {}

    if not os.path.exists(csv_file_path):
        print(f"错误：文件 {csv_file_path} 不存在！")
        return ip_data

    with open(csv_file_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter=",")

        for row_num, row in enumerate(reader, start=2):
            country = row["country"].strip()
            subdivision = row["subdivision"].strip()
            isp = row["isp"].strip()
            encoded_ip = row["encoded_ip"].strip()

            if encoded_ip not in ip_data:
                ip_data[encoded_ip] = []

            ip_data[encoded_ip].append({"country":country, "subdivision":subdivision, "isp":isp})

        print(f"共加载 {len(ip_data)} 条数据")

    return ip_data


def read_ipv6_csv(csv_file_path = "./question/4_question/ipv6.csv"):
    ipv6_data = {}

    if not os.path.exists(csv_file_path):
        print(f"错误：文件 {csv_file_path} 不存在！")
        return ipv6_data

    with open(csv_file_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter=",")

        for row_num, row in enumerate(reader, start=2):
            country = row["country"].strip()
            subdivision = row["subdivision"].strip()
            isp = row["isp"].strip()
            encoded_ip = row["encoded_ip"].strip()

            if encoded_ip not in ipv6_data:
                ipv6_data[encoded_ip] = []

            ipv6_data[encoded_ip].append({"country":country, "subdivision":subdivision, "isp":isp})

        print(f"共加载 {len(ipv6_data)} 条数据")

    return ipv6_data


def read_label_csv(csv_file_path = "./question/4_question/label.csv"):
    labeled_data = {}

    if not os.path.exists(csv_file_path):
        print(f"错误：文件 {csv_file_path} 不存在！")
        return labeled_data

    with open(csv_file_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter=",")

        for row_num, row in enumerate(reader, start=2):
            fqdn_no = row["fqdn_no"].strip()
            label = int(row["label"])

            if fqdn_no not in labeled_data:
                labeled_data[fqdn_no] = label

        print(f"共加载 {len(labeled_data)} 条数据")

    return labeled_data


def extract_domainname_features(encoded_fqdn):
    COMMON_TLDS = {"com", "net", "cn", "org"}
    result = {
        "is_common_tld": 0,

        "length": 0,

        "letter_count": 0,  # 字母数量
        "digit_count": 0,  # 数字数量
        "word_count": 0,  # 单词数量
        "dot_count": 0,  # 前缀中点数量

        "letter_ratio": 0.0,
        "digit_ratio": 0.0,
        "word_ratio": 0.0,
        "dot_ratio": 0.0,
        "label_count": 0,
        "max_label_length": 0,
    }

    parts = encoded_fqdn.split(".")
    suffix = parts[-1]
    prefix = ".".join(parts[:-1])

    result["length"] = len(prefix)

    if suffix.lower() in COMMON_TLDS:
        result["is_common_tld"] = 1

    # 匹配编码后的词块，如 [aaaa]、[aaaaaaaa]
    word_pattern = re.compile(r'\[a+\]')
    words = word_pattern.findall(prefix)
    result["word_count"] = len(words)

    pure_prefix = word_pattern.sub('', prefix)

    result["dot_count"] = pure_prefix.count(".")
    result["letter_count"] = pure_prefix.count("a")
    result["digit_count"] = pure_prefix.count("0")

    labels = [p for p in pure_prefix.split(".") if p]
    result["label_count"] = len(labels)
    result["max_label_length"] = max((len(x) for x in labels), default=0)

    total = len(pure_prefix) + result["word_count"]

    if total > 0:
        result["letter_ratio"] = round(result["letter_count"] / total * 100, 2)
        result["digit_ratio"] = round(result["digit_count"] / total * 100, 2)
        result["word_ratio"] = round(result["word_count"] / total * 100, 2)
        result["dot_ratio"] = round(result["dot_count"] / total * 100, 2)
    return result


def six_stats(arr):
    a = sorted(arr)
    n = len(a)
    mean = sum(a) / n
    var = sum((x - mean) ** 2 for x in a) / n
    std = var ** 0.5
    median = (a[n // 2 - 1] + a[n // 2]) / 2 if n % 2 == 0 else a[n // 2]
    return max(a), min(a), mean, var, median, std


def extract_access_features(access_data, access_ip_info=None):
    """
    access_data    : list of {request_cnt, date, hour}
    access_ip_info : dict[encoded_ip -> list[{date, hour, request_cnt}]]
                     来自 access_ip_map[fqdn_no]，用于计算客户端IP相关特征
    """
    result = {
        # 原有特征
        'total_request_cnt': 0,
        'active_days': 0,

        'daily_request_max': 0, 'daily_request_min': 0, 'daily_request_mean': 0,
        'daily_request_var': 0, 'daily_request_median': 0, 'daily_request_std': 0,

        'hour_max': 0, 'hour_min': 0, 'hour_mean': 0,
        'hour_var': 0, 'hour_median': 0, 'hour_std': 0,

        'request_mean': 0, 'request_var': 0,
        'night_request_ratio': 0,

        # 新增：时间范围
        'first_access_date': 0,   # 最早访问日期（yyyymmdd 转整数）
        'last_access_date': 0,    # 最晚访问日期

        # 新增：客户端IP特征（需要 access_ip_info）
        'unique_client_ip_count': 0,        # 访问IP种类数
        'consecutive_ip_ratio': 0.0,        # 连续访问IP / 所有IP
        'same_day_consecutive_ip_ratio': 0.0,  # 同一天连续IP / 所有连续IP
        'request_burstiness': 0.0,             # 请求突发性：std/mean
        'workhour_request_ratio': 0.0,         # 工作时段(9~18)请求占比
    }

    if not access_data:
        return result

    date_map = defaultdict(int)
    hour_bucket = defaultdict(int)
    total_request_cnt = 0
    request_list = []
    all_dates = []

    for item in access_data:
        d = item['date']
        date_map[d] += item['request_cnt']
        total_request_cnt += item['request_cnt']
        hour_bucket[int(item['hour'])] += item['request_cnt']
        request_list.append(item['request_cnt'])
        all_dates.append(d)

    result['total_request_cnt'] = total_request_cnt
    result['active_days'] = len(date_map)

    # 时间范围
    sorted_dates = sorted(date_map.keys())
    result['first_access_date'] = int(sorted_dates[0])  if sorted_dates else 0
    result['last_access_date']  = int(sorted_dates[-1]) if sorted_dates else 0

    daily_requests = list(date_map.values())
    dr_max, dr_min, dr_mean, dr_var, dr_median, dr_std = six_stats(daily_requests)
    result['daily_request_max']    = dr_max
    result['daily_request_min']    = dr_min
    result['daily_request_mean']   = dr_mean
    result['daily_request_var']    = dr_var
    result['daily_request_median'] = dr_median
    result['daily_request_std']    = dr_std

    hour_request_counts = list(hour_bucket.values())
    h_max, h_min, h_mean, h_var, h_median, h_std = six_stats(hour_request_counts)
    result['hour_max']    = h_max
    result['hour_min']    = h_min
    result['hour_mean']   = h_mean
    result['hour_var']    = h_var
    result['hour_median'] = h_median
    result['hour_std']    = h_std

    request_ratio_mean = sum(request_list) / len(request_list)
    request_ratio_var  = sum((x - request_ratio_mean) ** 2 for x in request_list) / len(request_list)
    result['request_mean'] = request_ratio_mean
    result['request_var']  = request_ratio_var

    night_requests = sum(item['request_cnt'] for item in access_data if int(item['hour']) < 6)
    result['night_request_ratio'] = night_requests / total_request_cnt if total_request_cnt > 0 else 0
    workhour_requests = sum(
        item['request_cnt'] for item in access_data
        if 9 <= int(item['hour']) <= 18
    )
    result['workhour_request_ratio'] = workhour_requests / total_request_cnt if total_request_cnt > 0 else 0.0
    if request_ratio_mean > 0:
        result['request_burstiness'] = (request_ratio_var ** 0.5) / request_ratio_mean

    # ── 客户端 IP 特征 ────────────────────────────────────────
    if access_ip_info:
        all_client_ips = list(access_ip_info.keys())
        n_ips = len(all_client_ips)
        result['unique_client_ip_count'] = n_ips

        if n_ips >= 2:
            # 连续IP定义：同一IP在连续两天都访问过该域名
            # 构建 ip -> set(date) 映射
            ip_dates = {}
            for ip, records in access_ip_info.items():
                ip_dates[ip] = set(r['date'] for r in records)

            all_access_dates_set = set(date_map.keys())
            sorted_all_dates = sorted(all_access_dates_set)

            # 连续日期对集合
            consecutive_date_pairs = set()
            for i in range(len(sorted_all_dates) - 1):
                d1, d2 = sorted_all_dates[i], sorted_all_dates[i+1]
                # 判断是否真正相邻（日期差1天）
                try:
                    from datetime import datetime
                    dt1 = datetime.strptime(d1, '%Y%m%d')
                    dt2 = datetime.strptime(d2, '%Y%m%d')
                    if (dt2 - dt1).days == 1:
                        consecutive_date_pairs.add((d1, d2))
                except Exception:
                    pass

            consecutive_ips = set()
            same_day_consecutive_ips = set()

            for ip, dates in ip_dates.items():
                # 连续IP：在任意两个连续日期都出现
                for d1, d2 in consecutive_date_pairs:
                    if d1 in dates and d2 in dates:
                        consecutive_ips.add(ip)
                        break
                # 同一天连续IP：该IP在同一天内连续小时出现
                # 按日期分组小时
                day_hours = defaultdict(set)
                for r in access_ip_info[ip]:
                    day_hours[r['date']].add(r['hour'])
                for day, hours in day_hours.items():
                    sorted_h = sorted(hours)
                    for i in range(len(sorted_h) - 1):
                        if sorted_h[i+1] - sorted_h[i] == 1:
                            same_day_consecutive_ips.add(ip)
                            break

            n_consec = len(consecutive_ips)
            result['consecutive_ip_ratio'] = n_consec / n_ips
            result['same_day_consecutive_ip_ratio'] = (
                len(same_day_consecutive_ips) / n_consec if n_consec > 0 else 0.0
            )

    return result


def get_cname_depth(fqdn_key, all_flint_dict, visited=None):
    if visited is None:
        visited = set()
    if fqdn_key in visited:
        return 0  # 成环，终止
    visited.add(fqdn_key)

    flint_data = all_flint_dict.get(fqdn_key, [])

    # 找到所有 CNAME 目标（去重）
    cname_targets = {item['encoded_value'] for item in flint_data if item['flintType'] == 5}

    # 有 A/AAAA 记录说明链到头了
    has_terminal = any(item['flintType'] in (1, 28) for item in flint_data)

    if not cname_targets:
        return 0  # 没有 CNAME，深度为 0

    if has_terminal:
        return 1  # 有 CNAME 也有 A 记录，算1跳就终止

    # 递归取所有 CNAME 目标中最大深度
    max_depth = 0
    for target in cname_targets:
        depth = get_cname_depth(target, all_flint_dict, visited)
        max_depth = max(max_depth, depth)

    return 1 + max_depth


def extract_flint_features(fqdn_key, flint_data, all_flint_dict, malicious_fqdns=None, malicious_ips=None):
    """
    malicious_fqdns : set of fqdn_no that are known malicious（来自 label.csv）
    malicious_ips   : set of encoded_ip that are resolved by known malicious domains
    """
    result = {
        # 原有
        'a_ratio': 0.0,
        'c_ratio': 0.0,
        'aaaa_ratio': 0.0,
        'cname_depth': 0,
        'cname_count': 0,
        'unique_ip_count': 0,
        'ip_concentration': 0,
        'avg_daily_ip': 0,
        'avg_ip_churn': 0,

        # 新增：时间范围
        'flint_first_date': 0,   # 最早解析日期
        'flint_last_date': 0,    # 最晚解析日期

        # 新增：每日请求统计
        'flint_daily_req_max': 0,
        'flint_daily_req_min': 0,
        'flint_daily_req_mean': 0,
        'flint_daily_req_std': 0,

        # 新增：恶意关联特征
        'is_pointed_by_malicious': 0,    # 是否被已知恶意域名指向（被CNAME到）
        'cname_target_is_malicious': 0,  # CNAME指向的域名是否为恶意域名
        'resolved_ip_is_malicious': 0,   # 解析到的IP是否被恶意域名使用
        'resolved_ipv6_is_malicious': 0, # 解析到的IPv6是否被恶意域名使用
        'ipv6_ratio_in_ips': 0.0,        # 在A/AAAA记录中，AAAA占比
    }

    flint_types = [item["flintType"] for item in flint_data]
    type_count = Counter(flint_types)
    total = len(flint_data)
    if total == 0:
        return result

    # ── 类型比例 ──────────────────────────────────────────────
    for t, key in [(1, 'a_ratio'), (5, 'c_ratio'), (28, 'aaaa_ratio')]:
        result[key] = type_count.get(t, 0) / total * 100

    # ── CNAME 深度/数量 ───────────────────────────────────────
    result['cname_depth'] = get_cname_depth(fqdn_key, all_flint_dict)
    unique_cname = {item['encoded_value'] for item in flint_data if item['flintType'] == 5}
    result['cname_count'] = len(unique_cname)

    # ── IP 特征 ───────────────────────────────────────────────
    unique_ips = {item['encoded_value'] for item in flint_data if item['flintType'] in (1, 28)}
    result['unique_ip_count'] = len(unique_ips)
    ip_records = sum(1 for item in flint_data if item['flintType'] in (1, 28))
    if ip_records > 0:
        result['ipv6_ratio_in_ips'] = type_count.get(28, 0) / ip_records

    ip_request_cnt = defaultdict(int)
    for item in flint_data:
        if item['flintType'] in (1, 28):
            ip_request_cnt[item['encoded_value']] += item['requestCnt']
    total_ip_request = sum(ip_request_cnt.values())
    if total_ip_request > 0:
        result['ip_concentration'] = max(ip_request_cnt.values()) / total_ip_request

    date_ip_map = defaultdict(set)
    date_req_map = defaultdict(int)
    for item in flint_data:
        date_req_map[item['date']] += item['requestCnt']
        if item['flintType'] in (1, 28):
            date_ip_map[item['date']].add(item['encoded_value'])

    sorted_dates = sorted(date_ip_map.keys())
    daily_ip_counts = [len(date_ip_map[d]) for d in sorted_dates]
    result['avg_daily_ip'] = sum(daily_ip_counts) / len(daily_ip_counts) if daily_ip_counts else 0

    diff_counts = []
    for i in range(len(sorted_dates) - 1):
        diff_counts.append(len(
            date_ip_map[sorted_dates[i]].symmetric_difference(date_ip_map[sorted_dates[i+1]])
        ))
    result['avg_ip_churn'] = sum(diff_counts) / len(diff_counts) if diff_counts else 0

    # ── 时间范围 ──────────────────────────────────────────────
    all_dates = sorted(date_req_map.keys())
    if all_dates:
        result['flint_first_date']  = int(all_dates[0])
        result['flint_last_date']   = int(all_dates[-1])

    # ── 每日请求统计 ──────────────────────────────────────────
    daily_reqs = list(date_req_map.values())
    if daily_reqs:
        mean_r = sum(daily_reqs) / len(daily_reqs)
        std_r  = (sum((x - mean_r)**2 for x in daily_reqs) / len(daily_reqs)) ** 0.5
        result['flint_daily_req_max']  = max(daily_reqs)
        result['flint_daily_req_min']  = min(daily_reqs)
        result['flint_daily_req_mean'] = mean_r
        result['flint_daily_req_std']  = std_r

    # ── 恶意关联特征 ──────────────────────────────────────────
    if malicious_fqdns is not None:
        # 是否被已知恶意域名通过CNAME指向本域名
        # 遍历所有域名的flint记录，找CNAME到本域名的
        # （此特征在 build_feature_vector 里预计算更高效，这里用传入值）
        pass  # 由外部预计算后传入，见下方说明

    if malicious_fqdns is not None:
        # CNAME指向的域名是否为恶意
        cname_targets = {item['encoded_value'] for item in flint_data if item['flintType'] == 5}
        if cname_targets & malicious_fqdns:
            result['cname_target_is_malicious'] = 1

    if malicious_ips is not None:
        # 解析到的IPv4是否被恶意域名使用
        resolved_v4 = {item['encoded_value'] for item in flint_data if item['flintType'] == 1}
        if resolved_v4 & malicious_ips:
            result['resolved_ip_is_malicious'] = 1
        # 解析到的IPv6
        resolved_v6 = {item['encoded_value'] for item in flint_data if item['flintType'] == 28}
        if resolved_v6 & malicious_ips:
            result['resolved_ipv6_is_malicious'] = 1

    return result


def extract_ip_features(fqdn_key, all_flint_dict, ip_dict, ipv6_dict, access_ip_map=None):
    result = {
        'country_diversity': 0,
        'subdivision_diversity': 0,   # 新增：省级地理多样性
        'isp_diversity': 0,
        'access_only_ip_count': 0,
        'total_unique_ip_count': 0,
        # 新增：客户端IP地理/ISP（访问侧）
        'client_subdivision_diversity': 0,
        'client_isp_diversity': 0,
    }
    flint_data = all_flint_dict.get(fqdn_key, [])

    flint_ip_set = {
        (item['encoded_value'], item['flintType'])
        for item in flint_data
        if item['flintType'] in (1, 28)
    }

    access_ips = set()
    if access_ip_map:
        for encoded_ip in access_ip_map.get(fqdn_key, {}).keys():
            access_ips.add(encoded_ip)

    flint_ip_encoded = {enc for enc, _ in flint_ip_set}
    access_only_ips = access_ips - flint_ip_encoded
    result['access_only_ip_count']    = len(access_only_ips)
    result['total_unique_ip_count']   = len(flint_ip_encoded | access_ips)

    if not flint_ip_set and not access_ips:
        return result

    # ── 解析侧 IP 地理/ISP（flint A/AAAA 记录）────────────────
    countries     = set()
    subdivisions  = set()
    isps          = set()

    for encoded_ip, ftype in flint_ip_set:
        source_dict = ipv6_dict if ftype == 28 else ip_dict
        info_list = source_dict.get(encoded_ip)
        if not info_list:
            continue
        info = info_list[0]
        if info.get('country'):
            countries.add(info['country'])
        if info.get('subdivision'):
            subdivisions.add(info['subdivision'])
        if info.get('isp'):
            isps.add(info['isp'])

    for encoded_ip in access_only_ips:
        info_list = ip_dict.get(encoded_ip) or ipv6_dict.get(encoded_ip)
        if not info_list:
            continue
        info = info_list[0]
        if info.get('country'):
            countries.add(info['country'])
        if info.get('subdivision'):
            subdivisions.add(info['subdivision'])
        if info.get('isp'):
            isps.add(info['isp'])

    result['country_diversity']    = len(countries)
    result['subdivision_diversity'] = len(subdivisions)
    result['isp_diversity']        = len(isps)

    # ── 客户端 IP 地理/ISP（access.csv 访问侧）────────────────
    if access_ip_map:
        cli_countries    = set()
        cli_subdivisions = set()
        cli_isps         = set()
        for encoded_ip in access_ip_map.get(fqdn_key, {}).keys():
            info_list = ip_dict.get(encoded_ip) or ipv6_dict.get(encoded_ip)
            if not info_list:
                continue
            info = info_list[0]
            if info.get('country'):
                cli_countries.add(info['country'])
            if info.get('subdivision'):
                cli_subdivisions.add(info['subdivision'])
            if info.get('isp'):
                cli_isps.add(info['isp'])
        result['client_subdivision_diversity'] = len(cli_subdivisions)
        result['client_isp_diversity']         = len(cli_isps)

    return result


def build_cooccur_graph(flint_dict, access_ip_map, all_fqdns):
    fqdn_index = {f: i for i, f in enumerate(all_fqdns)}
    n = len(all_fqdns)

    # 同前：建 (ip, date) -> {fqdn_no} 倒排索引
    ip_date_fqdns = defaultdict(set)
    for fqdn_no, records in flint_dict.items():
        for item in records:
            if item['flintType'] in (1, 28):
                ip_date_fqdns[(item['encoded_value'], item['date'])].add(fqdn_no)
    for fqdn_no, ip_dict_inner in access_ip_map.items():
        for encoded_ip, time_records in ip_dict_inner.items():
            for tr in time_records:
                ip_date_fqdns[(encoded_ip, tr['date'])].add(fqdn_no)

    mat = lil_matrix((n, n), dtype=np.float32)
    for (ip, date), fqdn_set in ip_date_fqdns.items():
        fqdn_list = [f for f in fqdn_set if f in fqdn_index]
        if len(fqdn_list) < 2 or len(fqdn_list) > 500:  # 过滤噪声 IP
            continue
        for i in range(len(fqdn_list)):
            for j in range(i + 1, len(fqdn_list)):
                a, b = fqdn_index[fqdn_list[i]], fqdn_index[fqdn_list[j]]
                mat[a, b] += 1
                mat[b, a] += 1

    return mat.tocsr()


def compute_spectral_embedding(cooccur_mat, n_components=8):
    dense_mat = cooccur_mat.toarray()

    dense_mat += np.eye(dense_mat.shape[0], dtype=np.float32) * 1e-6

    embedder = SpectralEmbedding(
        n_components=n_components,
        affinity='precomputed',
        random_state=42,
    )
    embedding = embedder.fit_transform(dense_mat)

    embedding = normalize(embedding, norm='l2')
    return embedding


REF_TS_MS = 1586908800000
MS_PER_DAY = 86_400_000

def _ns_root(ns: str) -> str:
    """从 NS 主机名中 提取二级域作为提供商标识。
    例如：ns1.cloudflare.com  ->  cloudflare
          ns1a.o365filtering.com  ->  o365filtering
    """
    parts = ns.strip().lower().split(".")
    return parts[-2] if len(parts) >= 2 else parts[0]


def _coalesce(records: list, field: str) -> object:
    """在多条记录中取最新（updateddate 最大）的非 null 值。
    若值为列表，取第一个非空元素。
    """
    sorted_records = sorted(
        records,
        key=lambda r: r.get("updateddate") or 0,
        reverse=True,
    )
    for r in sorted_records:
        val = r.get(field)
        if val is None or val == "":
            continue
        if isinstance(val, list):
            val = next((v for v in val if v), None)
        if val:
            return val
    return None


def _merge_records(records: list) -> dict:
    """将同一域名的多条 whois 记录按字段策略合并为一条。"""
    created_vals = [r["createddate"] for r in records if r.get("createddate")]
    expires_vals = [r["expiresdate"]  for r in records if r.get("expiresdate")]
    updated_vals = [r["updateddate"]  for r in records if r.get("updateddate")]

    all_ns: set[str] = set()
    for r in records:
        all_ns.update(r.get("nameservers") or [])

    all_ws: set[str] = set(
        r["whoisserver"] for r in records if r.get("whoisserver")
    )

    # 新增：注册国家/邮箱的全部非空值（用于计算多样性）
    # 字段可能是字符串或列表，统一展平处理
    def _flatten_field(recs, field):
        out = set()
        for r in recs:
            val = r.get(field)
            if not val:
                continue
            if isinstance(val, list):
                for v in val:
                    if v:
                        out.add(str(v))
            else:
                out.add(str(val))
        return out

    all_registrant_countries = _flatten_field(records, "registrant_country")
    all_registrant_emails    = _flatten_field(records, "registrant_email")

    return {
        "created":          min(created_vals) if created_vals else None,
        "latest_created":   max(created_vals) if created_vals else None,  # 新增
        "expires":          max(expires_vals) if expires_vals else None,
        "first_updated":    min(updated_vals) if updated_vals else None,  # 新增
        "updated":          max(updated_vals) if updated_vals else None,
        "record_count":     len(records),                                  # 新增
        "update_count":     len(updated_vals),                             # 新增
        "all_nameservers":  all_ns,
        "all_whoisservers": all_ws,
        # coalesce 联系信息
        "registrant_country": _coalesce(records, "registrant_country"),
        "registrant_email":   _coalesce(records, "registrant_email"),
        "admin_email":        _coalesce(records, "admin_email"),
        "tech_email":         _coalesce(records, "tech_email"),
        "sponsoring":         _coalesce(records, "sponsoring"),
        # 新增：多样性
        "registrant_country_count": len(all_registrant_countries),
        "registrant_email_count":   len(all_registrant_emails),
    }


def read_whois_json(json_path: str = "./question/4_question/whois.json") -> dict:
    """
    返回 {fqdn_no: merged_record} 字典。
    merged_record 已按字段策略将同一域名的多条记录合并。
    """
    if not os.path.exists(json_path):
        print(f"错误：文件 {json_path} 不存在！")
        return {}

    with open(json_path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    # 先按 fqdn_no 分组
    grouped: dict[str, list] = defaultdict(list)
    for record in raw:
        fqdn_no = record.get("fqdn_no", "").strip()
        if fqdn_no:
            grouped[fqdn_no].append(record)

    # 合并每组
    merged = {fqdn_no: _merge_records(recs) for fqdn_no, recs in grouped.items()}
    print(f"whois: 共加载 {len(raw)} 条原始记录，合并为 {len(merged)} 个域名")
    return merged


CONTACT_FIELDS = ["registrant_country", "registrant_email", "admin_email", "tech_email", "sponsoring"]

def extract_whois_features(fqdn_no: str, whois_dict: dict) -> dict:
    default = {
        "domain_age_days":          -1.0,
        "registration_period_days": -1.0,
        "days_to_expire":           -1.0,
        "update_lag_days":          -1.0,
        "first_update_lag_days":    -1.0,   # 新增
        "ns_count":                  0,
        "ns_provider_diversity":     0,
        "null_contact_count":        len(CONTACT_FIELDS),
        "r_whois_count":             0,
        # 新增
        "whois_record_count":        0,
        "whois_update_count":        0,
        "latest_created_lag_days":  -1.0,
        "registrant_country_count":  0,
        "registrant_email_count":    0,
    }

    rec = whois_dict.get(fqdn_no)
    if rec is None:
        return default

    created        = rec["created"]
    latest_created = rec.get("latest_created")
    expires        = rec["expires"]
    updated        = rec["updated"]
    first_updated  = rec.get("first_updated")
    all_ns         = rec["all_nameservers"]
    all_ws         = rec["all_whoisservers"]

    # ── 时间特征 ──────────────────────────────────────────────
    domain_age_days = (
        (REF_TS_MS - created) / MS_PER_DAY if created is not None else -1.0
    )
    registration_period_days = (
        (expires - created) / MS_PER_DAY
        if (expires is not None and created is not None) else -1.0
    )
    days_to_expire = (
        (expires - REF_TS_MS) / MS_PER_DAY if expires is not None else -1.0
    )
    update_lag_days = (
        (updated - created) / MS_PER_DAY
        if (updated is not None and created is not None) else -1.0
    )
    first_update_lag_days = (
        (first_updated - created) / MS_PER_DAY
        if (first_updated is not None and created is not None) else -1.0
    )
    # 最晚创建日期与最早创建日期的差（衡量数据一致性）
    latest_created_lag_days = (
        (latest_created - created) / MS_PER_DAY
        if (latest_created is not None and created is not None) else -1.0
    )

    ns_count              = len(all_ns)
    ns_provider_diversity = len({_ns_root(ns) for ns in all_ns})

    null_contact_count = sum(1 for f in CONTACT_FIELDS if rec.get(f) is None)

    return {
        "domain_age_days":          round(domain_age_days, 2),
        "registration_period_days": round(registration_period_days, 2),
        "days_to_expire":           round(days_to_expire, 2),
        "update_lag_days":          round(update_lag_days, 2),
        "first_update_lag_days":    round(first_update_lag_days, 2),  # 新增
        "ns_count":                  ns_count,
        "ns_provider_diversity":     ns_provider_diversity,
        "null_contact_count":        null_contact_count,
        "r_whois_count":             len(all_ws),
        # 新增
        "whois_record_count":        rec.get("record_count", 0),
        "whois_update_count":        rec.get("update_count", 0),
        "latest_created_lag_days":   round(latest_created_lag_days, 2),
        "registrant_country_count":  rec.get("registrant_country_count", 0),
        "registrant_email_count":    rec.get("registrant_email_count", 0),
    }


def build_ns_cooccur_graph(whois_dict: dict, all_fqdns: list) -> "csr_matrix":
    """
    构建基于 Nameserver 共享的域名共现稀疏矩阵。

    策略：
      - 将每个 NS 提供商根域（二级域）作为节点媒介。
      - 共享同一 NS 根域的域名对之间加边，权重 = 共享 NS 根域数量。
      - 过滤：某个 NS 根域被超过 1000 个域名使用时视为公共 NS（如 cloudflare），
        不用于建边，避免引入大量噪声。
    """
    fqdn_index = {f: i for i, f in enumerate(all_fqdns)}
    n = len(all_fqdns)

    # 构建 ns_root -> fqdn_no 倒排索引
    ns_root_to_fqdns: dict[str, set] = defaultdict(set)
    for fqdn_no, rec in whois_dict.items():
        if fqdn_no not in fqdn_index:
            continue
        for ns in rec.get("all_nameservers", set()):
            root = _ns_root(ns)
            ns_root_to_fqdns[root].add(fqdn_no)

    mat = lil_matrix((n, n), dtype=np.float32)

    size_dist = Counter(len(v) for v in ns_root_to_fqdns.values())
    print(sorted(size_dist.items()))

    for ns_root, fqdn_set in ns_root_to_fqdns.items():
        fqdn_list = [f for f in fqdn_set if f in fqdn_index]
        # 过滤：太少（单独一个没有共享关系）或太多（公共大 NS，噪声）
        if len(fqdn_list) < 2 or len(fqdn_list) > 100:
            continue
        for i in range(len(fqdn_list)):
            for j in range(i + 1, len(fqdn_list)):
                a = fqdn_index[fqdn_list[i]]
                b = fqdn_index[fqdn_list[j]]
                mat[a, b] += 1.0
                mat[b, a] += 1.0

    print(f"NS 共现图：{n} 个节点，非零元素 {mat.nnz} 个")
    return mat.tocsr()


def compute_ns_spectral_embedding(ns_mat, n_components: int = 8) -> np.ndarray:
    """基于 NS 共现矩阵计算谱嵌入，返回 L2 归一化后的 (n, n_components) 矩阵。"""
    dense = ns_mat.toarray()
    # 加微小自环，防止孤立节点导致奇异矩阵
    dense += np.eye(dense.shape[0], dtype=np.float32) * 1e-6

    embedder = SpectralEmbedding(
        n_components=n_components,
        affinity="precomputed",
        random_state=42,
    )
    embedding = embedder.fit_transform(dense)
    return normalize(embedding, norm="l2")


def extract_whois_all(
    all_fqdns: list,
    json_path: str = "./question/4_question/whois.json",
    cache_dir: str = "./cache",
    ns_components: int = 8,
) -> tuple[dict, dict]:
    """
    一次性完成：
      1. 读取并合并 whois.json
      2. 构建 NS 共享图 + 谱嵌入（带缓存）
      3. 返回
         - whois_feat_dict : {fqdn_no: {feature_name: value, ...}}
         - ns_embed_dict   : {fqdn_no: np.ndarray(ns_components,)}

    在 build_feature_vector 中调用，将两部分特征拼入 combined 即可。
    """
    os.makedirs(cache_dir, exist_ok=True)
    ns_mat_path   = os.path.join(cache_dir, "ns_cooccur_mat.npz")
    ns_emb_path   = os.path.join(cache_dir, "ns_embedding.npy")
    ns_order_path = os.path.join(cache_dir, "ns_fqdn_order.txt")

    # 1. 读取 whois
    whois_dict = read_whois_json(json_path)

    # 2. NS 共享图 / 谱嵌入（缓存判断）
    cache_ok = (
        os.path.exists(ns_mat_path)
        and os.path.exists(ns_emb_path)
        and os.path.exists(ns_order_path)
    )
    if cache_ok:
        with open(ns_order_path, "r", encoding="utf-8") as f:
            cached_fqdns = [l.strip() for l in f]
        if cached_fqdns != all_fqdns:
            print("NS 缓存域名顺序不一致，重新构建...")
            cache_ok = False

    if cache_ok:
        print("加载 NS 谱嵌入缓存...")
        ns_mat       = load_npz(ns_mat_path)
        ns_embedding = np.load(ns_emb_path)
    else:
        print("构建 NS 共现图...")
        ns_mat = build_ns_cooccur_graph(whois_dict, all_fqdns)
        print("计算 NS 谱嵌入...")
        ns_embedding = compute_ns_spectral_embedding(ns_mat, n_components=ns_components)
        save_npz(ns_mat_path, ns_mat)
        np.save(ns_emb_path, ns_embedding)
        with open(ns_order_path, "w", encoding="utf-8") as f:
            f.write("\n".join(all_fqdns))
        print(f"NS 缓存已保存至 {cache_dir}")

    # 3. 打包返回
    whois_feat_dict = {
        fqdn_no: extract_whois_features(fqdn_no, whois_dict)
        for fqdn_no in all_fqdns
    }
    ns_embed_dict = {
        fqdn_no: ns_embedding[i]
        for i, fqdn_no in enumerate(all_fqdns)
    }

    return whois_feat_dict, ns_embed_dict


def build_feature_vector(
        fqdn_no,
        fqdn_dict,
        access_dict,
        flint_dict,
        ip_dict,
        ipv6_dict,
        access_ip_map=None,
        whois_feat_dict=None,
        ns_embed_dict=None,
        ns_components=8,
        malicious_fqdns=None,   # 新增：已知恶意域名集合
        malicious_ips=None,     # 新增：恶意域名解析到的IP集合
        pointed_by_malicious=None,  # 新增：被恶意域名CNAME指向的域名集合
):
    domainname_features = extract_domainname_features(fqdn_dict.get(fqdn_no, ''))

    # 传入客户端IP信息
    access_ip_info = access_ip_map.get(fqdn_no, {}) if access_ip_map else {}
    access_features = extract_access_features(
        access_dict.get(fqdn_no, []),
        access_ip_info=access_ip_info,
    )

    flint_features = extract_flint_features(
        fqdn_no,
        flint_dict.get(fqdn_no, []),
        flint_dict,
        malicious_fqdns=malicious_fqdns,
        malicious_ips=malicious_ips,
    )

    # 被恶意域名指向特征（预计算后直接注入）
    if pointed_by_malicious is not None:
        flint_features['is_pointed_by_malicious'] = int(fqdn_no in pointed_by_malicious)

    ip_features = extract_ip_features(fqdn_no, flint_dict, ip_dict, ipv6_dict, access_ip_map)

    combined = {}
    combined.update(domainname_features)
    combined.update(access_features)
    combined.update(flint_features)
    combined.update(ip_features)

    # whois 数值特征
    if whois_feat_dict is not None:
        whois_feats = whois_feat_dict.get(
            fqdn_no, extract_whois_features(fqdn_no, {})
        )
    else:
        whois_feats = extract_whois_features(fqdn_no, {})
    combined.update(whois_feats)

    # NS 谱嵌入特征
    ns_vec = (
        ns_embed_dict.get(fqdn_no, [0.0] * ns_components)
        if ns_embed_dict is not None
        else [0.0] * ns_components
    )
    for dim, val in enumerate(ns_vec):
        combined[f'ns_embed_{dim}'] = float(val)

    return combined


def extract_all_features(cache_dir="./cache",
                         label_path="./question/4_question/label.csv"):
    os.makedirs(cache_dir, exist_ok=True)

    mat_path        = os.path.join(cache_dir, "cooccur_mat.npz")
    emb_path        = os.path.join(cache_dir, "embedding.npy")
    fqdn_order_path = os.path.join(cache_dir, "fqdn_order.txt")

    fqdn_dict                  = read_fqdn_csv("./question/4_question/fqdn.csv")
    access_dict, access_ip_map = read_access_csv("./question/4_question/access.csv")
    flint_dict                 = read_flint_csv("./question/4_question/flint.csv")
    ip_dict                    = read_ip_csv("./question/4_question/ip.csv")
    ipv6_dict                  = read_ipv6_csv("./question/4_question/ipv6.csv")

    all_fqdns = list(fqdn_dict.keys())

    # ── 预计算恶意关联集合 ───────────────────────────────────
    # 读取已知恶意域名集合
    malicious_fqdns = set()
    if os.path.exists(label_path):
        with open(label_path, "r", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                malicious_fqdns.add(row["fqdn_no"].strip())

    # 恶意域名解析到的所有IP（A+AAAA）
    malicious_ips = set()
    for fqdn_no in malicious_fqdns:
        for item in flint_dict.get(fqdn_no, []):
            if item['flintType'] in (1, 28):
                malicious_ips.add(item['encoded_value'])

    # 被恶意域名通过CNAME指向的域名集合
    pointed_by_malicious = set()
    for fqdn_no in malicious_fqdns:
        for item in flint_dict.get(fqdn_no, []):
            if item['flintType'] == 5:
                pointed_by_malicious.add(item['encoded_value'])

    print(f"恶意域名集合: {len(malicious_fqdns)} 个")
    print(f"恶意IP集合: {len(malicious_ips)} 个")
    print(f"被恶意域名指向的域名集合: {len(pointed_by_malicious)} 个")

    # ── IP 共现图 + 谱嵌入（带缓存）────────────────────────
    cache_exists = (
        os.path.exists(mat_path)
        and os.path.exists(emb_path)
        and os.path.exists(fqdn_order_path)
    )
    if cache_exists:
        print("加载缓存中的 IP 共现矩阵和谱嵌入...")
        cooccur_mat = load_npz(mat_path)
        embedding   = np.load(emb_path)
        with open(fqdn_order_path, "r", encoding="utf-8") as f:
            cached_fqdns = [line.strip() for line in f]
        if cached_fqdns != all_fqdns:
            print("警告：IP 缓存域名顺序与当前 fqdn.csv 不一致，重新构建...")
            cache_exists = False

    if not cache_exists:
        print("构建 IP 共现图...")
        cooccur_mat = build_cooccur_graph(flint_dict, access_ip_map, all_fqdns)
        print("计算 IP 谱嵌入...")
        embedding = compute_spectral_embedding(cooccur_mat, n_components=8)
        save_npz(mat_path, cooccur_mat)
        np.save(emb_path, embedding)
        with open(fqdn_order_path, "w", encoding="utf-8") as f:
            f.write("\n".join(all_fqdns))
        print(f"IP 图缓存已保存至 {cache_dir}")

    fqdn_embed = {fqdn: embedding[i] for i, fqdn in enumerate(all_fqdns)}

    # ── whois 特征 + NS 谱嵌入（带缓存）────────────────────
    NS_COMPONENTS = 8
    whois_feat_dict, ns_embed_dict = extract_whois_all(
        all_fqdns=all_fqdns,
        json_path="./question/4_question/whois.json",
        cache_dir=cache_dir,
        ns_components=NS_COMPONENTS,
    )

    # ── 构建特征矩阵 ────────────────────────────────────────
    X, fqdn_keys = [], []
    for fqdn_no in all_fqdns:
        combined = build_feature_vector(
            fqdn_no,
            fqdn_dict,
            access_dict,
            flint_dict,
            ip_dict,
            ipv6_dict,
            access_ip_map=access_ip_map,
            whois_feat_dict=whois_feat_dict,
            ns_embed_dict=ns_embed_dict,
            ns_components=NS_COMPONENTS,
            malicious_fqdns=malicious_fqdns,
            malicious_ips=malicious_ips,
            pointed_by_malicious=pointed_by_malicious,
        )

        for dim, val in enumerate(fqdn_embed.get(fqdn_no, [0.0] * 8)):
            combined[f'graph_embed_{dim}'] = float(val)

        X.append(combined)
        fqdn_keys.append(fqdn_no)

    feature_names = list(X[0].keys())
    X_matrix = np.array(
        [[row[k] for k in feature_names] for row in X],
        dtype=np.float32,
    )

    print(f"全量域名数: {len(fqdn_keys)}, 特征维度: {len(feature_names)}")
    return X_matrix, fqdn_keys, feature_names


if __name__ == "__main__":
    X_matrix, fqdn_keys, feature_names = extract_all_features()
