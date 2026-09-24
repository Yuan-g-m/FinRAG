# -*- coding: utf-8 -*-
"""FinRAG 运行时模型下载脚本。

公开模型从 HuggingFace / ModelScope 下载到 rag_qa/models/（可用
FINRAG_MODEL_ROOT 覆盖目标目录）。自定义分类器 bert_query_classifier_finance
不在下载范围，需本地训练或从原部署环境复制。
"""
import os
import sys


def models_root():
    root = os.getenv("FINRAG_MODEL_ROOT")
    if root:
        return root
    return os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "rag_qa", "models")


def download_hf(repo_id, target):
    from huggingface_hub import snapshot_download

    print(f"[HF] {repo_id} -> {target}")
    snapshot_download(repo_id=repo_id, local_dir=target)


def download_ms(repo_id, target):
    from modelscope import snapshot_download

    print(f"[MS] {repo_id} -> {target}")
    snapshot_download(repo_id, local_dir=target)


def main():
    root = models_root()
    os.makedirs(root, exist_ok=True)

    jobs = [
        ("hf", "google-bert/bert-base-chinese", "bert-base-chinese"),
        ("hf", "BAAI/bge-m3", "bge-m3"),
        ("hf", "BAAI/bge-reranker-large", "bge-reranker-large"),
        ("ms", "iic/nlp_bert_document-segmentation_chinese-base", "nlp_bert_document-segmentation_chinese-base"),
    ]

    failed = []
    for kind, repo, name in jobs:
        target = os.path.join(root, name)
        if os.path.isdir(target) and os.listdir(target):
            print(f"[SKIP] {target} 已存在")
            continue
        try:
            if kind == "hf":
                download_hf(repo, target)
            else:
                download_ms(repo, target)
        except Exception as exc:  # noqa: BLE001
            print(f"[FAIL] {repo}: {exc}", file=sys.stderr)
            failed.append(repo)

    print("\n目标目录:", root)
    if failed:
        print("以下模型下载失败，请重试或手动下载:", failed, file=sys.stderr)
        sys.exit(1)
    print("公开模型下载完成。自定义分类器 bert_query_classifier_finance 请另行训练或复制。")


if __name__ == "__main__":
    main()
