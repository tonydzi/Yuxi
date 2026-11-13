import pytest

from yuxi.modules.knowledge.evaluation.metrics import EvaluationMetricsCalculator


def test_retrieval_metrics_use_metadata_chunk_id():
    retrieved_chunks = [
        {"metadata": {"chunk_id": "chunk_a"}},
        {"metadata": {"chunk_id": "chunk_b"}},
    ]

    metrics = EvaluationMetricsCalculator.calculate_retrieval_metrics(retrieved_chunks, ["chunk_b", "chunk_c"], k_values=[1, 3])

    assert metrics["recall@1"] == 0.0
    assert metrics["recall@3"] == 0.5
    assert metrics["f1@3"] == pytest.approx(0.4)


def test_retrieval_metrics_ignore_generation_suffix_after_reindex():
    # 文件重新入库后 chunk_id 的 -gN 后缀会变，基准里存的旧 id 仍应命中
    retrieved_chunks = [
        {"metadata": {"chunk_id": "file_1_chunk_0-g3"}},
        {"metadata": {"chunk_id": "file_1_chunk_1-g3"}},
    ]

    metrics = EvaluationMetricsCalculator.calculate_retrieval_metrics(
        retrieved_chunks, ["file_1_chunk_1-g2"], k_values=[1, 3]
    )

    assert metrics["recall@1"] == 0.0
    assert metrics["recall@3"] == 1.0


def test_retrieval_metrics_generation_suffix_does_not_merge_different_chunks():
    retrieved_chunks = [{"metadata": {"chunk_id": "file_1_chunk_1-g3"}}]

    metrics = EvaluationMetricsCalculator.calculate_retrieval_metrics(
        retrieved_chunks, ["file_1_chunk_2-g3"], k_values=[3]
    )

    assert metrics["recall@3"] == 0.0


def test_overall_score_uses_answer_accuracy_when_available():
    # 有答案准确率时，综合得分取各题 score 的平均，且与检索指标无关
    retrieval = [{"recall@10": 1.0, "f1@10": 0.2}, {"recall@10": 0.0, "f1@10": 0.0}]
    answers = [{"score": 1.0}, {"score": 0.0}, {"score": 1.0}, {"score": 1.0}]

    score = EvaluationMetricsCalculator.calculate_overall_score(retrieval, answers)

    assert score == 0.75


def test_overall_score_uses_recall_at_10_without_answers():
    # 无答案准确率时，综合得分取各题 recall@10 的平均，不受 f1/其它 k 影响
    retrieval = [
        {"recall@1": 0.0, "recall@5": 0.5, "recall@10": 0.8, "f1@10": 0.1},
        {"recall@1": 1.0, "recall@5": 1.0, "recall@10": 0.4, "f1@10": 0.9},
    ]

    score = EvaluationMetricsCalculator.calculate_overall_score(retrieval, [])

    assert score == pytest.approx(0.6)


def test_overall_score_returns_none_without_any_metrics():
    score = EvaluationMetricsCalculator.calculate_overall_score([], [])

    assert score is None
