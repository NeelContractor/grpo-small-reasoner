from grpo_reasoner.metrics import accuracy, summarize


def test_summarize_counts_correct_records():
    records = [{"correct": True}, {"correct": False}, {"correct": True}]
    assert summarize(records) == {"total": 3, "correct": 2, "accuracy": 2 / 3}


def test_empty_results():
    assert summarize([]) == {"total": 0, "correct": 0, "accuracy": 0.0}
    assert accuracy([]) == 0.0
