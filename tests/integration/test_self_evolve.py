from precise_attribution.evolution.engine import run_self_evolve


class FakeLLM:
    def chat(self, messages, **kwargs):
        system = messages[0]["content"]
        if "改写为适合知识库检索" in system:
            return "线粒体 ATP 功能"
        if "判断给定证据是否足以回答" in system:
            return '{"sufficient":true,"reason":"evidence"}'
        if "答案验证模块" in system:
            return '{"supported":true,"complete":true,"issues":[]}'
        if "只依据提供的证据回答" in system:
            return "线粒体通过细胞呼吸产生 ATP，为细胞提供能量。"
        if "Proposer" in system:
            return '{"rationale":"test","candidate_prompt":"只依据证据回答，不添加无关内容。","expected_effect":"提高准确性"}'
        if "Critic" in system:
            return '{"approve_for_testing":true,"risks":[],"reason":"可进入测试"}'
        return "{}"


def test_self_evolve_writes_report_and_keeps_ood_out_of_gate(tmp_path):
    config = {
        "llm": {"model": "fake"},
        "harness": {"answer_system_prompt": "只依据证据回答。"},
        "evolution": {"minimum_gain": 0.0},
        "tasks": {
            "development": [{"id": "d1", "question": "线粒体功能？", "expected_answer": "线粒体通过细胞呼吸产生 ATP，为细胞提供能量。", "documents": [{"id": "x", "title": "线粒体", "text": "线粒体通过细胞呼吸产生 ATP，为细胞提供能量。"}]}],
            "ood": [{"id": "ood", "question": "光合作用？", "expected_answer": "光合作用释放氧气。", "documents": [{"id": "y", "title": "光合作用", "text": "光合作用释放氧气。"}]}],
        },
    }
    report = run_self_evolve(config, FakeLLM(), tmp_path)
    assert (tmp_path / "self_evolve_report.json").exists()
    assert report["ood_report_only"]["used_for_acceptance"] is False
    assert report["acceptance"]["accepted"] is True
