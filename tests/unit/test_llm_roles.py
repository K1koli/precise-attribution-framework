from precise_attribution.evolution.metrics import token_f1
from precise_attribution.evolution.roles import Critic, Proposer
from precise_attribution.harness.modules import HarnessConfig, RAGHarness


class FakeLLM:
    def __init__(self):
        self.responses = []

    def chat(self, messages, **kwargs):
        if self.responses:
            return self.responses.pop(0)
        system = messages[0]["content"]
        if "改写为适合知识库检索" in system:
            return "线粒体 主要功能 ATP 细胞呼吸"
        if "判断给定证据是否足以回答" in system:
            return '{"sufficient": true, "reason": "证据包含答案"}'
        if "答案验证模块" in system:
            return '{"supported": true, "complete": true, "issues": []}'
        if "只依据提供的证据回答" in system:
            return "线粒体通过细胞呼吸产生 ATP，为细胞活动提供能量。"
        if "Proposer" in system:
            return '{"rationale":"强化证据约束","candidate_prompt":"只依据证据回答。","expected_effect":"减少无依据内容"}'
        if "Critic" in system:
            return '{"approve_for_testing":true,"risks":[],"reason":"可测试"}'
        return "{}"


def test_rag_harness_executes_seven_observable_modules():
    llm = FakeLLM()
    result = RAGHarness(llm, HarnessConfig()).run({
        "id": "t1", "question": "线粒体的主要功能是什么？",
        "documents": [{"id": "d1", "title": "线粒体", "text": "线粒体通过细胞呼吸产生 ATP，为细胞活动提供能量。"}],
    })
    assert result["answer"]
    assert result["trace"]["module_order"] == ["QueryProcessor", "Retriever", "Ranker", "ContextBuilder", "ReasoningController", "AnswerGenerator", "Verifier"]
    assert result["verification"]["supported"] is True


def test_proposer_and_critic_json_roles():
    llm = FakeLLM()
    proposal = Proposer(llm).propose("old", "failures", [])
    review = Critic(llm).review("old", proposal)
    assert proposal["candidate_prompt"]
    assert review["approve_for_testing"] is True


def test_token_f1():
    assert token_f1("TCP 提供可靠传输", "TCP 提供可靠传输") == 1.0
    assert 0.0 < token_f1("TCP 提供可靠传输", "TCP 不可靠") < 1.0
