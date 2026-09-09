from __future__ import annotations

import pytest

from core.errors import PlanningError
from core.llm.base import LLMResponse, Message
from core.llm.fake import FakeProvider
from core.llm.null import NullProvider
from core.planner.llm_planner import LLMPlanner
from core.planner.plan import PlanStep
from core.planner.rule_based import RuleBasedPlanner


def test_rule_based_recognizes_list_directory(registry):
    planner = RuleBasedPlanner()
    plan = planner.create_plan("list the files in ./docs", registry)
    assert plan.steps[0].tool_name == "fs_list_directory"
    assert plan.steps[0].args["path"] == "./docs"


def test_rule_based_recognizes_finance_spend(registry):
    planner = RuleBasedPlanner()
    plan = planner.create_plan("I spent ₹340 on lunch", registry)
    assert plan.steps[0].tool_name == "finance_add_transaction"


def test_rule_based_recognizes_finance_query(registry):
    planner = RuleBasedPlanner()
    plan = planner.create_plan("How much did I spend on food this month?", registry)
    assert plan.steps[0].tool_name == "finance_query"


def test_rule_based_raises_on_unrecognized_request(registry):
    planner = RuleBasedPlanner()
    with pytest.raises(PlanningError):
        planner.create_plan("compose a symphony in D minor", registry)


def test_llm_planner_builds_valid_plan(registry):
    response = LLMResponse(content='{"rationale": "test", "steps": ['
                            '{"tool_name": "fs_list_directory", "args": {"path": "."}, '
                            '"description": "list"}]}')
    provider = FakeProvider(responses=[response])
    planner = LLMPlanner(provider)
    plan = planner.create_plan("list the current directory", registry)
    assert plan.steps[0].tool_name == "fs_list_directory"


def test_llm_planner_rejects_unknown_tool_and_falls_back(registry):
    response = LLMResponse(content='{"rationale": "x", "steps": ['
                            '{"tool_name": "not_a_real_tool", "args": {}}]}')
    provider = FakeProvider(responses=[response])
    fallback = RuleBasedPlanner()
    planner = LLMPlanner(provider, fallback=fallback)
    plan = planner.create_plan("I spent 50 on tea", registry)
    assert plan.steps[0].tool_name == "finance_add_transaction"


def test_llm_planner_rejects_unknown_tool_raises_without_fallback(registry):
    response = LLMResponse(content='{"rationale": "x", "steps": ['
                            '{"tool_name": "not_a_real_tool", "args": {}}]}')
    provider = FakeProvider(responses=[response])
    planner = LLMPlanner(provider)
    with pytest.raises(PlanningError):
        planner.create_plan("anything", registry)


def test_llm_planner_falls_back_when_llm_unavailable(registry):
    planner = LLMPlanner(NullProvider(), fallback=RuleBasedPlanner())
    plan = planner.create_plan("I spent 50 on tea", registry)
    assert plan.steps[0].tool_name == "finance_add_transaction"


def test_llm_planner_validates_args_against_schema(registry):
    # missing required 'text' arg for finance_add_transaction
    response = LLMResponse(content='{"rationale": "x", "steps": ['
                            '{"tool_name": "finance_add_transaction", "args": {}}]}')
    provider = FakeProvider(responses=[response])
    planner = LLMPlanner(provider)
    with pytest.raises(PlanningError):
        planner.create_plan("anything", registry)


# -- LLMPlanner.revise_step: LLM-driven correction --

def test_revise_step_returns_corrected_args(registry):
    original = PlanStep(tool_name="finance_add_transaction", args={"text": ""},
                         description="log a spend")
    response = LLMResponse(content='{"args": {"text": "I spent 50 on tea"}}')
    provider = FakeProvider(responses=[response])
    planner = LLMPlanner(provider)

    revised = planner.revise_step(original, ["tool call did not succeed: empty text"], registry)

    assert revised.args == {"text": "I spent 50 on tea"}
    # tool_name/description/expected carry over unchanged — a revision
    # can't smuggle in a different, unverified tool call.
    assert revised.tool_name == original.tool_name
    assert revised.description == original.description
    assert revised.expected == original.expected


def test_revise_step_prompt_includes_tool_name_args_and_reasons(registry):
    original = PlanStep(tool_name="finance_add_transaction", args={"text": "bad"})
    response = LLMResponse(content='{"args": {"text": "fixed"}}')
    provider = FakeProvider(responses=[response])
    planner = LLMPlanner(provider)

    planner.revise_step(original, ["it broke"], registry)

    sent = provider.calls[0][0].content
    assert "finance_add_transaction" in sent
    assert "bad" in sent
    assert "it broke" in sent


def test_revise_step_raises_on_unknown_tool(registry):
    planner = LLMPlanner(FakeProvider())
    step = PlanStep(tool_name="not_a_real_tool", args={})
    with pytest.raises(PlanningError):
        planner.revise_step(step, ["failed"], registry)


def test_revise_step_raises_when_llm_returns_no_json(registry):
    provider = FakeProvider(responses=[LLMResponse(content="sorry, I can't help")])
    planner = LLMPlanner(provider)
    step = PlanStep(tool_name="finance_add_transaction", args={"text": "x"})
    with pytest.raises(PlanningError):
        planner.revise_step(step, ["failed"], registry)


def test_revise_step_raises_when_response_has_no_args_key(registry):
    provider = FakeProvider(responses=[LLMResponse(content='{"rationale": "oops"}')])
    planner = LLMPlanner(provider)
    step = PlanStep(tool_name="finance_add_transaction", args={"text": "x"})
    with pytest.raises(PlanningError):
        planner.revise_step(step, ["failed"], registry)


def test_revise_step_raises_when_revised_args_fail_validation(registry):
    # finance_add_transaction requires "text"; the LLM's revision omits it.
    provider = FakeProvider(responses=[LLMResponse(content='{"args": {}}')])
    planner = LLMPlanner(provider)
    step = PlanStep(tool_name="finance_add_transaction", args={"text": "x"})
    with pytest.raises(PlanningError):
        planner.revise_step(step, ["failed"], registry)


def test_revise_step_raises_when_llm_unavailable(registry):
    planner = LLMPlanner(NullProvider())
    step = PlanStep(tool_name="finance_add_transaction", args={"text": "x"})
    with pytest.raises(PlanningError):
        planner.revise_step(step, ["failed"], registry)


def test_rule_based_planner_has_no_revise_step_capability():
    # AgentLoop._revise() checks for this with hasattr — RuleBasedPlanner
    # not having it is exactly what makes it fall back to identical retry.
    assert not hasattr(RuleBasedPlanner(), "revise_step")
