from __future__ import annotations

import pytest

from core.errors import PlanningError
from core.llm.base import LLMResponse, Message
from core.llm.fake import FakeProvider
from core.planner.llm_planner import LLMPlanner
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
    from core.llm.null import NullProvider
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
