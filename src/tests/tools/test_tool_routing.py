"""Tests for task routing decisions in the atomizer, planner, and URL extraction."""

from __future__ import annotations

from src.components.atomizer import DefaultAtomizer
from src.components.executors.retrieve import _extract_urls
from src.components.planner import DefaultPlanner
from src.core.models import NodeType, Task, TaskType


# ==============================================================================
# DefaultAtomizer.decide() routing decisions
# ==============================================================================


class TestAtomizerRouting:
    """Verify how the atomizer routes goals to PLAN vs EXECUTE node types."""

    def test_simple_atomic_task_returns_execute(self) -> None:
        """A simple, single-deliverable goal should be EXECUTE."""
        atomizer = DefaultAtomizer()
        task = Task(id="t1", goal="Summarize this paragraph", task_type=TaskType.GENERAL)

        decision = atomizer.decide(task)

        assert decision.node_type == NodeType.EXECUTE

    def test_multi_step_keyword_triggers_plan(self) -> None:
        """'compare A and B' implies multi-step sequencing → PLAN."""
        atomizer = DefaultAtomizer()
        task = Task(id="t2", goal="Compare A and B", task_type=TaskType.GENERAL)

        decision = atomizer.decide(task)

        assert decision.node_type == NodeType.PLAN

    def test_search_keyword_triggers_plan(self) -> None:
        """'search for population' triggers the search/fetch check → PLAN."""
        atomizer = DefaultAtomizer()
        task = Task(id="t3", goal="search for population of France", task_type=TaskType.GENERAL)

        decision = atomizer.decide(task)

        assert decision.node_type == NodeType.PLAN

    def test_url_keyword_triggers_plan(self) -> None:
        """'http://' in goal triggers the URL check → PLAN."""
        atomizer = DefaultAtomizer()
        task = Task(id="t4", goal="check http://example.com", task_type=TaskType.GENERAL)

        decision = atomizer.decide(task)

        assert decision.node_type == NodeType.PLAN

    def test_lookup_keyword_triggers_plan(self) -> None:
        """'look up capital' triggers the search/fetch check → PLAN."""
        atomizer = DefaultAtomizer()
        task = Task(id="t5", goal="look up capital of France", task_type=TaskType.GENERAL)

        decision = atomizer.decide(task)

        assert decision.node_type == NodeType.PLAN

    def test_www_keyword_triggers_plan(self) -> None:
        """'www.' in goal triggers the URL check → PLAN."""
        atomizer = DefaultAtomizer()
        task = Task(id="t6", goal="tell me what is on www.example.com", task_type=TaskType.GENERAL)

        decision = atomizer.decide(task)

        assert decision.node_type == NodeType.PLAN

    def test_retrieve_task_type_triggers_plan(self) -> None:
        """A task with TaskType.RETRIEVE should always trigger PLAN."""
        atomizer = DefaultAtomizer()
        task = Task(id="t7", goal="gather latest news", task_type=TaskType.RETRIEVE)

        decision = atomizer.decide(task)

        assert decision.node_type == NodeType.PLAN

    def test_plain_general_task_returns_execute(self) -> None:
        """A plain writing/thinking goal with no routing keywords should be EXECUTE."""
        atomizer = DefaultAtomizer()
        task = Task(id="t8", goal="write a poem about a cat", task_type=TaskType.GENERAL)

        decision = atomizer.decide(task)

        assert decision.node_type == NodeType.EXECUTE

    def test_find_keyword_triggers_plan(self) -> None:
        """'find something' triggers the search/fetch check → PLAN
        (note: atomizer does NOT have 'find ' in its keyword list — this should
         NOT trigger PLAN on the atomizer side; the atomizer only plans for
         'search', 'http', 'www.', '.com', 'url', 'look up')."""
        atomizer = DefaultAtomizer()
        task = Task(id="t9", goal="find the population of Canada", task_type=TaskType.GENERAL)

        decision = atomizer.decide(task)

        # The atomizer does NOT list "find " in its keyword check, so this
        # should remain EXECUTE at the atomizer level.
        assert decision.node_type == NodeType.EXECUTE

    def test_dotcom_keyword_triggers_plan(self) -> None:
        """'.com' in goal triggers the URL check → PLAN."""
        atomizer = DefaultAtomizer()
        task = Task(id="t10", goal="check news.com for updates", task_type=TaskType.GENERAL)

        decision = atomizer.decide(task)

        assert decision.node_type == NodeType.PLAN


# ==============================================================================
# DefaultPlanner._plan_general() subtask routing
# ==============================================================================


class TestPlannerRouting:
    """Verify how the planner routes the 'inputs' subtask to RETRIEVE vs THINK."""

    def test_retrieve_routing_for_search_keyword(self) -> None:
        """'search' in goal -> inputs subtask should be RETRIEVE."""
        planner = DefaultPlanner()
        task = Task(id="p1", goal="search the population of France", task_type=TaskType.GENERAL)

        plan = planner.plan(task)

        inputs_subtask = plan.subtasks[0]
        assert inputs_subtask.task_type == TaskType.RETRIEVE

    def test_think_routing_for_plain_goal(self) -> None:
        """A plain goal with no search keywords -> inputs subtask should be THINK."""
        planner = DefaultPlanner()
        task = Task(id="p2", goal="write a poem about a cat", task_type=TaskType.GENERAL)

        plan = planner.plan(task)

        inputs_subtask = plan.subtasks[0]
        assert inputs_subtask.task_type == TaskType.THINK

    def test_retrieve_routing_for_url(self) -> None:
        """A goal containing a URL -> inputs subtask should be RETRIEVE."""
        planner = DefaultPlanner()
        task = Task(id="p3", goal="check http://example.com/data", task_type=TaskType.GENERAL)

        plan = planner.plan(task)

        inputs_subtask = plan.subtasks[0]
        assert inputs_subtask.task_type == TaskType.RETRIEVE

    def test_retrieve_routing_for_lookup(self) -> None:
        """'look up' in goal -> inputs subtask should be RETRIEVE."""
        planner = DefaultPlanner()
        task = Task(id="p4", goal="look up the capital of France", task_type=TaskType.GENERAL)

        plan = planner.plan(task)

        inputs_subtask = plan.subtasks[0]
        assert inputs_subtask.task_type == TaskType.RETRIEVE

    def test_retrieve_routing_for_who_is(self) -> None:
        """'who is' in goal -> inputs subtask should be RETRIEVE."""
        planner = DefaultPlanner()
        task = Task(id="p5", goal="who is the president of France", task_type=TaskType.GENERAL)

        plan = planner.plan(task)

        inputs_subtask = plan.subtasks[0]
        assert inputs_subtask.task_type == TaskType.RETRIEVE

    def test_retrieve_routing_for_dotcom(self) -> None:
        """'.com' in goal -> inputs subtask should be RETRIEVE."""
        planner = DefaultPlanner()
        task = Task(id="p6", goal="check news.com/article for updates", task_type=TaskType.GENERAL)

        plan = planner.plan(task)

        inputs_subtask = plan.subtasks[0]
        assert inputs_subtask.task_type == TaskType.RETRIEVE

    def test_retrieve_task_type_uses_plan_retrieve(self) -> None:
        """A task with TaskType.RETRIEVE uses _plan_retrieve, which has the
        evidence subtask as TaskType.RETRIEVE."""
        planner = DefaultPlanner()
        task = Task(id="p7", goal="find background facts", task_type=TaskType.RETRIEVE)

        plan = planner.plan(task)

        # _plan_retrieve produces: queries (THINK), evidence (RETRIEVE), summary (WRITE)
        assert len(plan.subtasks) == 3
        assert plan.subtasks[0].task_type == TaskType.THINK  # queries
        assert plan.subtasks[1].task_type == TaskType.RETRIEVE  # evidence
        assert plan.subtasks[2].task_type == TaskType.WRITE  # summary

    def test_write_task_type_uses_plan_write(self) -> None:
        """A task with TaskType.WRITE uses _plan_write, which has the foundation
        subtask as THINK and the development subtask as WRITE."""
        planner = DefaultPlanner()
        task = Task(id="p8", goal="write a short story", task_type=TaskType.WRITE)

        plan = planner.plan(task)

        # _plan_write produces: foundation (THINK), development (WRITE), synthesis (WRITE)
        assert len(plan.subtasks) == 3
        assert plan.subtasks[0].task_type == TaskType.THINK  # foundation
        assert plan.subtasks[1].task_type == TaskType.WRITE  # development
        assert plan.subtasks[2].task_type == TaskType.WRITE  # synthesis


# ==============================================================================
# _extract_urls() from the retrieve executor
# ==============================================================================


class TestExtractUrls:
    """Verify URL extraction behaviour from goal text."""

    def test_extract_urls_from_goal(self) -> None:
        """Standard http:// and https:// URLs should be extracted."""
        urls = _extract_urls("Check http://example.com and https://secure.example.org/page")
        assert urls == ["http://example.com", "https://secure.example.org/page"]

    def test_extract_urls_from_bare_domain(self) -> None:
        """Bare domain paths (without a protocol) should be extracted."""
        urls = _extract_urls("Read en.wikipedia.org/wiki/Foo for details")
        assert "en.wikipedia.org/wiki/Foo" in urls

    def test_extract_urls_empty(self) -> None:
        """Text without any URLs should return an empty list."""
        urls = _extract_urls("This is a plain sentence with no links.")
        assert urls == []

    def test_extract_urls_duplicates(self) -> None:
        """Duplicate URLs should be deduplicated, preserving first-occurrence order."""
        urls = _extract_urls(
            "Visit http://example.com and also http://example.com again"
        )
        assert urls == ["http://example.com"]

    def test_extract_urls_multiple(self) -> None:
        """Multiple URLs should be returned in order of first appearance."""
        urls = _extract_urls(
            "First https://alpha.com, then http://beta.org, then https://gamma.net"
        )
        assert urls == ["https://alpha.com", "http://beta.org", "https://gamma.net"]

    def test_extract_urls_no_false_positives(self) -> None:
        """Regular text without any URL-like patterns should return empty."""
        urls = _extract_urls(
            "Once upon a time, there was a developer who wrote clean code."
        )
        assert urls == []
