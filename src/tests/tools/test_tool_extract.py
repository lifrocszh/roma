"""Tests for WebSearchToolkit._extract() via the Tavily API.

All tests require a real TAVILY_API_KEY environment variable and are skipped
when it is not set, using ``@pytest.mark.skipif(not os.getenv("TAVILY_API_KEY"))``.
"""

from __future__ import annotations

import os

import pytest

from src.core.registry import WebSearchToolkit


# ==============================================================================
# Integration tests (skipped without TAVILY_API_KEY)
# ==============================================================================


class TestExtractIntegration:
    """End-to-end tests for ``WebSearchToolkit._extract()`` with a live API key."""

    # ------------------------------------------------------------------
    # test_extract_returns_raw_content
    # ------------------------------------------------------------------

    @pytest.mark.skipif(not os.getenv("TAVILY_API_KEY"), reason="requires TAVILY_API_KEY env var")
    def test_extract_returns_raw_content(self) -> None:  # pragma: no cover
        """Extract from ``https://example.com`` should return ok=True
        and contain the text *Example Domain* in the output."""
        tool = WebSearchToolkit(api_key=os.environ["TAVILY_API_KEY"])
        result = tool.invoke(action="extract", url="https://example.com")

        assert result.ok is True, f"expected ok=True, got ok={result.ok}"
        assert "Example Domain" in result.output, (
            f"expected output to contain 'Example Domain', got:\n{result.output[:500]}"
        )

    # ------------------------------------------------------------------
    # test_extract_normalises_url
    # ------------------------------------------------------------------

    @pytest.mark.skipif(not os.getenv("TAVILY_API_KEY"), reason="requires TAVILY_API_KEY env var")
    def test_extract_normalises_url(self) -> None:  # pragma: no cover
        """Extract from ``example.com`` (no scheme) should still succeed
        because ``_extract`` prepends ``https://`` before the API call."""
        tool = WebSearchToolkit(api_key=os.environ["TAVILY_API_KEY"])
        result = tool.invoke(action="extract", url="example.com")

        assert result.ok is True, f"expected ok=True for normalised url, got ok={result.ok}"
        assert "Example Domain" in result.output, (
            f"expected output to contain 'Example Domain', got:\n{result.output[:500]}"
        )
        # Verify the metadata reflects the normalised URL
        assert result.metadata.get("url") == "https://example.com", (
            f"expected metadata url to be 'https://example.com', "
            f"got {result.metadata.get('url')!r}"
        )

    # ------------------------------------------------------------------
    # test_extract_fails_for_nonexistent_domain
    # ------------------------------------------------------------------

    @pytest.mark.skipif(not os.getenv("TAVILY_API_KEY"), reason="requires TAVILY_API_KEY env var")
    def test_extract_fails_for_nonexistent_domain(self) -> None:  # pragma: no cover
        """Extract from a non-existent domain should return ok=False."""
        tool = WebSearchToolkit(api_key=os.environ["TAVILY_API_KEY"])
        result = tool.invoke(
            action="extract",
            url="https://this-domain-does-not-exist-12345.com",
        )

        assert result.ok is False, (
            f"expected ok=False for a non-existent domain, got ok={result.ok}"
        )
        # The output should describe the failure
        assert "Failed to extract" in result.output or "No content extracted" in result.output, (
            f"expected a failure message in output, got:\n{result.output[:500]}"
        )

    # ------------------------------------------------------------------
    # test_extract_truncates_long_content
    # ------------------------------------------------------------------

    @pytest.mark.skipif(not os.getenv("TAVILY_API_KEY"), reason="requires TAVILY_API_KEY env var")
    def test_extract_truncates_long_content(self) -> None:  # pragma: no cover
        """Extract from a long Wikipedia article should produce output
        exceeding 15 000 characters, which triggers the truncation marker.

        The ``_extract`` method slices content at 15 000 chars and appends
        ``[... truncated ...]`` when the raw content exceeds that limit.
        """
        tool = WebSearchToolkit(api_key=os.environ["TAVILY_API_KEY"])
        result = tool.invoke(
            action="extract",
            url="https://en.wikipedia.org/wiki/Artificial_intelligence",
        )

        assert result.ok is True, (
            f"expected ok=True for a valid Wikipedia page, got ok={result.ok}"
        )
        assert "[... truncated ...]" in result.output, (
            "expected the truncation marker '[... truncated ...]' in the output "
            f"for a long Wikipedia article, but it was not found.\n"
            f"output length={len(result.output)}, preview:\n{result.output[:300]}"
        )
        assert len(result.output) > 15000, (
            f"expected output length > 15000 chars, got {len(result.output)}"
        )

    # ------------------------------------------------------------------
    # test_extract_metadata_contains_url_and_action
    # ------------------------------------------------------------------

    @pytest.mark.skipif(not os.getenv("TAVILY_API_KEY"), reason="requires TAVILY_API_KEY env var")
    def test_extract_metadata_contains_url_and_action(self) -> None:  # pragma: no cover
        """The result metadata should contain the keys ``url``, ``action``,
        ``content_length``, and ``failed_count`` after a successful extract."""
        tool = WebSearchToolkit(api_key=os.environ["TAVILY_API_KEY"])
        result = tool.invoke(action="extract", url="https://example.com")

        assert result.ok is True, (
            f"expected ok=True for a successful extract, got ok={result.ok}"
        )
        meta = result.metadata
        assert "url" in meta, f"metadata missing 'url' key; keys: {list(meta.keys())}"
        assert "action" in meta, f"metadata missing 'action' key; keys: {list(meta.keys())}"
        assert "content_length" in meta, (
            f"metadata missing 'content_length' key; keys: {list(meta.keys())}"
        )
        assert "failed_count" in meta, (
            f"metadata missing 'failed_count' key; keys: {list(meta.keys())}"
        )

        # Validate value types / ranges
        assert meta["url"] == "https://example.com", f"unexpected url: {meta['url']!r}"
        assert meta["action"] == "extract", f"unexpected action: {meta['action']!r}"
        assert isinstance(meta["content_length"], int), (
            f"expected content_length to be int, got {type(meta['content_length'])}"
        )
        assert meta["content_length"] > 0, (
            f"expected content_length > 0, got {meta['content_length']}"
        )
        assert isinstance(meta["failed_count"], int), (
            f"expected failed_count to be int, got {type(meta['failed_count'])}"
        )
        assert meta["failed_count"] >= 0, (
            f"expected failed_count >= 0, got {meta['failed_count']}"
        )
