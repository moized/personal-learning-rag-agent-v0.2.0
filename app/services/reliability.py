"""Helpers that keep a temporary AI-service failure from becoming permanent saved state."""
import re

from ..providers.base import ProviderAuthenticationError, ProviderRateLimitError

# Gemini wraps exhausted retries in a plain RuntimeError, so the status code and
# transport failures are recognised from the message as well as from the type.
_TRANSIENT_MESSAGE = re.compile(r"\b(429|500|502|503|504)\b|timed? ?out|connection|temporar|unavailable", re.IGNORECASE)


def is_retryable_error(exc: BaseException) -> bool:
    """True when retrying later could help: rate limits, quota, outages, timeouts, bad credentials.

    Callers that persist a result must let these propagate so the job retry logic
    runs, instead of saving a degraded result that is never revisited.
    """
    if isinstance(exc, (ProviderAuthenticationError, ProviderRateLimitError)):
        return True
    return bool(_TRANSIENT_MESSAGE.search(str(exc)))


def profile_needs_refresh(profile: dict) -> bool:
    """True when a stored source profile is missing, outdated or a degraded fallback.

    Earlier versions saved an empty fallback profile marked as current whenever the
    AI call failed, so an empty profile with no identity at all is treated as degraded.
    """
    if profile.get("schema_version") != 2 or profile.get("degraded"):
        return True
    return (
        not profile.get("course_title")
        and not profile.get("primary_learning_goal")
        and not profile.get("course_identity_confidence")
    )
