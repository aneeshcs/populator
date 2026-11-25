"""Calendar helpers of the daily (noleap) dataset and trainer."""
import pytest

pytest.importorskip("credit.datasets.gen_2")   # CREDIT Gen 2 only

from pop_emulator.credit_ext.dataset_daily import (  # noqa: E402
    interpolation_months, month_centre, month_first_day, month_of_day)
from pop_emulator.credit_ext.trainer_daily import month_blocks  # noqa: E402


def test_month_of_day_noleap():
    assert month_of_day(0) == (0, 0, 31)
    assert month_of_day(31) == (1, 0, 28)
    assert month_of_day(58) == (1, 27, 28)          # 28 February; no leap day follows
    assert month_of_day(59) == (2, 0, 31)
    assert month_of_day(365) == (12, 0, 31)
    assert all(month_first_day(month_of_day(d)[0]) <= d for d in range(0, 800, 7))


def test_interpolation_weights():
    # Mid-January lies on the January node.
    assert interpolation_months(15, 24)[2] == pytest.approx(0.0, abs=0.04)
    # The last day of January lies about halfway between the January and February centres.
    a, b, w = interpolation_months(30, 24)
    assert (a, b) == (0, 1) and 0.45 < w < 0.55
    assert w == pytest.approx((30.5 - month_centre(0)) / (month_centre(1) - month_centre(0)))
    # Clamped at both ends of the record.
    assert interpolation_months(0, 24) == (0, 1, 0.0)
    assert interpolation_months(2 * 365 - 1, 24) == (22, 23, 1.0)


def test_month_blocks_cover_the_rollout():
    for first in (0, 31, 59, 334):
        for n in (31, 62, 124, 372):
            blocks = month_blocks(first, n)
            assert sum(length for length, _ in blocks) == n
            # 31 N days from a first-of-month start hold N complete months.
            assert sum(c for _, c in blocks) >= n // 31
    assert month_blocks(31, 31) == [(28, True), (3, False)]
    assert month_blocks(5, 40) == [(26, False), (14, False)]
