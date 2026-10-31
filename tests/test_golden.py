"""Golden behavior baselines: pin the exact point set each method emits for a
fixed ``random_state``. Unlike the shape/bounds checks in ``test_sampling.py``,
these lock the actual numbers, so any future drift in the RNG wiring or an
algorithm's output is caught. Bless intended changes with ``pyclawd golden
update`` (humans review the diff); agents only run the compare.
"""

import pytest

from pysampling.sample import sample

ALGORITHMS = ["random", "lhs", "halton", "sobol"]


@pytest.mark.golden
@pytest.mark.parametrize("algorithm", ALGORITHMS, ids=ALGORITHMS)
def test_sample_is_reproducible(algorithm):
    # Fixed random_state -> deterministic output for the stochastic methods
    # (random, lhs); sobol/halton are deterministic regardless. riesz is not
    # pinned here: it runs an optimizer, so its exact points differ slightly
    # across Python/numpy builds (see test_riesz_is_deterministic_and_in_bounds).
    return sample(algorithm, 25, 3, random_state=1)


def test_riesz_is_deterministic_and_in_bounds():
    # riesz minimizes an energy iteratively, so the exact numbers vary across
    # platforms and cannot be pinned to a tight golden tolerance. What must
    # hold everywhere is: same seed -> same points, shape as requested, in [0, 1].
    a = sample("riesz", 25, 3, random_state=1)
    b = sample("riesz", 25, 3, random_state=1)
    assert a.shape == (25, 3)
    assert (a >= 0).all() and (a <= 1).all()
    assert (a == b).all()
