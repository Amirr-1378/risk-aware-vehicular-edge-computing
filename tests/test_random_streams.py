"""Master-seed reproducibility, creation-order independence, and RNG isolation."""

import random
import unittest

from risk_aware_v2x.models.randomness import RandomStreams


class RandomStreamTests(unittest.TestCase):
    def test_same_seed_and_name_repeat_trajectories(self) -> None:
        first, second = RandomStreams(123), RandomStreams(123)
        for name in ("arrivals", "mobility", "shadowing", "fading", "background", "agent"):
            with self.subTest(name=name):
                self.assertEqual(
                    [first.stream(name).random() for _ in range(10)],
                    [second.stream(name).random() for _ in range(10)],
                )

    def test_extra_fading_draws_do_not_affect_other_streams(self) -> None:
        first, second = RandomStreams(123), RandomStreams(123)
        for _ in range(100):
            first.stream("fading").random()
        for name in ("arrivals", "mobility", "shadowing", "background", "agent"):
            with self.subTest(name=name):
                self.assertEqual(first.stream(name).random(), second.stream(name).random())

    def test_creation_order_and_global_rng_do_not_affect_named_streams(self) -> None:
        first, second = RandomStreams(123), RandomStreams(123)
        first.stream("mobility")
        first.stream("arrivals")
        second.stream("arrivals")
        second.stream("mobility")
        global_state = random.getstate()
        self.assertEqual(first.stream("mobility").random(), second.stream("mobility").random())
        self.assertEqual(random.getstate(), global_state)
        self.assertIs(first.stream("mobility"), first.stream("mobility"))

    def test_different_seed_or_name_changes_stream(self) -> None:
        self.assertNotEqual(RandomStreams(1).stream("fading").random(), RandomStreams(2).stream("fading").random())
        self.assertNotEqual(RandomStreams(1).stream("fading").random(), RandomStreams(1).stream("arrivals").random())

    def test_invalid_master_seeds_and_names_are_rejected(self) -> None:
        for seed in (-1, True, 1.5):
            with self.subTest(seed=seed):
                with self.assertRaises((TypeError, ValueError)):
                    RandomStreams(seed)
        with self.assertRaises(ValueError):
            RandomStreams(1).stream(" ")
