"""Rule-based fact capture (the zero-cost extraction floor, rules.py)."""

from elastimem import rules


class TestNameCapture:
    def test_plain_name(self):
        assert rules.capture("my name is kumar") == [("name", "kumar")]

    def test_capitalized_name(self):
        assert rules.capture("my name is Kumar") == [("name", "Kumar")]

    def test_full_name_phrasing(self):
        """Regression test: 'my full name is X' didn't match the old regex
        (it only accepted 'my name is X' with nothing in between) - a
        common phrasing when a user expands on a shorter name they gave
        earlier."""
        assert rules.capture("my full name is kumarswami") == [("name", "kumarswami")]

    def test_real_name_phrasing(self):
        assert rules.capture("my real name is Priya") == [("name", "Priya")]

    def test_call_me(self):
        assert rules.capture("you can call me kumar") == [("name", "kumar")]

    def test_leading_stopword_rejected(self):
        assert rules.capture("my name is not important") == []


class TestLocationCapture:
    def test_live_in(self):
        assert rules.capture("I live in Bangalore") == [("location", "Bangalore")]

    def test_stay_in(self):
        """Regression test: 'I stay in X' is common phrasing (especially in
        Indian English) that the old rules never matched at all - only
        'I live in X' was covered."""
        assert rules.capture("i stay in banglore") == [("location", "banglore")]

    def test_stay_in_capitalized(self):
        assert rules.capture("I stay in Bangalore") == [("location", "Bangalore")]

    def test_from(self):
        assert rules.capture("I'm from Austin") == [("location", "Austin")]

    def test_live_in_the_moment_not_captured(self):
        assert rules.capture("i live in the moment") == []
