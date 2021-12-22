from src import num

def test_can_increment():
    assert num.increment(3) == 4

def test_can_decrement():
    assert num.decrement(3) == 2
