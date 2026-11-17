from src.food11.data import category_from_filename


def test_deliberate_lab5_failure():
    assert category_from_filename("0_12.jpg") == "Egg"
