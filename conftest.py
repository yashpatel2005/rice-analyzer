import pytest

@pytest.fixture
def smoke_config():
    from config import load_config
    return load_config()


def test_module_imports():
    __import__("core.grading")
    __import__("core.classification")
    __import__("core.preprocessing")
    __import__("core.segmentation")
