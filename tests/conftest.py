def pytest_configure(config):
    """Configure pytest with custom marks."""
    config.addinivalue_line(
        "markers",
        "integration: mark test as an integration test that makes real API calls",
    )
