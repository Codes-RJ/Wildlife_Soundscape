def test_preflight_warns_when_calibration_is_disabled():
    from wildlife_soundscape.core.config import AppConfig
    from wildlife_soundscape.dashboard.preflight import build_preflight_checks

    checks = build_preflight_checks(AppConfig())

    calibration = next(check for check in checks if check.title == "TDOA calibration")
    assert calibration.status == "warning"
