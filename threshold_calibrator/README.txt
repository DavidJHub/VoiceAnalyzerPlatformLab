Example of use:

from threshold_calibrator.threshold_calibrator import calibrate_thresholds

# Calibración de thresholds
thresholds=calibrate_thresholds(df, percent=20)

#ver los 3 datos:
print("Calibrated Thresholds:", thresholds)