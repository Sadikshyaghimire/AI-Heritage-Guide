@echo off
cd /d "D:\FYP Artefacts"

call venv\Scripts\activate

echo ======================================
echo Starting EfficientNet Training...
echo ======================================
python model_training\train_efficientnet.py

if errorlevel 1 (
    echo.
    echo EfficientNet FAILED!
    pause
    exit /b
)

echo.
echo ======================================
echo Starting MobileNetV2 Training...
echo ======================================
python model_training\train_mobilenetv2.py

if errorlevel 1 (
    echo.
    echo MobileNetV2 FAILED!
    pause
    exit /b
)

echo.
echo ======================================
echo ALL TRAINING COMPLETED
echo ======================================
pause