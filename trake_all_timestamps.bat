@echo off
setlocal enabledelayedexpansion

REM 获取当前路径
set "current_path=%cd%"

REM 构建目标路径和新目标路径  訓練部分請更改路徑為"trainv2\images"  測試部分請更改路徑為"32_33_AI_CUP_testdataset\AI_CUP_testdata\images"
set "target_path=%current_path%\32_33_AI_CUP_testdataset\AI_CUP_testdata\images"


REM 检查目标路径是否存在
if not exist "%target_path%" (
    echo Target path does not exist: %target_path%
    exit /b 1
)


REM 进入目标路径
pushd "%target_path%"

REM 遍历目标路径下的所有文件夹
for /d %%i in (*) do (
    set "absolute_path=%target_path%\%%i"
    echo "!absolute_path!"
    REM 执行 Python 指令 請更改路徑 "your_path\Team_5084-main"
    pushd "E:\AI_CUP\AI-driven_Future\Team_5084-main"   
    python tools\final_mc_demo_yolov7.py --source "!absolute_path!" --name "%%i"
    popd

)

REM 返回上一级目录
popd

endlocal
