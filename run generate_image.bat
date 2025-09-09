@echo off
call "D:\Program Files (x86)\Microsoft Visual Studio\2019\Community\VC\Auxiliary\Build\vcvarsall.bat" x64
call conda activate MAT
cd /d D:\MAT2
python generate_image.py