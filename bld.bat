"%PYTHON%" -m pip install git+https://github.com/rtqichen/torchdiffeq
"%PYTHON%" setup.py install --single-version-externally-managed --record=record.txt
if errorlevel 1 exit 1
