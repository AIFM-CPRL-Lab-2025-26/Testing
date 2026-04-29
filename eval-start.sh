# /bin/bash
echo "Nothing is happening (yet)!"

cd $(cat /workdir)
curl 10.150.4.148:5050/getorder
uv run eval/eval.py
