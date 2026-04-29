# /bin/bash
echo "Hi! Switching Directory!"

cd $(cat /workdir)
pwd

echo "Checking current Leaderboard Order!"
curl 10.150.4.148:5050/getorder


echo "Running Eval-Script!"
uv run eval/eval.py

echo "Results:"
cat eval/eval_data_mo-lunar-lander-v3.json
