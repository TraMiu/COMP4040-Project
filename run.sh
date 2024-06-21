# pip install jupyter => sudo chmod 777 run.sh =>tmux => ./run.sh
# ctr b + d => tmux ls =>  tmux a -t 0
# ctr c => exit

# jupyter nbconvert --to script test-bdq-eng.ipynb
# python test-bdq-eng.py > run_log.log

# jupyter nbconvert --to script test-eng-bdq.ipynb
# python test-eng-bdq.py > run_en_ba_log.log

jupyter nbconvert --to script test-eng-bdq-bible-only.ipynb
python test-eng-bdq-bible-only.py > run_en_ba_log.log