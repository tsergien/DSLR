python3 -m venv my_env
pip3 install -r requirements.txt
./describe resources/data_train.csv
./histogram.py resources/data_train.csv
./pair_plot.py resources/data_train.csv
./scatter_plot.py resources/data_train.csv
