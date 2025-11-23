
tokencount.py: allows you to count the total tokens in a file, that way it makes it easy to see how long it will take for training of your model will be. seperate from basinmarkov. does not interfere with basinmarkov at all.

basin.py: the main script for basinmarkov. run basin.py for a list of commands to run or check main readme for more info for training or chat.

basin_markov.db: the model itself! it is created in the same directory as basin.py.

basin_markov.db-shm: worker file for basin_model.db when chatting or training. do not delete!

basin_markov.db-wal: worker file for basin_model.db when chatting or training. do not delete!
