# 2022 Ebay University Machine Learning Competition: Name Entity Recognition

We(nlpp) ranked the third in the public leaderboard.

This is the document that describes the code and its usage for Ebay NER problem.

Major files are shown below.

-  `baseline.py`: the main function to train the model.

-  `baseline_gen.py`: the main function to generate labels for unlabled data based on the trained model.

-  `data.py`: the designed data loaders for training and testing phrases.

-  `model.py`: the designed model structure.

-  `baseline_sub.py`: the main function to test the model with quiz data by a single trained model.

-  `baseline_sub_multi.py`: the main function to test the model with quiz data by multiple trained models and their performance as weights.