# OVA

2.- One vs All Do It Yourself (50 points)

Create a program with the following characteristics:

Name: ova_diy.py
Accepts 2 csv files (2 paths pointing to 2 csv files)
The first csv file has 5 columns (f1, f2, f3, f4, target)
The second csv has 4 columns (f1, f2, f3, f4)
The target will have values 1, 2, 3 
The program needs to implement the one versus all logic for logistic regression 
To do so it will train 3 logistic regression classifiers (1 vs the rest, 2 vs the rest, 3 vs the rest) using the data in the first file
It will apply these classifiers to the data in the second file and obtain three probabilities per record.
It will assign the class corresponding to the classifier that produces the higher probability
It will produce a csv called predictions.csv with 5 columns (f1, f2, f3, f4, predicted_value), being f1, f2, f3, f4 the columns in the second file.
No loops are allowed in the code