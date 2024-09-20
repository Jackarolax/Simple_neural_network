import csv
import random


def between0and1(num):
    num= int(num)
    return([num/255])

def numtovector(num):
    num = int(num)
    if not num in range(10):
        raise Exception(f"num has to be between 0 and 9, it is {num}")
    
    vector = []
    for i in range(10):
        if i == num:
            vector.append([1])
        else:
            vector.append([0])

    return(vector)

def vectortonum(vector):
    num = 11
    for i, value in vector:
        if value[0] == 1:
            num = i
    if num == 11:
        raise Exception("1 not founnd in vector")
    return(num)


def get_trainingdata(number_of_examples):
    
    rows = []
    with open("train.csv", 'r') as file:
        csvreader = csv.reader(file)
        header = next(csvreader)
        for row in csvreader:
            rows.append(row)


    training_data = []

    for i in range(number_of_examples):
        row = random.choice(rows)
        targetvector = numtovector(row[0])
        inputvector = list(map(between0and1, row[1:]))

        training_data.append([inputvector, targetvector])

    return(training_data)