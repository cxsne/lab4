"""
 Name: Your Name
 Assignment: Lab 4 - Decision Tree
 Course: CS 330
 Semester: Fall 2021
 Instructor: Dr. Cao
 Date: the current date
 Sources consulted: any books, individuals, etc consulted

 Known Bugs: description of known bugs and other program imperfections

 Creativity: anything extra that you added to the lab

 Instructions: After a lot of practice in Python, in this lab, you are going to design the program for decision tree and implement it from scrath! Don't be panic, you still have some reference, actually you are going to translate the JAVA code to Python! The format should be similar to Lab 2!

"""
import sys
import argparse
import math
import os


datamap= {}
attvalues={}
atts=[]
numatts=0
numclasses=0
root=None

class Treenode:
    def __init__(self, parent):
        self.parent= parent
        self.children= {}
        self.attribute= "None"
        self.returnval =None


# You may need to define the Tree node and add extra helper functions here
def read_training_data(infile,percent):
    global datamap,attvalues,atts,numatts,numclasses
    try:
        datamap= {}
        attvalues={}

        with open(infile, 'r') as f:
            attline= f.readline().strip()[1:]
            atts= attline.split('|')
            numatts= len(atts)-1

            for a in atts:
                attvalues[a]=[]

            index=0
            for line in f:
                tokens = line.strip().split()
                dataclass= tokens[0]

                if dataclass not in attvalues[atts[0]]:
                    attvalues[atts[0]].append(dataclass)

                if dataclass not in datamap:
                    datamap[dataclass]= []

                datapoint= []
                for i in range(numatts):
                    val =tokens[i+1]
                    datapoint.append(val)

                    if val not in attvalues[atts[i+1]]:
                        attvalues[atts[i+1]].append(val)

                if index % 100 < percent:
                    datamap[dataclass].append(datapoint)

                index += 1

            numclasses= len(datamap.keys())

    except IOError:
        print("Error reading file: ", infile)
        exit(1)

def build_tree():
    global root
    root = Treenode(None)
    currfreeatts = [atts[i+1] for i in range(numatts)]
    root= build_tree_node(None,currfreeatts,datamap)


def build_tree_node(parent, currfreeatts, nodedata):
    curr = Treenode(parent)
    total_rows = sum(len(rows) for rows in nodedata.values())
    if total_rows == 0:
        max_class = max(nodedata, key=lambda k: len(nodedata[k])) if nodedata else "undefined"
        curr.returnval = max_class
        return curr

    classes_with_data = [k for k, v in nodedata.items() if len(v) > 0]
    if len(classes_with_data) == 1:
        curr.returnval = classes_with_data[0]
        return curr

    minent = float('inf')
    minatt = None

    for i, att in enumerate(currfreeatts):
        if att is None:
            continue
        vals = attvalues[att]
        partition = [[0]*numclasses for _ in range(len(vals))]

        for j in range(numclasses):
            outcome = attvalues[atts[0]][j]
            l = nodedata.get(outcome, [])
            for row in l:
                partition[vals.index(row[i])][j] += 1

        ent = partition_entropy(partition)
        if ent < minent:
            minent = ent
            minatt = att

    if minatt is None:
        max_class = max(nodedata, key=lambda k: len(nodedata[k]))
        curr.returnval = max_class
        return curr

    curr.attribute = minatt
    attindex = currfreeatts.index(minatt)
    currfreeatts[attindex] = None

    for v in attvalues[minatt]:
        tempmap = {}
        for j in range(numclasses):
            outcome = attvalues[atts[0]][j]
            l = nodedata.get(outcome, [])
            trimlist = [row for row in l if row[attindex] == v]
            tempmap[outcome] = trimlist

        if sum(len(rows) for rows in tempmap.values()) == 0:
            max_class = max(nodedata, key=lambda k: len(nodedata[k]))
            child = Treenode(curr)
            child.returnval = max_class
            curr.children[v] = child
        else:
            curr.children[v] = build_tree_node(curr, currfreeatts[:], tempmap)

    currfreeatts[attindex] = minatt
    return curr


def partition_entropy(partition):
    totalent= 0.0
    total=0.0

    for row in partition:
        n= sum(row)
        total += n
        totalent += n * entropy(row)

    if total==0:
        return 0.0

    return totalent / total

def entropy(classcounts):
    total= sum (classcounts)
    if total ==0:
        return 0.0

    sument= 0.0
    for count in classcounts:
        if count > 0:
            prob = count/total
            sument -= prob *log2(prob)

    return sument

def log2(x):
    if x== 0:
        return 0
    return math.log(x) / math.log(2)

def save_model(modelfile):
    try:
        with open(modelfile,'w') as f:
            for i in range (numatts):
                f.write(atts[i+1]+ '')
            f.write("\n")

            write_node(f,root)
    except IOError as e:
        print("Error writing to file", e)
    exit(1)

def write_node(outfile,curr):
    if curr.returnval is not None:
        outfile.write("[" +curr.returnval+"]")
        return
    outfile.write(curr.attribute + " ( ")
    for val, child in curr.children.items():
        outfile.write(val + " ")
        write_node(outfile, child)
    outfile.write(" ) ")

def read_model(modelfile):
    global root

    with open(modelfile, 'r') as f:
            attline = f.readline().strip()
            attarr = attline.split()
            tokens = f.read().replace('(', ' ( ').replace(')', ' ) ').split()

    root = None
    stack = []
    current_node = None
    index = 0
    while index < len(tokens):
        token = tokens[index]

        if token.startswith('['):
            leaf = Treenode(None)
            leaf.returnval = token[1:-1]

            if stack:
                parent, key = stack.pop()
                leaf.parent = parent
                parent.children[key] = leaf
            else:
                root = leaf
            index += 1
        elif token == '(':
            index += 1
        elif token == ')':
            current_node = None if not stack else stack[-1][0]
            index += 1
        else:
            if current_node is None or current_node.returnval is not None:
                node = Treenode(None)
                node.attribute = token

                if not root:
                    root = node
                current_node = node
                index += 1
            else:
                stack.append((current_node, token))
                index += 1

    return root, attarr


def DTtrain(data, model):
    global datamap, attvalues, atts,numatts, numclasses, root

    read_training_data(data,100)
    build_tree()
    save_model(model)

def trace_tree(node, data, attarr):

    while node.returnval is None:
        att = node.attribute
        try:
            index = attarr.index(att) - 1

            if index < 0 or index >= len(data):
                node = list(node.children.values())[0]
            else:
                val = data[index]
                node = node.children.get(val, list(node.children.values())[0])
        except ValueError:
            node = list(node.children.values())[0]
    return node.returnval
    
def DTpredict(data, model, prediction):
    """
    This is the main function to make predictions on the test dataset. It will load saved model file,
    and also load testing data TestDataNoLabel.txt, and apply the trained model to make predictions.
    You should save your predictions in prediction file, each line would be a label, such as:
    1
    0
    0
    1
    ...
    """
    # implement your code here
    global root, atts
    root, atts = read_model(model)

    predictions = []
    with open(data, 'r') as f:
        for line in f:
            tokens = line.strip().split()
            datapoint = tokens[1:]
            pred = trace_tree(root, datapoint,atts)
            predictions.append(pred)

    with open(prediction, 'w') as f:
        for pred in predictions:
            f.write(pred + '\n')


def EvaDT(predictionLabel, realLabel, output):
    """
    This is the main function. You should compare line by line,
     and calculate how many predictions are correct, how many predictions are not correct. The output could be:

    In total, there are ??? predictions. ??? are correct, and ??? are not correct.

    """
    correct,incorrect, length = 0,0,0
    with open(predictionLabel,'r') as file1, open(realLabel, 'r') as file2:
        pred = [line for line in file1]
        real = [line for line in file2]
        length = len(pred)
        for i in range(length):
            if pred.pop(0) == real.pop(0):
                correct += 1
            else:
                incorrect += 1
    Rate = correct/length

    result = "In total, there are "+str(length)+" predictions. "+str(correct)+" are correct and "+ str(incorrect) + " are incorrect. The percentage is "+str(Rate)
    with open(output, "w") as fh:
        fh.write(result)

def main():
    options = parser.parse_args()
    mode = options.mode       # first get the mode
    print("mode is " + mode)
    if mode == "T":
        """
        The training mode
        """
        inputFile = options.input
        outModel = options.output
        if inputFile == '' or outModel == '':
            showHelper()
        DTtrain(inputFile, outModel)
    elif mode == "P":
        """
        The prediction mode
        """
        inputFile = options.input
        modelPath = options.modelPath
        outPrediction = options.output
        if inputFile == '' or modelPath == '' or outPrediction == '':
            showHelper()
        DTpredict(inputFile,modelPath,outPrediction)
    elif mode == "E":
        """
        The evaluating mode
        """
        predictionLabel = options.input
        trueLabel = options.trueLabel
        outPerf = options.output
        if predictionLabel == '' or trueLabel == '' or outPerf == '':
            showHelper()
        EvaNB(predictionLabel,trueLabel, outPerf)
    pass

def showHelper():
    parser.print_help(sys.stderr)
    print("Please provide input augument. Here are examples:")
    print("python " + sys.argv[0] + " --mode T --input TrainingData.txt --output DTModel.txt")
    print("python " + sys.argv[0] + " --mode P --input TestDataNoLabel.txt --modelPath DTModel.txt --output TestDataLabelPrediction.txt")
    print("python " + sys.argv[0] + " --mode E --input TestDataLabelPrediction.txt --trueLabel LabelForTest.txt --output Performance.txt")
    sys.exit(0)


if __name__ == "__main__":
    #------------------------arguments------------------------------#
    #Shows help to the users                                        #
    #---------------------------------------------------------------#
    parser = argparse.ArgumentParser()
    parser._optionals.title = "Arguments"
    parser.add_argument('--mode', dest='mode',
    default = '',    # default empty!
    help = 'Mode: T for training, and P for making predictions, and E for evaluating the machine learning model')
    parser.add_argument('--input', dest='input',
    default = '',    # default empty!
    help = 'The input file. For T mode, this is the training data, for P mode, this is the test data without label, for E mode, this is the predicted labels')
    parser.add_argument('--output', dest='output',
    default = '',    # default empty!
    help = 'The output file. For T mode, this is the model path, for P mode, this is the prediction result, for E mode, this is the final result of evaluation')
    parser.add_argument('--modelPath', dest='modelPath',
    default = '',    # default empty!
    help = 'The path of the machine learning model ')
    parser.add_argument('--trueLabel', dest='trueLabel',
    default = '',    # default empty!
    help = 'The path of the correct label ')
    if len(sys.argv)<3:
        showHelper()
    main()
