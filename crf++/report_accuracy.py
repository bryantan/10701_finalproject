import sys


if __name__ == "__main__":
    filename = sys.argv[1]
    total = 0
    correct = 0
    with open(filename, "r") as file:
        for line in file:
            tokens = line.replace("\n", "").split("\t")
            if len(tokens) >= 3:
                actual_label = tokens[-2]
                predicted = tokens[-1]
                #predicted_label = predicted[:predicted.find("/")]
                predicted_label = predicted
                if actual_label == predicted_label:
                    correct += 1
                total += 1
    print "Accuracy: %0.2f%%" % (float(correct) * 100 / total)
