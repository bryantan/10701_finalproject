import sys


if __name__ == "__main__":
    filename = sys.argv[1]
    total = 0
    correct = 0
    d = {}
    with open(filename, "r") as file:
        for line in file:
            tokens = line.replace("\n", "").split("\t")
            if len(tokens) >= 3:
                actual_label = tokens[-2]
                predicted_label = tokens[-1]
                # Find the total number of actual labels
                if actual_label not in d:
                    d[actual_label] = {}
                if "total_actual" not in d[actual_label]:
                    d[actual_label]["total_actual"] = 0
                d[actual_label]["total_actual"] += 1
                # Find the total number of classified labels
                if predicted_label not in d:
                    d[predicted_label] = {}
                if "total_predicted" not in d[predicted_label]:
                    d[predicted_label]["total_predicted"] = 0
                d[predicted_label]["total_predicted"] += 1
                # Find the number of correctly classified
                if actual_label == predicted_label:
                    if "correctly_predicted" not in d[actual_label]:
                        d[actual_label]["correctly_predicted"] = 0
                    d[actual_label]["correctly_predicted"] += 1
    for label in d.keys():
        d[label]["precision"] = float(d[label]["correctly_predicted"]) / d[label]["total_actual"]
        d[label]["recall"] = float(d[label]["correctly_predicted"]) / d[label]["total_predicted"]
        d[label]["F1"] = 2 * d[label]["precision"] * d[label]["recall"] / (d[label]["precision"] + d[label]["recall"])
        print label, d[label]["precision"], d[label]["recall"], d[label]["F1"]



