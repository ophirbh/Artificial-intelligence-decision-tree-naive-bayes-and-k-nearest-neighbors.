
from math import ceil
from src.DecisionTree import DecisionTreeLearner
from src.KNN import KNNLearner
from src.NaiveBayes import NaiveBayesLearner

def read_examples_file(examples_file_name, row_delim = "\n",
                       column_delim = "\t"):
    """Returns lists of attributes, examples and classifications after
    reading the argument examples file."""
    attributes_list = []
    examples_list = []
    classifications_list = []
    with open(examples_file_name, "r") as file:
        lines_list = file.readlines()
        attributes_list += (lines_list[0].strip(row_delim).split(column_delim))
        for line in lines_list[1:]:
            parsed_line = line.strip(row_delim).strip().split(column_delim)
            classifications_list.append(parsed_line[-1])
            examples_list.append(parsed_line[:len(parsed_line) - 1])
    return attributes_list, examples_list, classifications_list

def write_output_file(learners_predictions_list, test_classifications,
                      learners_precision_list, column_delim = "\t",
                      row_delim = "\n" ):
    """Writes the output file as requested in the instructions, includes:
    learners predictions, learners accuracies."""
    dt = 0
    knn = 1
    nb = 2
    dt_pfc_list, knn_pfc_list, nb_pfc_list = learners_predictions_list[dt],\
                              learners_predictions_list[knn],\
                              learners_predictions_list[nb]
    dt_prec, knn_prec, nb_prec = learners_precision_list[dt],\
                                 learners_precision_list[knn],\
                                 learners_precision_list[nb]
    with open("output.txt", "w") as output:
        i = 1
        lines_list = []
        headline = ["Num", "DT", "KNN", "naiveBayes"]
        lines_list.append(column_delim.join(headline))
        for actual, dt_pfc, knn_pfc, nb_pfc in zip(test_classifications,
                                                      dt_pfc_list,
                                                      knn_pfc_list,
                                                      nb_pfc_list):
            lines_list.append("{}\t{}\t{}\t{}".format(i, dt_pfc, knn_pfc,
                                                      nb_pfc))
            i += 1
        lines_list.append("\t{}\t{}\t{}".format(dt_prec, knn_prec, nb_prec))
        output.writelines(row_delim.join(lines_list))

def get_classification_precision(predictions, actuals):
    '''Returns the classification precision. Gets the prediction
    classifications and the actual classifications, and calculates the
     precision percentage with a resolution of 1/100'''
    assert len(predictions) == len(actuals)
    correct = 0.0
    for predicted, actual in zip(predictions, actuals):
        if predicted == actual:
            correct += 1
    precision = ceil((correct / len(actuals)) * 100) / 100
    return precision

if __name__ == "__main__":
    attributes, training_examples, training_classifications = \
        read_examples_file("train.txt")
    attributes_space_holder, test_examples, test_classifications = \
        read_examples_file("test.txt")

    dt = DecisionTreeLearner(training_examples, training_classifications,
                                 attributes[:len(attributes) - 1])
    knn = KNNLearner(training_examples, training_classifications, k=5)
    nb = NaiveBayesLearner(training_examples, training_classifications)
    learners = [dt, knn, nb]
    learners_predictions = []
    learners_precisions = []
    for learner in learners:
        predictions = []
        for example, classification in zip(test_examples,
                                           test_classifications):
            predictions.append(learner.get_predicted_classification(example))
        learners_predictions.append(predictions)
        learners_precisions.append(get_classification_precision(
            predictions, test_classifications))
    dt.write_tree_representation_to_file()
    write_output_file(learners_predictions, test_classifications,
                      learners_precisions)