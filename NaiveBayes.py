
from collections import Counter
FIRST_EXAMPLE = 0

class NaiveBayesLearner:
    """This class represents a Naive Bayes algorithm, data
    learner."""
    def __init__(self, training_examples, training_classifications):
        self.training_examples = training_examples
        self.training_classifications = training_classifications
        self.attributes_possibilities_dictionary = \
            self.get_attributes_possibilities_dictionary()
        self.classifications_dictionary = self.get_classifications_dictionary()


    def get_classifications_dictionary(self):
        """Returns a dictionary with a classification : example relation."""
        classification_dictionary = {}
        for classification, example in zip(self.training_classifications,
                                           self.training_examples):
            if classification not in classification_dictionary:
                classification_dictionary.update({classification: [example]})
            else:
                classification_dictionary[classification].append(example)
        return classification_dictionary


    def get_attributes_possibilities_dictionary(self):
        """Returns a dictionary with a attribute: attribute possibilities
        relation."""
        attributes_possibilities_dictionary = {}
        for attribute_i in range(len(self.training_examples[FIRST_EXAMPLE])):
            attribute_possibilities = set()
            for example in self.training_examples:
                attribute_possibilities.add(example[attribute_i])
            attributes_possibilities_dictionary.update(
                {attribute_i: [attribute_possibilities]})
        return attributes_possibilities_dictionary


    def get_prior_probability(self, classification):
        """Returns the prior probability of the argument classification."""
        occurences = len(self.get_classifications_dictionary()[classification])
        all_examples = float(len(self.training_examples))
        return occurences / all_examples

    def get_positive_classification(self):
        """Returns the positive classification as described in the Q&A file."""
        for classification in self.classifications_dictionary.keys():
            if classification in ["true", "yes"]:
                return classification

    def get_probabilities_per_classification(
            self, same_classification_examples, example):
        """Returns the probability for the argument example to be classified
         the same as the other classification examples."""
        num_examples_per_classifiaction = len(same_classification_examples)
        attribute_given_classification_probabilities = []
        for attribute_i in range(len(example)):
            num_possibilities_per_atrribute = len(
                self.get_attributes_possibilities_dictionary())
            count_feature_occurances = 0
            for training_example in same_classification_examples:
                if training_example[attribute_i] == example[attribute_i]:
                    count_feature_occurances += 1
            probability = float(count_feature_occurances) / \
                          (num_examples_per_classifiaction +
                           num_possibilities_per_atrribute)
            attribute_given_classification_probabilities.append(probability)
        prior_probability = float(num_examples_per_classifiaction) \
                            / len(self.training_examples)
        res = 1
        for att_probability in attribute_given_classification_probabilities:
            res *= att_probability
        res *= prior_probability
        return res

    def get_default_classification(self):
        """Returns the default classification, first by majority, and then
        by positive classification."""
        classifications_histogram = Counter()
        for classification in self.training_classifications:
            classifications_histogram[classification] += 1
        most_common = classifications_histogram.most_common(1)
        most_commons = []
        for key in classifications_histogram.keys():
            if classifications_histogram[key] == most_common[0][1]:
                most_commons.append(key)
        if len(most_commons) == 2:
            return self.get_positive_classification()
        return most_commons[0]

    def get_predicted_classification(self, example):
        """Returns the predicted classification for the argument example."""
        probabilities_by_classification_dictionary = {}
        most_probable_classification = self.classifications_dictionary
        maximal_probability = 0
        for classification in self.classifications_dictionary:
            probability = self.get_probabilities_per_classification(
                self.classifications_dictionary[classification], example)
            if maximal_probability < probability:
                maximal_probability = probability
                most_probable_classification = classification
            probabilities_by_classification_dictionary.update({classification: probability})
        maximal_probabilities = []
        for key in probabilities_by_classification_dictionary.keys():
            if probabilities_by_classification_dictionary[key] == maximal_probability:
                maximal_probabilities.append(key)
        if len(maximal_probabilities) == 2:
            return self.get_default_classification()
        return most_probable_classification