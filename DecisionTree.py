from collections import Counter
from math import log

UNIQUE = 1
TUPLE = 0
CLASSIFICATION = 0
SOLE_CLASSIFICATION = 0

class DecisionTreeLearner:

    def __init__(self, training_examples, training_classifications,
                 attributes):
        self.training_examples = training_examples
        self.training_classifications = training_classifications
        self.attributes = attributes
        self.attributes_possibilities_dictionary = \
            self.get_attributes_possibilities_dictionary()
        classified_examples = \
            [(example, tag) for example, tag in
             zip(self.training_examples, self.training_classifications)]
        self.decisionTree = \
            DecisionTree(self.build_decision_tree(
                attributes, classified_examples, 0,
                self.get_default_classification(training_classifications)),
                attributes)

    def build_decision_tree(self, attributes, classified_examples, depth,
                            default_classification = None):
        """Builds a decision tree according to the algorithm from class."""
        examples = self.get_examples(classified_examples)
        if len(examples) == 0:
            return DecisionNode(None, depth, classification =
            default_classification, is_a_leaf = True)
        classifications = self.get_classifications(classified_examples)
        if len(set(classifications)) == 1:
            return DecisionNode(
                None, depth, classification =
                classifications[SOLE_CLASSIFICATION], is_a_leaf = True)
        if len(attributes) == 0:
            return DecisionNode(
                None, depth, classification = self.get_default_classification(
                    classifications), is_a_leaf = True)
        attributes_left = attributes[:]
        best_attribute = self.get_best_attribute(attributes, examples,
                                                 classifications)
        attribute_i = self.get_attribute_i(best_attribute)
        decision_node = DecisionNode(best_attribute, depth)
        attributes_possibilities_dictionary = \
            self.get_attributes_possibilities_dictionary()
        attributes_left.remove(best_attribute)
        for possibility in attributes_possibilities_dictionary[best_attribute]:
            examples_and_tags_vi = \
                self.get_classified_examples_for_argument_possibility(
                    attribute_i, possibility, examples, classifications)
            decision_node.children_dictionary[possibility] =\
                self.build_decision_tree(
                    attributes_left, examples_and_tags_vi, depth + 1,
                    self.get_default_classification(classifications))
        return decision_node

    def get_classified_examples_for_argument_possibility(
            self, attribute_i, possibility, examples, classifications):
        """Returns a tuple list of classified examples per attribute
         possibility."""
        classified_examples = []
        for example, classification in zip(examples, classifications):
            if example[attribute_i] == possibility:
                classified_examples.append((example, classification))
        return classified_examples


    def get_classifications(self, classified_examples):
        """Extract the classifications out of the classified examples."""
        classifications = []
        for example, classification in classified_examples:
            classifications.append(classification)
        return classifications

    def get_examples(self, classified_examples):
        """Extract the examples out of the classified examples."""
        examples = []
        for example, classification in classified_examples:
            examples.append(example)
        return examples

    def get_info_gain_calculation(self, attribute, examples, classifications):
        """Returns the information gain per attribute."""
        entropy_per_possibility_list = []
        entropy =  self.get_entropy_calculation(classifications)
        attribute_i = self.get_attribute_i(attribute)
        for possibility in self.attributes_possibilities_dictionary[attribute]:
            classifications_per_possibility = \
                self.get_classifications_for_argument_possibility(
                    attribute_i, possibility, examples, classifications)
            entropy_per_possibility = self.get_entropy_calculation(
                classifications_per_possibility)
            entropy_per_possibility_list.append(
                entropy_per_possibility * (float(len(
                    classifications_per_possibility)) / len(examples)))
        attribute_info_gain =  entropy - sum(entropy_per_possibility_list)
        return attribute_info_gain

    def get_classifications_for_argument_possibility(
            self, feature_index, possibility, examples, classifications):
        """Returns the classifications list per attribute possibility."""
        classifications_per_possibility = []
        for example, classification in zip(examples, classifications):
            if example[feature_index] == possibility:
                classifications_per_possibility.append(classification)
        return classifications_per_possibility


    def get_entropy_calculation(self, classifications):
        """Returns the entropy calculation for the argument classifications."""
        classifications_histogram = Counter()
        if len(classifications) > 0:
            for classification in classifications:
                classifications_histogram[classification] += 1
            entropy = 0
            classifications_probabilities =\
                self.get_classifications_probabilities(
                    classifications_histogram, classifications)
            for probability in classifications_probabilities:
                if probability == 0:
                    return 0
                entropy -= log(probability, 2) * probability
            return entropy
        else:
            return 0

    def get_best_attribute(self, attributes, examples, classifications):
        """Returns the best attribute according to information gain."""
        best_attribute = attributes[0]
        best_info_gain = 0
        attributes_info_gains_dictionary =\
            self.get_attributes_info_gains_dictionary(
                attributes, examples, classifications)
        for attribute in attributes:
            if best_info_gain < attributes_info_gains_dictionary[attribute]:
                best_attribute = attribute
                best_info_gain = attributes_info_gains_dictionary[attribute]
        return best_attribute

    def get_attributes_info_gains_dictionary(self, attributes, examples,
                                             classifications):
        """Returns a dictionary with attribute - info gain relation."""
        attributes_info_gains = {}
        for attribute in attributes:
            attributes_info_gains.update(
                {attribute: self.get_info_gain_calculation(
                    attribute, examples, classifications)})
        return attributes_info_gains

    def get_classifications_probabilities(self, classifications_histogram,
                                          classifications):
        """Returns the probabilities for each classification from the
         arguments: classification histogram and the classifications
          themselves."""
        classifications_probabilities = []
        for tag in classifications_histogram:
            classifications_probabilities.append(
                float(classifications_histogram[tag]) / len(classifications))
        return classifications_probabilities

    def get_attribute_i(self, attribute):
        """Returns the i-th location of the argument attribute from the
        attributes list."""
        for i in range(len(self.attributes)):
            if attribute == self.attributes[i]:
                return i


    def get_attributes_possibilities_dictionary(self):
        """Returns the a dictionary that holds an attribute : attribute
        possibilities relation."""
        attributes_possibilities_dictionary = {}
        for attribute_i in range(len(self.training_examples[0])):
            attribute_possibilities = set()
            for example in self.training_examples:
                attribute_possibilities.add(example[attribute_i])
            attributes_possibilities_dictionary.update(
                {self.attributes[attribute_i]: attribute_possibilities})
        return attributes_possibilities_dictionary

    def write_tree_representation_to_file(
            self, output_file_name = "output_tree.txt"):
        """Writes the tree representation to file."""
        with open(output_file_name, "w") as output:
            representation = self.decisionTree.to_string(
                self.decisionTree.get_root())
            representation = representation[:len(representation) - 1]
            output.write(representation)

    def get_most_common_classification(self, classifications):
        """Returns a list (may be empty) of the most common unique
         classification."""
        classifications_histogram = Counter()
        for classification in classifications:
            classifications_histogram[classification] += 1
        classifications_values = list(classifications_histogram.values())
        most_common = []
        if classifications_values[0] != classifications_values[1]:
            most_common.append(
                classifications_histogram.
                    most_common(UNIQUE)[TUPLE][CLASSIFICATION])
        return most_common

    def get_positive_classification(self):
        """Returns the positive classification as described in the Q&A file.
         """
        for classification in self.training_classifications:
            if classification in ["true", "yes"]:
                return classification

    def get_default_classification(self, classifications):
        """Returns the default classification as defined in the Q&A file,
        meaning first return major. if there is no major return positive."""
        most_common_classification = self.get_most_common_classification(
            classifications)
        if not most_common_classification:
            return self.get_positive_classification()
        return most_common_classification[0]

    def get_predicted_classification(self, example):
        """Gets the tree's prediction for argument example."""
        return self.decisionTree.get_classification_from_leaf(example)


class DecisionTree(object):
    """This class represents a decision tree."""
    def __init__(self, root, attributes):
        self.root = root
        self.attributes = attributes

    def get_root(self):
        """Returns the decision's tree root."""
        return self.root

    def get_attributes(self):
        """Returns the decision's tree attributes."""
        return self.attributes

    def get_attribute_i(self, attribute):
        """Returns the attribute index."""
        for i in range(len(self.attributes)):
            if attribute == self.attributes[i]:
                return i

    def to_string(self, node, depth_token = "\t", child_token = "|",
                        possibility_delim = "=", classification_delim = ":",
                        node_delim = "\n"):
        """Returns the tree's string as described in the instructions."""
        string_format = ""
        for child in sorted(node.children_dictionary):
            string_format += depth_token * node.depth
            if node.depth > 0:
                string_format += child_token
            string_format += node.attribute + possibility_delim + child
            if not node.children_dictionary[child].is_a_leaf:
                string_format += node_delim + self.to_string(
                    node.children_dictionary[child])
            else:
                string_format +=\
                    classification_delim +\
                    node.children_dictionary[child].classification + node_delim
        return string_format


    def get_classification_from_leaf(self, example):
        """Returns the argument example classification from the appropriate
         leaf node."""
        curr_node = self.get_root()
        while not curr_node.check_if_leaf():
            curr_node = curr_node.children_dictionary[example[
                self.get_attribute_i(curr_node.attribute)]]
        return curr_node.get_classification()


class DecisionNode(object):
    """This class represents a decsion tree node."""
    def __init__(self, attribute, depth, classification = None,
                 is_a_leaf = False):
        self.attribute = attribute
        self.children_dictionary = {}
        self.depth = depth
        self.is_a_leaf = is_a_leaf
        self.classification = classification

    def get_attribute(self):
        """Returnes the node's attribute."""
        return self.attribute

    def check_if_leaf(self):
        """Returns true if the node is a leaf, false otherwise."""
        return self.is_a_leaf

    def get_classification(self):
        return self.classification

