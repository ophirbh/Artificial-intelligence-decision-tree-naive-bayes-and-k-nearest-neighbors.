from collections import Counter

FIRST_INSTANCE = 0
ATTRIBUTES = 0
CLASSIFICATION = 1
EXAMPLE = 0
VALUE = 0
TAKE_ONLY_MOST_COMMON = 1
SECOND_ELEMENT = 1


class KNNLearner:
    """This class represents a K Nearest Neighbours algorithm, data
    learner."""

    def __init__(self, training_examples, training_classifications, k):
        """Constructor. Creates and initializes a new KNN object."""
        self.training_examples = training_examples
        self.training_classifications = training_classifications
        self.k = k

    @staticmethod
    def calculate_hamming_distance(example_one ,example_two):
        """This method calculates and returns the hamming distance between two
        argument examples."""
        hamming_distance = 0
        assert len(example_one) == len(example_two)
        for feature_one, feature_two in zip(example_one, example_two):
            if feature_one != feature_two: hamming_distance += 1
        return hamming_distance

    def get_hamming_distances_from_training_examples(self, example):
        """This method returns a list of tuples of kind (train example,
         distance), that connects each train example to is corresponding
         hamming distance from the argument example."""
        hamming_distances = []
        classified_training_examples = self.get_classified_training_examples()
        for classified_training_example in classified_training_examples:
            hamming_distance = self.calculate_hamming_distance(
                example, classified_training_example[ATTRIBUTES])
            hamming_distances.append((classified_training_example,
                                      hamming_distance))
        return hamming_distances

    def get_classified_training_examples(self):
        """This method returns a list of tuples of kind (train example,
        train example classification) of the training set."""
        classified_training_examples = [(training_example,
                                         training_classification) for
                                        training_example,
                                        training_classification in
                                        zip(self.training_examples,
                                            self.training_classifications)]
        return classified_training_examples

    @staticmethod
    def sort_classified_training_examples_by_ascending_distances(
            classified_training_examples):
        """This method returns a sorted list of tuples of kind (train example,
        hamming distance from argument example), sort by ascending distance
        order."""
        sorted_ascending_distances = sorted(classified_training_examples,
                                            key=lambda x: x[SECOND_ELEMENT])
        return sorted_ascending_distances

    def get_k_nearest_neighbours(self, distance_sorted_classified_examples):
        """This method returns the k nearest neighbours according to their
        hamming distance."""
        return distance_sorted_classified_examples[:self.k]

    @staticmethod
    def get_the_most_common_classification(k_nearest_classified_examples):
        """This method returns the most common classification of the k nearest
        neighbours."""
        classifications_counter = Counter()
        for example in k_nearest_classified_examples:
            classifications_counter[example[EXAMPLE][CLASSIFICATION]] += 1
        histogram = classifications_counter.most_common(TAKE_ONLY_MOST_COMMON)
        return histogram[FIRST_INSTANCE][VALUE]

    def get_predicted_classification(self, example):
        """This method gets an example as argument and returns its
        classification prediction according to the KNN algorithm, and its
        training set."""
        distances_from_example = \
            self.get_hamming_distances_from_training_examples(example)
        ascending_distances_from_example =\
            self.sort_classified_training_examples_by_ascending_distances\
                (distances_from_example)
        k_nearest_examples =\
            self.get_k_nearest_neighbours(ascending_distances_from_example)
        return self.get_the_most_common_classification(k_nearest_examples)