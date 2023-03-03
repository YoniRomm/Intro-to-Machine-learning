#################################
# Your name: Yonatan Romm
#################################
import math

import numpy as np
import matplotlib.pyplot as plt
import intervals


class Assignment2(object):
    """Assignment 2 skeleton.

    Please use these function signatures for this assignment and submit this file, together with the intervals.py.
    """

    def sample_from_D(self, m):
        """Sample m data samples from D.
        Input: m - an integer, the size of the data sample.

        Returns: np.ndarray of shape (m,2) :
                A two dimensional array of size m that contains the pairs where drawn from the distribution P.
        """
        x_array = np.sort(np.random.uniform(size=m))  # we need to assume the samples are sorted
        y_0 = np.logical_or(np.logical_and(x_array > 0.2, x_array < 0.4), np.logical_and(x_array > 0.6, x_array < 0.8))
        y_1 = np.logical_not(y_0)
        y_array = np.array([self.get_y_based_on_x(y_1[i]) for i in range(m)]).reshape((m,))
        sample = np.array([(x_array[i], y_array[i]) for i in range(m)])
        return sample

    def experiment_m_range_erm(self, m_first, m_last, step, k, T):
        """Runs the ERM algorithm.
        Calculates the empirical error and the true error.
        Plots the average empirical and true errors.
        Input: m_first - an integer, the smallest size of the data sample in the range.
               m_last - an integer, the largest size of the data sample in the range.
               step - an integer, the difference between the size of m in each loop.
               k - an integer, the maximum number of intervals.
               T - an integer, the number of times the experiment is performed.

        Returns: np.ndarray of shape (n_steps,2).
            A two dimensional array that contains the average empirical error
            and the average true error for each m in the range accordingly.
        example: (10, 100, 5, 3, 100)
        """

        n_array = np.arange(m_first, m_last + 1, step)
        empirical_result = np.zeros(len(n_array))
        true_result = np.zeros(len(n_array))
        for idx, n in enumerate(n_array):
            sum_empirical = 0
            sum_true = 0
            for j in range(T):
                sample = self.sample_from_D(n)
                best_k_intervals, empirical_error = intervals.find_best_interval(sample[:, 0], sample[:, 1], k)
                sum_empirical += empirical_error
                sum_true += self.calculate_true_error(best_k_intervals)
            empirical_result[idx] = sum_empirical / (T * n)
            true_result[idx] = sum_true / T

        plt.plot(n_array, empirical_result, label='Empirical')
        plt.xlabel("n")
        plt.plot(n_array, true_result, label='True')
        plt.ylabel("Error")
        plt.legend()
        plt.show()

        return np.array([(empirical_result[i], true_result[i]) for i in range(len(n_array))])

    def experiment_k_range_erm(self, m, k_first, k_last, step):
        """Finds the best hypothesis for k= 1,2,...,10.
        Plots the empirical and true errors as a function of k.
        Input: m - an integer, the size of the data sample.
               k_first - an integer, the maximum number of intervals in the first experiment.
               m_last - an integer, the maximum number of intervals in the last experiment.
               step - an integer, the difference between the size of k in each experiment.

        Returns: The best k value (an integer) according to the ERM algorithm.
        """
        sample = self.sample_from_D(m)
        k_array = np.arange(k_first, k_last + 1, step)
        empirical_result = np.zeros(len(k_array))
        true_result = np.zeros(len(k_array))
        for idx, k in enumerate(k_array):
            best_k_intervals, empirical_result[idx] = intervals.find_best_interval(sample[:, 0], sample[:, 1], k)
            true_result[idx] = self.calculate_true_error(best_k_intervals)

        empirical_result /= m
        plt.plot(k_array, empirical_result, label='empirical')
        plt.xlabel("k")
        plt.plot(k_array, true_result, label='true')
        plt.ylabel("Error")
        plt.legend()
        plt.show()
        return k_array[np.argmin(empirical_result)]

    def experiment_k_range_srm(self, m, k_first, k_last, step):
        """Run the experiment in (c).
        Plots additionally the penalty for the best ERM hypothesis.
        and the sum of penalty and empirical error.
        Input: m - an integer, the size of the data sample.
               k_first - an integer, the maximum number of intervals in the first experiment.
               m_last - an integer, the maximum number of intervals in the last experiment.
               step - an integer, the difference between the size of k in each experiment.

        Returns: The best k value (an integer) according to the SRM algorithm.
        """
        delta = 0.1
        sample = self.sample_from_D(m)
        k_array = np.arange(k_first, k_last + 1, step)
        empirical_result = np.zeros(len(k_array))
        true_result = np.zeros(len(k_array))
        penalty_result = np.zeros(len(k_array))
        for idx, k in enumerate(k_array):
            best_k_intervals, empirical_result[idx] = intervals.find_best_interval(sample[:, 0], sample[:, 1], k)
            # we proved in the theoretical part that the dimension of H_k is 2k
            penalty_result[idx] = self.get_srm_penalty(2 * k, delta, m)
            true_result[idx] = self.calculate_true_error(best_k_intervals)

        empirical_result /= m
        penalty_empirical = np.add(penalty_result, empirical_result)

        plt.plot(k_array, empirical_result, label='Empirical')
        plt.plot(k_array, true_result, label='True')
        plt.plot(k_array, penalty_result, label='Penalty')
        plt.plot(k_array, penalty_empirical, label='Penalty + Empirical')
        plt.xlabel("k")
        plt.legend()
        plt.show()

        return k_array[np.argmin(penalty_empirical)]

    def cross_validation(self, m):
        """Finds a k that gives a good test error.
        Input: m - an integer, the size of the data sample.

        Returns: The best k value (an integer) found by the cross validation algorithm.
        """
        sample = self.sample_from_D(m)  # it is sorted
        np.random.shuffle(sample)  # shaffle it before divide to S1 and S2
        k_array = np.arange(1, 11)
        training_data = sample[:int(0.8 * len(sample))]  # 80% if for training
        training_data = np.array(sorted(training_data, key=lambda x: x[0]))  # sort the array by X value
        validation_data = sample[int(0.8 * len(sample)):]  # 20% for validation, doesn't matter if it is sorted
        empirical_result = np.zeros(len(k_array))
        holdout_errors = np.zeros(len(k_array))
        best_k_intervals_array = []
        for idx, k in enumerate(k_array):
            best_k_intervals, empirical_result[idx] = intervals.find_best_interval(training_data[:, 0],
                                                                                   training_data[:, 1], k)
            best_k_intervals_array.append(best_k_intervals)
            holdout_errors[idx] = self.get_holdout_error(best_k_intervals, validation_data[:, 0], validation_data[:, 1])

        return k_array[np.argmin(holdout_errors)]

    #################################
    # Place for additional methods

    #################################

    # get_y_based_on_x gets boolean with value true if x in ([0,0.2] or [0.4,0.6] or [0.8,1])
    def get_y_based_on_x(self, b):
        if b:
            return np.random.choice([0, 1], size=1, p=[0.2, 0.8])
        return np.random.choice([0.0, 1.0], size=1, p=[0.9, 0.1])

    def calculate_true_error(self, input_intervals):  # input_intervals = {[l_1,u_1],...,[l_k,u_k]}
        intervals_a = [(0, 0.2), (0.4, 0.6), (0.8, 1)]
        intervals_b = [(0.2, 0.4), (0.6, 0.8)]

        error = 0
        sum_a = 0
        sum_b = 0
        for interval in intervals_a:
            sum_a += self.calculate_intersection(input_intervals, interval)  # how much of intervals_a in input_intervals
        for interval in intervals_b:
            sum_b += self.calculate_intersection(input_intervals, interval)  # how much of intervals_b in input_intervals

        error += sum_a * 0.2  # P(y=0| h(x)=1, x in intervals_a)
        error += (0.6 - sum_a) * 0.8  # P(y=1| h(x)=0, x in intervals_a)
        error += sum_b * 0.9  # P(y=0| h(x)=1, x in intervals_b)
        error += (0.4 - sum_b) * 0.1  # P(y=1| h(x)=0, x in intervals_b)

        return error

    def calculate_intersection(self, input_intervals, prob_interval):
        sum = 0
        for interval in input_intervals:
            start = max(interval[0], prob_interval[0])
            end = min(interval[1], prob_interval[1])
            if start < end:
                sum += (end - start)  # we only add the part inside (prob_interval[0], prob_interval[1])
        return sum

    def get_srm_penalty(self, dimension, delta, n):
        numerator = dimension + math.log(2 / delta, math.e)
        return 2 * math.sqrt(numerator / n)

    def get_holdout_error(self, intervals, x_array, y_array):
        # x_array[i] should match y_array[i] otherwise we increment the error count
        error = 0
        for idx, x in enumerate(x_array):
            label_by_interval = 0
            for interval in intervals:
                start = interval[0]
                end = interval[1]
                if start < x < end:
                    label_by_interval = 1
                    break
            if label_by_interval != y_array[idx]:
                error += 1
        return error / len(x_array)


if __name__ == '__main__':
    ass = Assignment2()
    ass.experiment_m_range_erm(10, 100, 5, 3, 100)
    ass.experiment_k_range_erm(1500, 1, 10, 1)
    ass.experiment_k_range_srm(1500, 1, 10, 1)
    ass.cross_validation(1500)
