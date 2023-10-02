# setting path
import auxiliarymethods.auxiliary_methods as aux
from auxiliarymethods.datasets import get_dataset
import kernel_models as kb
from auxiliarymethods.kernel_evaluation import kernel_svm_evaluation
from auxiliarymethods.kernel_evaluation import linear_svm_evaluation


def main():
    dataset = [
        # ["ENZYMES", True],
        # ["IMDB-BINARY", False],
        # ["IMDB-MULTI", False],
        # ["PROTEINS", True],
        # ["REDDIT-BINARY", False],
        # ["PTC_FM", False]
    ]

    # Number of repetitions of 10-CV.
    num_reps = 5

    results = []
    for dataset, use_labels in dataset:
        classes = get_dataset(dataset)

        # 1-WL kernel, number of iterations in [1:6].
        all_matrices = []
        for i in range(1, 6):
            gm = kb.compute_wl_1_dense(dataset, i, use_labels, False)
            gm_n = aux.normalize_gram_matrix(gm)
            all_matrices.append(gm_n)
        acc, s_1 = kernel_svm_evaluation(
            all_matrices, classes, num_repetitions=num_reps, all_std=False
        )
        print(dataset + " " + "WL1 " + str(acc) + " " + str(s_1))
        results.append(dataset + " " + "WL1 " + str(acc) + " " + str(s_1))

        # WLOA kernel, number of iterations in [1:6].
        all_matrices = []
        for i in range(1, 6):
            gm = kb.compute_wloa_dense(dataset, i, use_labels, False)
            gm_n = aux.normalize_gram_matrix(gm)
            all_matrices.append(gm_n)
        acc, s_1 = kernel_svm_evaluation(
            all_matrices, classes, num_repetitions=num_reps, all_std=False
        )
        print(dataset + " " + "WLOA " + str(acc) + " " + str(s_1))
        results.append(dataset + " " + "WLOA " + str(acc) + " " + str(s_1))

        # Graphlet kernel.
        all_matrices = []
        gm = kb.compute_graphlet_dense(dataset, use_labels, False)
        gm_n = aux.normalize_gram_matrix(gm)
        all_matrices.append(gm_n)
        acc, s_1 = kernel_svm_evaluation(
            all_matrices, classes, num_repetitions=num_reps, all_std=False
        )
        print(dataset + " " + "GR " + str(acc) + " " + str(s_1))
        results.append(dataset + " " + "GR " + str(acc) + " " + str(s_1))

        # Shortest-path kernel.
        all_matrices = []
        gm = kb.compute_shortestpath_dense(dataset, use_labels)
        gm_n = aux.normalize_gram_matrix(gm)
        all_matrices.append(gm_n)
        acc, s_1 = kernel_svm_evaluation(
            all_matrices, classes, num_repetitions=num_reps, all_std=False
        )
        print(dataset + " " + "SP " + str(acc) + " " + str(s_1))
        results.append(dataset + " " + "SP " + str(acc) + " " + str(s_1))

    for r in results:
        print(r)


if __name__ == "__main__":
    main()
