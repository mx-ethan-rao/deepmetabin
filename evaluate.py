from tqdm import trange, tqdm
from absl import app, flags
from src.utils.util import (
    summary_bin_list_from_csv,
    evaluate,
    get_non_labeled_id_list_from_bin_lists,
)


class EvaluateManager:
    """Evaluation Manager, aimming to calculate the metrics of target
    method to compare with.

    Alerts:
        labels csv file should contains the length of contig
            as name in the first column each row.
    Args:
        gd_labels_path (string): ground truth labels path (csv file).
        result_labels_path (string): result labels path (csv file).
        dataset_name (string): dataset usage for logging.
        method_name (string): method usage for logging.
    """
    def __init__(
        self,
        gd_labels_path: str,
        result_labels_path: str,
        dataset_name: str,
        method_name: str,
        filter_threshold: int = 1000,
        **kwargs,
    ):
        self.gd_labels_path = gd_labels_path
        self.result_labels_path = result_labels_path
        self.dataset_name = dataset_name
        self.method_name = method_name
        self.filter_threshold = filter_threshold

    def evaluate_result(self):
        ground_truth_bins = summary_bin_list_from_csv(self.gd_labels_path)
        result_bins = summary_bin_list_from_csv(self.result_labels_path)
        non_labeled_list = get_non_labeled_id_list_from_bin_lists(
            gd_bin_list=ground_truth_bins,
            result_bin_list=result_bins,
        )
        precision, recall, ARI, F1 = evaluate(
            gd_bin_list=ground_truth_bins,
            result_bin_list=result_bins,
            non_labeled_id_list=non_labeled_list,
            unclassified=len(non_labeled_list),
        )
        print("{} method on {} dataset performance metrics are: \
            precision {}; recall {}; F1 {}; ARI {}; \
        ".format(self.method_name, self.dataset_name, precision, recall, F1, ARI))


def main(argv=None):
    evaluate_manager = EvaluateManager(**FLAGS.flag_values_dict())
    evaluate_manager.evaluate_result()


if __name__ == "__main__":
    FLAGS = flags.FLAGS

    flags.DEFINE_string("gd_labels_path", "", "")
    flags.DEFINE_string("result_labels_path", "", "")
    flags.DEFINE_string("dataset_name", "", "")
    flags.DEFINE_string("method_name", "", "")
    app.run(main)
