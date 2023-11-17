import argparse

from Method_2_transformer.run_train_pipeline import TransformerModel

RANDOM_STATE = 42

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="")
    parser.add_argument(
        "--type_of_run", help='"train" of "inference"?', default="train"
    )
    parser.add_argument(
        "--path",
        help="Data path",
        default="D:/Programming/DB's/toxic_db_for_transformer/train.csv",  # Home-PC
        # default="D:/Programming/db's/toxicity_kaggle_1/train.csv",  # Work-PC
    )
    parser.add_argument(
        "--random_state", help="Choose seed for random state", default=RANDOM_STATE
    )
    parser.add_argument(
        # TODO: Max length of ???
        "--max_len",
        help="Max length of ???",
        default=128  # home_PC
        # default=512 # work_PC
    )
    parser.add_argument("--train_batch_size", help="Train batch size", default=16)
    parser.add_argument("--valid_batch_size", help="Valid batch size", default=16)
    parser.add_argument("--epochs", help="Number of epochs", default=0)
    parser.add_argument(
        "--learning_rate", help="Learning rate", default=1e-05
    )  # 0.001, 0.005, 0.01, 0.05, 0.1
    parser.add_argument("--n_samples", help="How many samples to pass?", default=800)
    parser.add_argument(
        "--threshold", help="What's the threshold for toxicity?", default=0.5
    )
    parser.add_argument(
        "--num_classes", help="Choose number of classes to predict", default=6
    )
    args = parser.parse_args()
    if args.type_of_run == "train":
        classifier = TransformerModel(
            path=args.path,
            n_samples=args.n_samples,
            random_state=args.random_state,
            max_len=args.max_len,
            train_batch_size=args.train_batch_size,
            valid_batch_size=args.valid_batch_size,
            epochs=args.epochs,
            learning_rate=args.learning_rate,
            threshold=args.threshold,
            num_classes=args.num_classes,
        )

        classifier.train()
        classifier.predict()
        for eval_type in ["train", "valid", "test"]:
            classifier.evaluate(type_=eval_type)
