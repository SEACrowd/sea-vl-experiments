import argparse

from image_duplicate_finder.duplicate_finder import ImageDuplicateFinder


def main():
    parser = argparse.ArgumentParser(
        description="Find duplicate and similar images using perceptual hashing techniques.",
    )

    parser.add_argument(
        "folder",
        help="Path to the folder containing images",
    )
    parser.add_argument(
        "--hash-type",
        default="phash",
        choices=[
            "ahash",
            "phash",
            "dhash",
            "whash-haar",
            "whash-db4",
            "colorhash",
        ],
        help="Type of hash function to use",
    )
    parser.add_argument(
        "--threshold",
        type=int,
        default=8,
        help="Threshold for hash difference to consider as similar",
    )
    parser.add_argument(
        "--csv",
        default=None,
        help="Path to save the CSV file with duplicate image groups",
    )
    parser.add_argument(
        "--remove",
        action="store_true",
        help="Remove duplicate images, keeping only one",
    )
    parser.add_argument(
        "--print",
        action="store_true",
        help="Print the duplicate image groups",
    )

    parser.add_argument(
        "--all",
        action="store_true",
        help="Print, save and remove duplicate image groups",
    )

    args = parser.parse_args()

    finder = ImageDuplicateFinder(
        hash_type=args.hash_type,
        threshold=args.threshold,
    )

    duplicate_groups = finder.find_duplicates(args.folder)

    if args.print:
        finder.print_results(duplicate_groups)

    if args.csv:
        finder.save_results_to_csv(duplicate_groups, args.csv)

    if args.remove:
        finder.remove_duplicates(args.folder, duplicate_groups)

    if args.all:
        finder.print_results(duplicate_groups)
        finder.save_results_to_csv(duplicate_groups, args.csv)
        finder.remove_duplicates(args.folder, duplicate_groups)


if __name__ == "__main__":
    main()
