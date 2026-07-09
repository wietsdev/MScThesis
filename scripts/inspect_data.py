from datasets import load_dataset

N = 4  # num examples to print per split


def print_mcq(row):
    print(f"  Q: {row['Question']}")
    for option in row["Options"].split("--------------------"):
        option = option.strip()
        if option:
            print(f"     {option}")
    print(f"  Answer: {row['Answer']}")
    print(f"  Complexity: {row['Complexity']}")


def print_cloze(row):
    print(f"  Q: {row['Question']}")
    print(f"  Answer: {row['Answer']}")


def main():
    print("Loading ClimaQA-Gold...")
    ds = load_dataset("Rose-STL-Lab/ClimaQA", "Gold")

    print(f"\nSplits: {list(ds.keys())}")
    print(f"  mcq:   {len(ds['mcq'])} items")
    print(f"  cloze: {len(ds['cloze'])} items")
    print(f"  ffq:   {len(ds['ffq'])} items")

    print(f"\n--- MCQ (first {N}) ---")
    for i, row in enumerate(ds["mcq"].select(range(N))):
        print(f"\n[{i}]")
        print_mcq(row)

    print(f"\n--- Cloze (first {N}) ---")
    for i, row in enumerate(ds["cloze"].select(range(N))):
        print(f"\n[{i}]")
        print_cloze(row)


if __name__ == "__main__":
    main()
