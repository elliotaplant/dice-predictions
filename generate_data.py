import pandas as pd
import random
import argparse


def generate_data(n: int) -> pd.DataFrame:
    data = {'dice_sides': [], 'outcome': []}
    dice_options = range(1, 11)

    for _ in range(n):
        dice_sides = random.choice(dice_options)
        outcome = random.randint(1, dice_sides)
        data['dice_sides'].append(dice_sides)
        data['outcome'].append(outcome)

    df = pd.DataFrame(data)
    return df


def main(n: int):
    df = generate_data(n)
    df.to_csv('data/dice_data.csv', index=False)
    print(f'Dataset with {n} rows has been saved to dice_data.csv')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate a dataset for neural network training")
    parser.add_argument('--num_rows', type=int, default=1000,
                        help='Number of rows to generate')
    args = parser.parse_args()
    main(args.num_rows)
