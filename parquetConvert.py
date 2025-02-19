import pandas as pd

df = pd.read_parquet("training.parquet")

df.to_csv("output.txt", sep="\t", index=False)