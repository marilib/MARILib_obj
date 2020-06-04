import pandas as pd

dtf = pd.read_csv("explore_design.txt",sep=";")

print(dtf.head())
print(str(type(dtf.iloc[0,2])) + "  -->  " + str(dtf.iloc[0,2]))

dtf.to_csv("explore_design_pandas.csv",sep=";")
