import numpy
import pandas as pda
x = numpy.array(["a","b"])

# print(x)

a = pda.Series([8,5,14,4])
b = pda.Series([8,5,14,4],index=["one","two","three","four"])
print(b)