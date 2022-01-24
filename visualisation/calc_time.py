nPE = 264 
nSlice = 44

TR = 2.31

overPE = 0.10
overSlice = 0.00

resPE = 0.50
resSlice = 0.64

actualPE = round(nPE * (1+overPE) * resPE)
totalTR = actualPE * TR
actualSlice = round(nSlice * resSlice)
totalTime = totalTR * actualSlice

print(actualPE)
print(round(totalTime / 1000, 2))