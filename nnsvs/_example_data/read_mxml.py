from nnmnkwii.io import hts
import pysinsy

filepath = "/home/nishizawa/nnsvs/nnsvs/_example_data/"
filename = "1st_color_sub"
contexts = pysinsy.extract_fullcontext(filepath + filename + ".musicxml")
# print(contexts)
labels = hts.HTSLabelFile.create_from_contexts(contexts)
print("----------------------")
with open(filepath + filename + ".txt", "w") as f:
    f.write(str(labels))