from jittor.utils.pytorch_converter import convert

file = open("../run.py", "r")
pytorch_code = file.read()
file.close()

jittor_code = convert(pytorch_code)
print(jittor_code)