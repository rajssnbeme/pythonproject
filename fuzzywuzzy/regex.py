import re
pat = re.compile("^[O][0-9]{5}$")
#pattern.match("a1")
test = input("Enter the string: ")
if re.fullmatch(pat, test):
    print(f"'{test}' is an order number!")
else:
    print(f"'{test}' is NOT an order number!")