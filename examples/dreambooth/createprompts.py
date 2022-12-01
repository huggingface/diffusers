import json

f = open("C:/Users/saeed3a/Documents/repos/diffusers/examples/dreambooth/prompts")
promptdict = json.load(f)

print(promptdict.get("2.jpg"))
