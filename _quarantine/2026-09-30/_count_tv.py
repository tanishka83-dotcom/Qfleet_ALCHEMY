import re
text = open("config.py").read()
items = re.findall(r'"TV-(\d+)', text)
unique = sorted(set(items), key=int)
print(f"TODO_VERIFY count: {len(unique)}")
print("Items:", unique)
