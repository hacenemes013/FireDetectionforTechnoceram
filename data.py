import yaml

# Open and read the YAML file
with open("D:\Technoceram Yolo fire detection\dataset\data.yaml", "r") as file:
    data = yaml.safe_load(file)

print(data)