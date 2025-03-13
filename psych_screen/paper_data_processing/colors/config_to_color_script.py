import json

# Input data (provided in the question)
input_data = [
    {"path": ["Manually Annotated Layers"], "color": [255, 255, 255]},
    {"path": ["Manually Annotated Layers", "WM"], "color": [27, 27, 27]},
    {"path": ["Manually Annotated Layers", "L1"], "color": [223, 54, 126]},
    {"path": ["Manually Annotated Layers", "L2"], "color": [76, 121, 180]},
    {"path": ["Manually Annotated Layers", "L3"], "color": [98, 174, 82]},
    {"path": ["Manually Annotated Layers", "L4"], "color": [145, 78, 161]},
    {"path": ["Manually Annotated Layers", "L5"], "color": [247, 220, 56]},
    {"path": ["Manually Annotated Layers", "L6"], "color": [239, 140, 40]},
    {"path": ["BayesSpace (k=9)"], "color": [0, 0, 0]},
    {"path": ["BayesSpace (k=9)", "1"], "color": [88, 81, 87]},
    {"path": ["BayesSpace (k=9)", "2"], "color": [228, 224, 226]},
    {"path": ["BayesSpace (k=9)", "3"], "color": [229, 71, 55]},
    {"path": ["BayesSpace (k=9)", "4"], "color": [239, 34, 244]},
    {"path": ["BayesSpace (k=9)", "5"], "color": [105, 252, 77]},
    {"path": ["BayesSpace (k=9)", "6"], "color": [83, 120, 248]},
    {"path": ["BayesSpace (k=9)", "7"], "color": [243, 182, 54]},
    {"path": ["BayesSpace (k=9)", "8"], "color": [163, 37, 102]},
    {"path": ["BayesSpace (k=9)", "9"], "color": [111, 251, 207]},
    {"path": ["BayesSpace (k=16)"], "color": [0, 0, 0]},
    {"path": ["BayesSpace (k=16)", "1"], "color": [88, 81, 87]},
    {"path": ["BayesSpace (k=16)", "2"], "color": [228, 224, 226]},
    {"path": ["BayesSpace (k=16)", "3"], "color": [229, 71, 55]},
    {"path": ["BayesSpace (k=16)", "4"], "color": [239, 34, 244]},
    {"path": ["BayesSpace (k=16)", "5"], "color": [105, 252, 77]},
    {"path": ["BayesSpace (k=16)", "6"], "color": [83, 120, 248]},
    {"path": ["BayesSpace (k=16)", "7"], "color": [243, 182, 54]},
    {"path": ["BayesSpace (k=16)", "8"], "color": [163, 37, 102]},
    {"path": ["BayesSpace (k=16)", "9"], "color": [111, 251, 207]},
    {"path": ["BayesSpace (k=16)", "10"], "color": [147, 174, 50]},
    {"path": ["BayesSpace (k=16)", "11"], "color": [106, 210, 252]},
    {"path": ["BayesSpace (k=16)", "12"], "color": [217, 159, 251]},
    {"path": ["BayesSpace (k=16)", "13"], "color": [162, 1, 248]},
    {"path": ["BayesSpace (k=16)", "14"], "color": [237, 166, 161]},
    {"path": ["BayesSpace (k=16)", "15"], "color": [63, 85, 152]},
    {"path": ["BayesSpace (k=16)", "16"], "color": [183, 83, 39]},
    {"path": ["BayesSpace (k=28)"], "color": [0, 0, 0]},
    {"path": ["BayesSpace (k=28)", "1"], "color": [88, 81, 87]},
    {"path": ["BayesSpace (k=28)", "2"], "color": [228, 224, 226]},
    {"path": ["BayesSpace (k=28)", "3"], "color": [229, 71, 55]},
    {"path": ["BayesSpace (k=28)", "4"], "color": [239, 34, 244]},
    {"path": ["BayesSpace (k=28)", "5"], "color": [105, 252, 77]},
    {"path": ["BayesSpace (k=28)", "6"], "color": [83, 120, 248]},
    {"path": ["BayesSpace (k=28)", "7"], "color": [243, 182, 54]},
    {"path": ["BayesSpace (k=28)", "8"], "color": [163, 37, 102]},
    {"path": ["BayesSpace (k=28)", "9"], "color": [111, 251, 207]},
    {"path": ["BayesSpace (k=28)", "10"], "color": [147, 174, 50]},
    {"path": ["BayesSpace (k=28)", "11"], "color": [106, 210, 252]},
    {"path": ["BayesSpace (k=28)", "12"], "color": [217, 159, 251]},
    {"path": ["BayesSpace (k=28)", "13"], "color": [162, 1, 248]},
    {"path": ["BayesSpace (k=28)", "14"], "color": [237, 166, 161]},
    {"path": ["BayesSpace (k=28)", "15"], "color": [63, 85, 152]},
    {"path": ["BayesSpace (k=28)", "16"], "color": [183, 83, 39]},
    {"path": ["BayesSpace (k=28)", "17"], "color": [58, 129, 88]},
    {"path": ["BayesSpace (k=28)", "18"], "color": [73, 29, 8]},
    {"path": ["BayesSpace (k=28)", "19"], "color": [167, 31, 159]},
    {"path": ["BayesSpace (k=28)", "20"], "color": [245, 233, 71]},
    {"path": ["BayesSpace (k=28)", "21"], "color": [14, 38, 45]},
    {"path": ["BayesSpace (k=28)", "22"], "color": [233, 56, 134]},
    {"path": ["BayesSpace (k=28)", "23"], "color": [234, 58, 189]},
    {"path": ["BayesSpace (k=28)", "24"], "color": [243, 229, 165]},
    {"path": ["BayesSpace (k=28)", "25"], "color": [185, 120, 164]},
    {"path": ["BayesSpace (k=28)", "26"], "color": [117, 32, 178]},
    {"path": ["BayesSpace (k=28)", "27"], "color": [181, 245, 58]},
    {"path": ["BayesSpace (k=28)", "28"], "color": [192, 203, 253]},
]


# Function to convert RGB to hex
def rgb_to_hex(rgb):
    return "#{:02x}{:02x}{:02x}".format(rgb[0], rgb[1], rgb[2])


# Prepare the output structure
output_data = {"sets": []}

# Process input data and organize by "path"
for item in input_data:
    path = item["path"]
    color = item["color"]
    set_name = path[0]  # First part of "path" is the setName
    label = path[1] if len(path) > 1 else ""  # Second part is the label if it exists
    color_hex = rgb_to_hex(color)

    # Check if the setName already exists
    set_exists = next(
        (s for s in output_data["sets"] if s["setName"] == set_name), None
    )

    if set_exists:
        # Add color to the existing set
        set_exists["colors"].append({"label": label, "hex": color_hex})
    else:
        # Create a new set and add the first color
        output_data["sets"].append(
            {"setName": set_name, "colors": [{"label": label, "hex": color_hex}]}
        )

# Print the final output in JSON format
print(json.dumps(output_data, indent=4))
