import onnx

model = onnx.load("rtmw-dw-x-l_simcc-cocktail14_270e-256x192_20231122.onnx")

print("=== INPUTLAR ===")
for inp in model.graph.input:
    print(inp.name)

print("=== OUTPUTLAR ===")
for out in model.graph.output:
    print(out.name)