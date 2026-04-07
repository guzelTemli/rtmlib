import openvino as ov

core = ov.Core()
model = core.read_model("openvino_format\\yolo11n_openvino_model.xml")

print("=== INPUTLAR ===")
for inp in model.inputs:
    print("isim   :", inp.get_any_name())
    print("shape  :", inp.partial_shape)
    print("dtype  :", inp.element_type)
    print("---")

print("=== OUTPUTLAR ===")
for out in model.outputs:
    print("isim   :", out.get_any_name())
    print("shape  :", out.partial_shape)
    print("dtype  :", out.element_type)
    print("---")