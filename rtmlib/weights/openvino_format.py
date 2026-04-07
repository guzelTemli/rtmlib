import openvino as ov

model = ov.convert_model("yolo11n.onnx")
ov.save_model(model, "openvino_format\\yolo11n_openvino_model.xml")