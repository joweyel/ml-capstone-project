from flask import Flask
from flask import request, jsonify
from PIL import Image
import torch
from torchvision import transforms as T
from DINOClassifier import DINOClassifier

lbl_mapping = { 0: "Bike", 1: "Car" }

model_cfg = {
    "model_type": "s",
    "hidden_layer_dims": [64],
    "use_dropout": False, 
    "dropout_prob": 0.0, 
    "device": "cpu"
}

model = DINOClassifier(**model_cfg)

model_file_pth = "model_v1.pth"
model.load_state_dict(torch.load(model_file_pth).state_dict())

model = model.to("cpu")
model.eval()


preprocess = T.Compose([
    T.Resize((224, 224)),  # Resize to the same size
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406],
                [0.229, 0.224, 0.225]),  # ImageNet statistics
])

app = Flask("vehicles")

@app.route("/predict", methods=["POST"])
def predict():
    # Obtaining Image data
    img_data = request.files['file']
    pil_img = Image.open(img_data)
    # Preprocessing for torch model
    img = preprocess(pil_img).unsqueeze(0)

    # Prediction already probability
    with torch.no_grad():
        pred, _ = model(img)
    pred_cls = int(torch.round(pred).item())
    pred_cls_name = lbl_mapping[pred_cls]

    response_pred = float(pred.item())

    response = {
        "prob": response_pred,
        "pred_class_id": pred_cls,
        "pred_class_name": pred_cls_name
    }

    return jsonify(response)


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=9696) 