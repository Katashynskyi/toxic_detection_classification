# import torch
# import gradio as gr
# from transformers import DistilBertTokenizer
#
# from src.utils.utils_bert import DistilBERTClass
#
# # from your_module import DistilBERTClass  # Import your model class here
#
# # Load the trained model
# model_path='model/model_weights1e-05.pth'
# num_classes = 6
# model = DistilBERTClass(num_classes)
# model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
# model.eval()
#
# # Load the DistilBERT tokenizer
# tokenizer = DistilBertTokenizer.from_pretrained(
#     "distilbert-base-uncased", truncation=True, do_lower_case=True
# )
#
# # Define the prediction function
# def predict_text(input_text):
#     print("get text", input_text)
#     inputs = tokenizer(input_text, return_tensors="pt", padding=True, truncation=True)
#     outputs = model(**inputs)
#     print("get outputs", len(outputs))
#     # probabilities = torch.sigmoid(outputs).detach().numpy()[0]
#     print(outputs)
#     probabilities = torch.nn.functional.softmax(outputs, dim=1).squeeze().tolist()
#     print("prob", probabilities)
#     pred_class = torch.argmax(outputs, dim=1).item()
#     # return {f"Class {i}": probabilities[i] for i in range(num_classes)}
#     print(f'prob: {max(probabilities)}, {pred_class}' )
#
#     return f'{pred_class}' if max(probabilities) > 0.5 else 'non toxic'
#
#
# # Create a Gradio interface
# # iface = gr.Interface(
# #     fn=predict_text,
# #     inputs=gr.Textbox(),
# #     outputs=[gr.components.Label(num_top_classes=num_classes)]
# #
# #
# # )
#
# iface = gr.Interface(
#     fn=predict_text,
#     inputs='text',
#     outputs=gr.components.Label(num_top_classes=num_classes)
#
#
# )
#
# # Launch the Gradio interface
# iface.launch()
#
#
#
label2id = {
    "class1": 0,
    "class2": 1,
    "class3": 2,
    "class4": 3,
    "class5": 4,
    "class6": 5,
}
id2label = {v: k for k, v in label2id.items()}
# замість
# класс - твої
# класи
#
# та
# функція
# твоя
# щось
# типу
# такого


def predict_classes(text, threshold=0.5):
    # Tokenize the input text
    inputs = tokenizer(
        text, return_tensors="pt", truncation=True, padding=True, max_length=512
    )
    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]

    # Get the model's predictions
    with torch.no_grad():
        logits = model(input_ids=input_ids, attention_mask=attention_mask)

    # Convert logits to probabilities
    probs = torch.sigmoid(logits).squeeze().tolist()

    # Get classes that exceed the threshold
    predicted_labels = [id2label[i] for i, prob in enumerate(probs) if prob > threshold]

    # Return the classes and their associated probabilities
    return ", ".join(
        [f"{label} ({probs[label2id[label]]:.2%})" for label in predicted_labels]
    )
