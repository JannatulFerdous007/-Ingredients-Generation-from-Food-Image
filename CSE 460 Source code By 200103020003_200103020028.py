# Required libreries
import os
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms
import torch
import torch.nn as nn
import torchvision.models as models
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
import streamlit as st

# Path to the image directory
image_dir = 'D:/Dataset/Recipes5k/images'

# Path to the text file of ingredients
ingredients_file = 'D:/Dataset/Recipes5k/annotations/ingredients_Recipes5k.txt'

# Store the datas
images = []
labels = []
ingredients = []

# Resize the images
transform = transforms.Compose([transforms.Resize((224, 224))])

# Load & read the ingredients
with open(ingredients_file, 'r') as f:
    all_ingredients = f.read().splitlines()

# The full dataset has 4,833 images but using the full dataset results in crashing the server, so here we defined a small number of samples.
# Define the number of samples
num_samples = 1000

# Iterate through all subdirectories & files for images, & then check for jpg files
for root, dirs, files in os.walk(image_dir):
    for i, file in enumerate(sorted(files)):
        if file.endswith('.jpg') and len(images) < num_samples:
            image_path = os.path.join(root, file)
            img = Image.open(image_path)
            img = transform(img)
            images.append(img)

            label = os.path.basename(os.path.dirname(image_path))
            labels.append(label)
            ingredient_list = all_ingredients[i].split(', ')
            ingredients.append(ingredient_list)

# Tokenize titles & ingredients
tokenized_titles = [title.split() for title in labels]
tokenized_ingredients = [ingredient for sublist in ingredients for ingredient in sublist]

# Calculate vocabulary sizes
vocab_size = len(set(token for sublist in tokenized_titles for token in sublist))
title_vocab_size = len(set(token for sublist in tokenized_titles for token in sublist))
ingredients_vocab_size = len(set(tokenized_ingredients))

# Print vocabulary sizes
print("Vocabulary Size:", vocab_size)
print("Title Vocabulary Size:", title_vocab_size)
print("Ingredients Vocabulary Size:", ingredients_vocab_size)

# Loaddinh pre-trained ResNet-50 model
resnet50 = models.resnet50(pretrained=True)

# Remove the final classification layer & then evaluate mode
resnet50 = nn.Sequential(*list(resnet50.children())[:-1])
resnet50.eval()

# Feature extraction  
def extract_features(image):
    with torch.no_grad():
        image_tensor = transforms.ToTensor()(image).unsqueeze(0)
        features = resnet50(image_tensor)
    return features.squeeze().numpy()

# Mapping & string convertion for titles & ingredients
label_to_index = {label: idx for idx, label in enumerate(set(labels))}
numerical_labels_titles = [label_to_index[label] for label in labels]

ingredient_to_index = {ingredient: idx for idx, ingredient in enumerate(set(tokenized_ingredients))}
numerical_ingredients = [[ingredient_to_index[ingredient] for ingredient in sublist] for sublist in ingredients]

# Convert features & labels to tensors for titles & ingredients
features_tensor_titles = torch.tensor([extract_features(img) for img in images])
numerical_labels_tensor_titles = torch.tensor(numerical_labels_titles) 

numerical_ingredients_flat = [item for sublist in numerical_ingredients for item in sublist]
numerical_ingredients_tensor = torch.tensor(numerical_ingredients_flat)

# Split data into training and testing sets for titles & ingredients
features_train_titles, features_test_titles, labels_train_titles, labels_test_titles = train_test_split(
    features_tensor_titles, numerical_labels_tensor_titles, test_size=0.2, random_state=42
)

features_train_ingredients, features_test_ingredients, labels_train_ingredients, labels_test_ingredients = train_test_split(
    features_tensor_titles, numerical_ingredients_tensor, test_size=0.2, random_state=42
)

# Separate branches for titles & ingredients
class FoodGenerator(nn.Module):
    def __init__(self, input_size, hidden_size, output_size_titles, output_size_ingredients):
        super(FoodGenerator, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2_titles = nn.Linear(hidden_size, output_size_titles)
        self.fc2_ingredients = nn.Linear(hidden_size, output_size_ingredients)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x_titles = self.fc2_titles(x)
        x_ingredients = self.fc2_ingredients(x)
        return x_titles, x_ingredients

# Initialize the model for titles
hidden_size_titles = 128
input_size_titles = features_tensor_titles.shape[1]
output_size_titles = len(set(numerical_labels_titles))

title_model = FoodGenerator(input_size_titles, hidden_size_titles, output_size_titles, output_size_titles)

# Loss function & optimizer for titles
criterion_titles = nn.CrossEntropyLoss()
optimizer_titles = torch.optim.Adam(title_model.parameters(), lr=0.001)

# Training loop for titles: loss, optimizer & monitorize the proces
num_epochs_titles = 100
for epoch in range(num_epochs_titles):
    outputs_titles, _ = title_model(features_train_titles)
    loss_titles = criterion_titles(outputs_titles, labels_train_titles)
    
    optimizer_titles.zero_grad()
    loss_titles.backward()
    optimizer_titles.step()
 
    print(f'Epoch [{epoch+1}/{num_epochs_titles}], Loss (Titles): {loss_titles.item():.4f}')

# Initialize the model for ingredients
hidden_size_ingredients = 128
input_size_ingredients = features_tensor_titles.shape[1]
output_size_ingredients = len(set(numerical_ingredients_flat))

ingredients_model = FoodGenerator(input_size_ingredients, hidden_size_ingredients, output_size_ingredients, output_size_ingredients)

# Loss function and optimizer for ingredients
criterion_ingredients = nn.CrossEntropyLoss()
optimizer_ingredients = torch.optim.Adam(ingredients_model.parameters(), lr=0.001)

# Training loop for ingredients: loss, optimizer & monitorize the proces
num_epochs_ingredients = 100
for epoch in range(num_epochs_ingredients): 
    _, outputs_ingredients = ingredients_model(features_train_ingredients) 
    loss_ingredients = criterion_ingredients(outputs_ingredients, labels_train_ingredients)

    optimizer_ingredients.zero_grad()
    loss_ingredients.backward()
    optimizer_ingredients.step()

    print(f'Epoch [{epoch+1}/{num_epochs_ingredients}], Loss (Ingredients): {loss_ingredients.item():.4f}')

# Evaluate the model on the test set for titles
with torch.no_grad():
    title_model.eval()
    test_outputs_titles, _ = title_model(features_test_titles)
    _, predicted_titles = torch.max(test_outputs_titles, 1)
    accuracy_titles = (predicted_titles == labels_test_titles).sum().item() / len(labels_test_titles)

print(f'Testing Accuracy (Titles): {accuracy_titles:.2f}')

# Evaluate the model on the test set for ingredients
with torch.no_grad():
    ingredients_model.eval()
    _, test_outputs_ingredients = ingredients_model(features_test_ingredients)
    _, predicted_ingredients = torch.max(test_outputs_ingredients, 1)
    accuracy_ingredients = (predicted_ingredients == labels_test_ingredients).sum().item() / len(labels_test_ingredients)

print(f'Testing Accuracy (Ingredients): {accuracy_ingredients:.2f}')

# Streamlit implementation

def generate_title_and_ingredients(uploaded_file, ingredients_file_path):
    image = Image.open(uploaded_file)

    # Preprocess the uploaded image for titles
    processed_image_titles = transform(image)
    processed_image_tensor_titles = torch.tensor(extract_features(processed_image_titles)).unsqueeze(0)

    # Predict titles
    with torch.no_grad():
        title_model.eval()
        prediction_titles, _ = title_model(processed_image_tensor_titles)
        _, predicted_label_titles = torch.max(prediction_titles, 1)

    # Map the predicted title back to the original class name
    predicted_title = [key for key, value in label_to_index.items() if value == predicted_label_titles.item()][0]

    # Preprocess the uploaded image for ingredients
    processed_image_ingredients = transform(image)
    processed_image_tensor_ingredients = torch.tensor(extract_features(processed_image_ingredients)).unsqueeze(0)

    # Predict ingredients
    with torch.no_grad():
        ingredients_model.eval()
        _, prediction_ingredients = ingredients_model(processed_image_tensor_ingredients)
        _, predicted_label_ingredients = torch.max(prediction_ingredients, 1)

    # Map the predicted ingredient back to the original class name
    predicted_ingredient = [key for key, value in ingredient_to_index.items() if value == predicted_label_ingredients.item()][0]

    return predicted_title, predicted_ingredient

# Set title & file upload
st.title("Ingredients Generation from Food Image")
uploaded_file = st.file_uploader("Choose a food image...", type=["jpg", "jpeg", "png"])

# Check, generate & display the predicted title & ingredients
if uploaded_file is not None:
    st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)

    predicted_title, predicted_ingredient = generate_title_and_ingredients(uploaded_file, ingredients_file)

    st.header("Predicted Title:")
    st.write(predicted_title)

    st.header("Predicted Ingredients:")
    st.write(f"The dish includes: {predicted_ingredient}")
    
