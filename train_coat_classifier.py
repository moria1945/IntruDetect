import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
import os

# --- 1. הגדרות בסיסיות ---
# הנתיב לתיקיית הדאטה-סט הספציפית למעיל
DATA_DIR = './train_data/coat_classifier'

# נתיב לשמירת המודל המאומן
MODEL_SAVE_PATH = './models/best_coat_model.pth'

# הגדרת DEVICE (CPU או GPU אם זמין)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using DEVICE: {DEVICE}")

# הגדרות אימון
BATCH_SIZE = 32
NUM_EPOCHS = 15
LEARNING_RATE = 0.001

# --- 2. טרנספורמציות לנתונים ---
# טרנספורמציות בסיסיות לתמונות: שינוי גודל, חיתוך למרכז, היפוך אקראי, המרה ל-Tensor, נורמליזציה
data_transforms = {
    'train': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

# --- 3. טעינת הדאטה-סט ---
# ImageFolder מניח שארגנת את התמונות בתיקיות לפי קטגוריות
image_datasets = {x: datasets.ImageFolder(os.path.join(DATA_DIR, x),
                                          data_transforms[x])
                  for x in ['train', 'val']}

# DataLoaders לטעינת תמונות בבאצ'ים (קבוצות)
dataloaders = {x: DataLoader(image_datasets[x], batch_size=BATCH_SIZE,
                                              shuffle=True, num_workers=4 if DEVICE.type == 'cuda' else 0)
               for x in ['train', 'val']}

dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
# שמות הקטגוריות (לדוגמה: ['with_coat', 'without_coat'])
class_names = image_datasets['train'].classes
print(f"Class names (coat): {class_names}")
print(f"Train dataset size (coat): {dataset_sizes['train']}")
print(f"Validation dataset size (coat): {dataset_sizes['val']}")

# --- 4. הגדרת המודל (ResNet18 מאומן מראש) ---
# נשתמש במודל ResNet18 מאומן מראש על ImageNet.
model = models.resnet18(pretrained=True)

# שינוי השכבה האחרונה כדי להתאים למספר הקטגוריות שלך (2: עם/בלי מעיל)
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, len(class_names))

model = model.to(DEVICE) # העבר את המודל ל-GPU או CPU

# הגדרת פונקציית הפסד ואופטימייזר
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

# --- 5. פונקציית אימון ---
def train_model(model, criterion, optimizer, num_epochs=NUM_EPOCHS):
    best_acc = 0.0

    for epoch in range(num_epochs):
        print(f'Epoch {epoch}/{num_epochs - 1}')
        print('-' * 10)

        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0

            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(DEVICE)
                labels = labels.to(DEVICE)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

            # שמור את המודל הטוב ביותר (לפי דיוק ולידציה)
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                torch.save(model.state_dict(), MODEL_SAVE_PATH)

        print()

    print(f'Training complete. Best validation Acc: {best_acc:.4f}')
    return model

# --- 6. הרצת האימון ---
if __name__ == '__main__':
    # ודא שתיקיית 'models/' קיימת
    if not os.path.exists(os.path.dirname(MODEL_SAVE_PATH)):
        os.makedirs(os.path.dirname(MODEL_SAVE_PATH))

    model_trained = train_model(model, criterion, optimizer, NUM_EPOCHS)
    print(f"Coat model saved as: {MODEL_SAVE_PATH}")