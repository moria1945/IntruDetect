# train_clothing_classifier.py
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
import os

# --- 1. הגדרות בסיסיות ---
# נתיב לתיקיית הדאטה-סט שלך
DATA_DIR = './my_clothing_classifier' # ודא שזו התיקייה שבה נמצאות train/ ו-val/

# הגדרת DEVICE (CPU או GPU אם זמין)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"משתמש ב- DEVICE: {DEVICE}")

# הגדרות אימון
BATCH_SIZE = 32
NUM_EPOCHS = 10 # אפשר להתחיל מ-10 ולראות אם זה מספיק, אחר כך להעלות
LEARNING_RATE = 0.001

# --- 2. טרנספורמציות לנתונים ---
# טרנספורמציות בסיסיות לתמונות: שינוי גודל, חיתוך אקראי (לאימון), המרה ל-Tensor, נורמליזציה
# הערה: ערכי הנורמליזציה (mean ו-std) הם עבור ImageNet, מתאימים למודלים מאומנים מראש.
# אם התמונות שלך שונות מאוד, ייתכן שתרצי לחשב ערכים אלה עבור הדאטה-סט שלך.
data_transforms = {
    'train': transforms.Compose([
        transforms.Resize(256),         # שנה גודל ל-256 פיקסלים
        transforms.CenterCrop(224),     # חתוך למרכז 224x224
        transforms.RandomHorizontalFlip(), # היפוך אקראי אופקי (לגיוון הדאטה)
        transforms.ToTensor(),          # המר ל-PyTorch Tensor
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]) # נורמליזציה
    ]),
    'val': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

# --- 3. טעינת הדאטה-סט ---
# ImageFolder מניח שארגנת את התמונות בתיקיות לפי קטגוריות (כפי שהומלץ)
image_datasets = {x: datasets.ImageFolder(os.path.join(DATA_DIR, x),
                                          data_transforms[x])
                  for x in ['train', 'val']}

# DataLoaders לטעינת תמונות בבאצ'ים (קבוצות)
dataloaders = {x: DataLoader(image_datasets[x], batch_size=BATCH_SIZE,
                                              shuffle=True, num_workers=4 if DEVICE.type == 'cuda' else 0) # num_workers יכול להיות 0 או 2 במעבד (CPU)
               for x in ['train', 'val']}

dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
class_names = image_datasets['train'].classes # שמות הקטגוריות (לדוגמה: ['with_coat', 'without_coat'])
print(f"שמות הקטגוריות: {class_names}")
print(f"גודל סט אימון: {dataset_sizes['train']}")
print(f"גודל סט ולידציה: {dataset_sizes['val']}")

# --- 4. הגדרת המודל (ResNet18 מאומן מראש) ---
# נשתמש במודל ResNet18 מאומן מראש על ImageNet. זה עוזר מאוד כי הוא כבר למד פיצ'רים כלליים.
model = models.resnet18(pretrained=True)

# שינוי השכבה האחרונה (Fully Connected layer) כדי להתאים למספר הקטגוריות שלך
# in_features הוא מספר הפיצ'רים שהשכבה האחרונה במודל המקורי מקבלת
num_ftrs = model.fc.in_features
# out_features הוא מספר הקטגוריות שלך (2 עבור מעיל, 2 עבור חולצה לבנה)
model.fc = nn.Linear(num_ftrs, len(class_names))

model = model.to(DEVICE) # העבר את המודל ל-GPU או CPU

# הגדרת פונקציית הפסד ואופטימייזר
criterion = nn.CrossEntropyLoss() # מתאים לסיווג מרובה קטגוריות
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE) # אופטימייזר Adam נפוץ ויעיל

# --- 5. פונקציית אימון ---
def train_model(model, criterion, optimizer, num_epochs=NUM_EPOCHS):
    best_acc = 0.0 # עקוב אחר הדיוק הטוב ביותר

    for epoch in range(num_epochs):
        print(f'Epoch {epoch}/{num_epochs - 1}')
        print('-' * 10)

        # לכל Epoch יש שלבי אימון וולידציה
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # העבר את המודל למצב אימון
            else:
                model.eval()   # העבר את המודל למצב הערכה (ללא עדכוני משקולות)

            running_loss = 0.0
            running_corrects = 0

            # חזור על הנתונים
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(DEVICE)
                labels = labels.to(DEVICE)

                # איפוס גרדיאנטים
                optimizer.zero_grad()

                # forward
                # מעקב אחר היסטוריה רק בשלב האימון
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1) # קבל את הקטגוריה עם הניקוד הגבוה ביותר
                    loss = criterion(outputs, labels)

                    # backward + optimize רק אם בשלב האימון
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # סטטיסטיקות
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

            # שמור את המודל הטוב ביותר (לפי דיוק ולידציה)
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                torch.save(model.state_dict(), f'./my_clothing_classifier/best_clothing_model_{phase}.pth')

        print()

    print(f'אימון הושלם. דיוק ולידציה הטוב ביותר: {best_acc:.4f}')
    return model

# --- 6. הרצת האימון ---
if __name__ == '__main__':
    # מודל עבור מעיל
    # ודאי שהדאטה-סט שלך מכיל תיקיות 'with_coat' ו-'without_coat'
    print("\n--- אימון מודל זיהוי מעיל ---")
    # שנה את ה-DATA_DIR לתיקייה הספציפית למעיל אם בנית אותה בנפרד
    # (לדוגמה, './my_clothing_classifier/coat_data')
    # אם בנית הכל באותה תיקייה, ודא שבתיקיות ה-train וה-val יש רק את הקטגוריות הרצויות לאימון הנוכחי.
    # ייתכן שתצטרכי להריץ את סקריפט האימון פעמיים, פעם אחת לכל מודל.
    # או להגדיר את ImageFolder כך שיקרא רק את הקטגוריות הרצויות.
    # הדרך הפשוטה ביותר לבינתיים היא להעביר את התמונות של "מעיל" לתיקיות 'with_coat' ו-'without_coat'
    # ולהעביר את התמונות של "חולצה" לתיקיות 'white_shirt' ו-'non_white_shirt'.

    # נניח שאת מריצה אימון נפרד לכל קטגוריה:
    # 1. אימון למעיל:
    #    הדאטה-סט שלך ב-my_clothing_classifier/train/ ו-val/ צריך להכיל רק:
    #    - with_coat
    #    - without_coat
    #    ואז תריצי את הסקריפט. קובץ המודל יישמר כ-best_clothing_model_val.pth
    #    אפשר לשנות את שם השמירה ל-best_coat_model.pth

    # 2. אימון לחולצה לבנה:
    #    אחרי האימון למעיל, תשני את התמונות בתיקיות train/ ו-val/ שיכילו רק:
    #    - white_shirt
    #    - non_white_shirt
    #    ואז תריצי שוב את הסקריפט. קובץ המודל יישמר שוב, אז תצטרכי לשנות את השם
    #    הדרך הטובה יותר היא ליצור שני סקריפטים נפרדים או קוד שמטפל בשני מודלים.

    # כדי להפשט:
    # צרי 2 תיקיות אימון:
    # my_clothing_classifier/coat_data/train/with_coat/...
    # my_clothing_classifier/coat_data/train/without_coat/...
    # my_clothing_classifier/shirt_data/train/white_shirt/...
    # my_clothing_classifier/shirt_data/train/non_white_shirt/...

    # ואז, עבור מודל מעיל:
    # DATA_DIR = './my_clothing_classifier/coat_data'
    # ואז קוד האימון
    # שמור: torch.save(model.state_dict(), './my_clothing_classifier/best_coat_model.pth')

    # ועבור מודל חולצה:
    # DATA_DIR = './my_clothing_classifier/shirt_data'
    # ואז קוד האימון
    # שמור: torch.save(model.state_dict(), './my_clothing_classifier/best_white_shirt_model.pth')


    # לצורך הדוגמה כאן, נניח שה-DATA_DIR הנוכחי מכיל רק קטגוריות שמתאימות ל-2 קטגוריות בלבד (לדוגמה, עם/בלי מעיל או חולצה לבנה/לא לבנה).
    # ודא שבתיקיות train/val יש רק 2 תיקיות קטגוריה שרלוונטיות למודל אחד.
    model_trained = train_model(model, criterion, optimizer, NUM_EPOCHS)
    torch.save(model_trained.state_dict(), './my_clothing_classifier/final_clothing_classifier.pth') # שמירה סופית
    print("מודל נשמר כ- final_clothing_classifier.pth")