import torch 
import cv2 
import numpy as np 
import argparse # ספרייה לניתוח ארגומנטים משורת הפקודה
import sys # ספרייה לגישה למערכת ההפעלה, כולל קלט/פלט סטנדרטי
from PIL import Image # ספריית Pillow לעיבוד תמונות
from torchvision import transforms # מודול טרנספורמציות מ-PyTorch
from torchvision.models.detection import fasterrcnn_resnet50_fpn # מודל Faster R-CNN
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor # כלי להתאמת ראש הסיווג
from torchvision.models.detection.faster_rcnn import FasterRCNN_ResNet50_FPN_Weights # לייבוא משקלים מאומנים מראש 

sys.stdout.reconfigure(encoding='utf-8')
sys.stderr.reconfigure(encoding='utf-8')

# 1. הגדרת מודל Faster R-CNN וטעינת משקלים מאומנים
# מס הקלאסים
num_classes_weapons = 5 

# א. בניית ארכיטקטורת המודל
# יוצרת מודל Faster R-CNN לאיתור אובייקטים
model = fasterrcnn_resnet50_fpn(weights=FasterRCNN_ResNet50_FPN_Weights.COCO_V1)

# ב. התאמת המודל לאימון כלי נשק וכלי פריצה בלבד
in_features = model.roi_heads.box_predictor.cls_score.in_features
model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes_weapons)

# טעינת המודל המאומן
try:
    model_weapon_detection = model
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model_weapon_detection.load_state_dict(torch.load('fasterrcnn_weapon_best_epoch.pth', map_location=device))

    model_weapon_detection.to(device) # העבר את המודל למכשיר 
    model_weapon_detection.eval() # הגדר את המודל למצב הערכה 
    print("Faster R-CNN model loaded successfully.", file=sys.stderr) # הדפסה לשגיאות סטנדרטיות 

except FileNotFoundError:
    print("Error: Weapon detection weights file 'fasterrcnn_weapon_best_epoch.pth' not found.", file=sys.stderr)
    print("RESULTS_START") # סימון תחילת פלט עבור C++
    print("weapon:Error: Model not found.") # הודעת שגיאה מובנית
    print("RESULTS_END") # סימון סוף פלט עבור C++
    sys.exit(1) # יציאה עם קוד שגיאה
except Exception as e:
    print(f"Error loading weapon detection model: {e}", file=sys.stderr) # הדפסה לשגיאות סטנדרטיות 
    print("RESULTS_START") # סימון תחילת פלט עבור C++
    print(f"weapon:Error: {e}") # הודעת שגיאה מובנית
    print("RESULTS_END") # סימון סוף פלט עבור C++
    sys.exit(1) # יציאה עם קוד שגיאה

# 2. הגדרת ארגומנטים ופרמטרים
parser = argparse.ArgumentParser(description='Weapon Detection using Faster R-CNN.')
parser.add_argument('--image', type=str, help='Path to the input cropped image of a person.')
args = parser.parse_args()

# מפת הקלאסים 
# קלאס 0 = רקע 
weapon_classes = {
    1: 'אקדח', 
    2: 'רובה', 
    3: 'סכין',
    4: 'כלי פריצה' 
}

detection_threshold = 0.7 # סף רגישות כלי נשק 

# טוענת תמונה, מעבדת אותה כך שהמודל יוכל להבין אותה, מריצה את מודל זיהוי כלי הנשק, 
# ולבסוף מדווחת אם נמצא כלי נשק ספציפי ומהו סוגו, תוך התעלמות מזיהויים לא אמינים או מרקע
def detect_weapons(image_path):
    """
    Performs weapon detection on a given image path.
    מבצע זיהוי כלי נשק על נתיב תמונה נתון.
    """
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Could not load image from path: {image_path}", file=sys.stderr)
        return "Error: Image not found" # החזר הודעת שגיאה אם התמונה לא נטענה

    # המרה מ-BGR (OpenCV) ל-RGB (PyTorch)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_pil = Image.fromarray(img_rgb)

    # טרנספורמציה סטנדרטית למודלי זיהוי
    transform = transforms.Compose([transforms.ToTensor()])
    img_tensor = transform(img_pil).to(device)

    # ביצוע החיזוי
    with torch.no_grad(): 
        prediction = model_weapon_detection([img_tensor])

    detected_weapon_type = "None" # אם שום דבר לא זוהה, תחזיר "None"

    # ניתוח תוצאות החיזוי
    if prediction and len(prediction[0]['labels']) > 0:
        scores = prediction[0]['scores'].cpu().numpy() # ציוני ביטחון
        labels = prediction[0]['labels'].cpu().numpy() # תוויות הקלאסים

        # מצא את הזיהוי עם ציון הביטחון הגבוה ביותר מעל סף הזיהוי
        best_score = -1
        best_label_id = -1

        for i, score in enumerate(scores):
            # ודא שהקלאס אינו קלאס הרקע (תווית 0) ושעבר את סף הביטחון
            if score > detection_threshold and labels[i] != 0:
                if score > best_score:
                    best_score = score
                    best_label_id = labels[i]

        if best_label_id != -1: # אם נמצא זיהוי תקף
            if best_label_id in weapon_classes:
                detected_weapon_type = weapon_classes[best_label_id] # מצא את שם הקלאס בעברית
            else:
                detected_weapon_type = "Unknown Weapon Type" # קלאס זוהה אך לא קיים במפה

    return detected_weapon_type

# בלוק הרצה ראשי 
# מריץ את פונקציית זיהוי כלי הנשק על תמונה שצוינה, ומדפיס את סוג כלי הנשק שזוהה בפורמט מוגדר 
# שמתאים ליישום C++ אחר שצורך את הפלט הזה.
if __name__ == '__main__':
    # קבל את סוג כלי הנשק שזוהה
    weapon_type = detect_weapons(args.image)
    
    # הדפס את התוצאה בפורמט קריא עבור יישום C++
    print("RESULTS_START")
    print(f"weapon:{weapon_type}")
    print("RESULTS_END")

    sys.exit(0) # יציאה מוצלחת 