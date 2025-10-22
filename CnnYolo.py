import torch
import torch.nn as nn
import cv2
import numpy as np
import argparse
import sys
from PIL import Image
from torchvision import transforms

sys.stdout.reconfigure(encoding='utf-8')
sys.stderr.reconfigure(encoding='utf-8')
# הגדרת DEVICE
DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
print(f"CnnYolo.py: user in- DEVICE: {DEVICE}", file=sys.stderr)

# 1. בניית מודל סיווג לבוש 

class SimpleClothingClassifier(nn.Module):
    # הגדרת המבנה הפנימי של מודל רשת נוירונים פשוטה לסיווג ביגוד, הכולל שכבת קונבולוציה ראשונית לזיהוי מאפיינים, 
    # שכבת איגום להקטנת מידות, ושכבה לינארית סופית שמוציאה חיזוי עבור מספר הקטגוריות (סוגי הביגוד) הרצוי.
    def __init__(self, num_classes):
        super(SimpleClothingClassifier, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        # התאמה של גודל ה-input ל-fc בהתאם לגודל התמונה שמזינים למודל ולשכבות הקונבולוציה.
        # אם קרופים הם 64x64, אחרי 1 maxpool (מחלק ב-2) זה יהיה 32x32.
        # אז הגודל הוא 16 * 32 * 32.
        self.fc = nn.Linear(16 * 32 * 32, num_classes) # **וודא שזה מתאים לגודל ה-input האמיתי שלך!**
        # אין צורך ב-Softmax אם משתמשים ב-CrossEntropyLoss לאימון (נפוץ לסיווג רב-קלאסי)
        # אם המודל אומן עם Loss Function אחר שדורש הסתברויות, ייתכן שתצטרכי להוסיף softmax.
        # לצורך הסקה, torch.max() יעבוד גם על logits (ללא softmax).

# תיאור זרימת הנתונים בתוך המודל
    def forward(self, x):
        x = self.pool1(self.relu1(self.conv1(x)))
        x = x.view(-1, 16 * 32 * 32) # Flatten
        x = self.fc(x)
        return x # מחזיר logits, לא הסתברויות (נפוץ ב-CrossEntropyLoss)

# מפת קלאסים של לבוש (על פי הקלאסים המומלצים החדשים)
# יש לוודא שהאינדקסים כאן תואמים בדיוק לאינדקסים שהמודל שלך יפיק לאחר האימון.
clothing_classes_map = {
    0: 'מעיל',
    1: 'חולצה לבנה',
    2: 'שרוולים קצרים',
    3: 'שרוולים ארוכים',
    4: 'אחר'
}
NUM_CLOTHING_CLASSES = len(clothing_classes_map) # מס' הקטגוריות

# טיעינה והכנה מודל רשת נוירונים (CNN) שאומן מראש לזיהוי סוגי ביגוד,
#  כולל טעינת המשקולות שלו, העברתו למצב הערכה,
#  והגדרת טרנספורמציות נחוצות לתמונות לפני שהן יועברו למודל לצורך חיזוי, ובמקביל מטפל בשגיאות 
try:
    model_clothing = SimpleClothingClassifier(NUM_CLOTHING_CLASSES)
    model_clothing.load_state_dict(torch.load('path/to/your/clothing_model_weights.pth', map_location=DEVICE))
    model_clothing.to(DEVICE)
    model_clothing.eval() # הגדר למצב הערכה
    print("CnnYolo.py: מודל סיווג לבוש נטען בהצלחה.", file=sys.stderr)
except FileNotFoundError:
    print("CnnYolo.py: שגיאה: קובץ משקולות מודל לבוש לא נמצא. אנא ודא/י את הנתיב.", file=sys.stderr)
    print("RESULTS_START")
    print("clothing:Error: Model not found.")
    print("RESULTS_END")
    sys.exit(1)
except Exception as e:
    print(f"CnnYolo.py: שגיאה בטעינת מודל סיווג לבוש: {e}", file=sys.stderr)
    print("RESULTS_START")
    print(f"clothing:Error: {e}")
    print("RESULTS_END")
    sys.exit(1)

# טרנספורמציות לתמונה לפני הזנה למודל סיווג לבוש (התאם לגודל הקלט של המודל שלך)
transform_clothing = transforms.Compose([
    transforms.Resize((64, 64)), # **ודא שזה הגודל שהמודל שלך אומן עליו!**
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.229, 0.229]) # נורמליזציה של ImageNet
])

# ממירה מחרוזת המייצגת קואורדינטות של תיבת תוחמת לרשימה של מספרים שלמים
def parse_bbox_string(bbox_str):
    """ממיר מחרוזת bbox (לדוגמה: "x1,y1,x2,y2") לרשימת אינטים."""
    if not bbox_str:
        return None
    try:
        return [int(x) for x in bbox_str.split(',')]
    except ValueError:
        print(f"CnnYolo.py: שגיאה בפענוח bbox: {bbox_str}", file=sys.stderr)
        return None

# לוקחת מחרוזת המכילה מספר קואורדינטות של תיבות תוחמות וממירה אותה לרשימה של רשימות של מספרים שלמים
def parse_all_bboxes_string(all_bboxes_str):
    """ממיר מחרוזת של מספר bboxes (לדוגמה: "x1,y1,x2,y2;X1,Y1,X2,Y2") לרשימת רשימות."""
    if not all_bboxes_str:
        return []
    bboxes = []
    for bbox_s in all_bboxes_str.split(';'):
        parsed = parse_bbox_string(bbox_s)
        if parsed:
            bboxes.append(parsed)
    return bboxes

# בודקת אם שתי תיבות תוחמות קרובות מספיק זו לזו, הן מבחינת מיקום הפינות שלהן והן מבחינת הדמיון בשטחן
def is_bbox_within_tolerance(bbox1, bbox2, tolerance_pixels=5, tolerance_area_ratio=0.2):
    """בודק האם שני BBoxes קרובים מספיק זה לזה."""
    # השוואה לפי פינות 
    x1_1, y1_1, x2_1, y2_1 = bbox1
    x1_2, y1_2, x2_2, y2_2 = bbox2

    # בודק אם הפינות קרובות בטווח הפיקסלים
    if not (abs(x1_1 - x1_2) <= tolerance_pixels and \
            abs(y1_1 - y1_2) <= tolerance_pixels and \
            abs(x2_1 - x2_2) <= tolerance_pixels and \
            abs(y2_1 - y2_2) <= tolerance_pixels):
        return False
    
    # בודק גם דמיון בשטח (מונע התאמה של תיבה קטנה בתוך גדולה בטעות)
    area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
    area2 = (x2_2 - x1_2) * (y2_2 - y1_2)

    if min(area1, area2) == 0: # אם אחת מהתיבות היא בעלת שטח 0, לא נשווה יחס
        return True # נניח שההתאמה היא רק לפי פיקסלים אם השטח 0
    
    # בודק אם היחס בין השטחים קרוב
    if abs(area1 - area2) / max(area1, area2) > tolerance_area_ratio:
        return False
    
    return True

# מקבל תמונה ומידע על אנשים בה, מזהה את הלבוש של אדם ספציפי ושל שאר האנשים באמצעות מודל למידת מכונה
# ואז קובע סטטוס לבוש כולל בהתבסס על השוואת הלבוש של האדם הממוקד מול שאר האנשים
#  ולבסוף מדפיס את התוצאה לתוכנה חיצונית
def get_person_clothing_prediction(person_crop_img):
    """
    מקבל קרופ של אדם, מבצע טרנספורמציה ומחזיר את הקלאס המנובא ביותר (label name).
    """
    if person_crop_img.size[0] == 0 or person_crop_img.size[1] == 0:
        return None # תמונה ריקה או לא תקינה

    img_tensor = transform_clothing(person_crop_img).unsqueeze(0).to(DEVICE) 
    with torch.no_grad():
        output = model_clothing(img_tensor) 
    
    _, predicted_class_id = torch.max(output, 1) # מוצא את הקלאס עם ה-logit הגבוה ביותר
    predicted_class_name = clothing_classes_map.get(predicted_class_id.item(), "Unknown")
    
    return predicted_class_name, predicted_class_id.item()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Classify clothing of a focused person vs. others.')
    parser.add_argument('full_image_path', type=str, help='Path to the full input image.')
    parser.add_argument('focused_person_bbox_str', type=str, help='Bounding box of the focused person (x1,y1,x2,y2).')
    parser.add_argument('all_persons_bboxes_str', type=str, help='All person bounding boxes (x1,y1,x2,y2;X1,Y1,X2,Y2;...).')
    args = parser.parse_args()

    full_img = cv2.imread(args.full_image_path)
    if full_img is None:
        print(f"CnnYolo.py: שגיאה: לא ניתן לטעון תמונה מלאה מ: {args.full_image_path}", file=sys.stderr)
        print("RESULTS_START")
        print("clothing:Error: Full image not found.")
        print("RESULTS_END")
        sys.exit(1)

    full_img_pil = Image.fromarray(cv2.cvtColor(full_img, cv2.COLOR_BGR2RGB))

    focused_bbox = parse_bbox_string(args.focused_person_bbox_str)
    all_bboxes = parse_all_bboxes_string(args.all_persons_bboxes_str)

    if focused_bbox is None or not all_bboxes:
        print("CnnYolo.py: שגיאה: Bounding box של אדם ממוקד או רשימת כל האנשים ריקים או לא תקינים.", file=sys.stderr)
        print("RESULTS_START")
        print("clothing:Error: Invalid bboxes.")
        print("RESULTS_END")
        sys.exit(1)

    focused_person_clothing_label = "Unknown"
    focused_person_clothing_id = -1
    other_persons_clothing_labels = []
    other_persons_clothing_ids = []

    # סיווג האדם הממוקד
    found_focused_in_all = False
    for bbox in all_bboxes:
        # השוואת ה-bbox של האדם הממוקד לתיבות התוחם של כל האנשים
        if is_bbox_within_tolerance(focused_bbox, bbox):
            x1, y1, x2, y2 = bbox
            focused_person_crop = full_img_pil.crop((x1, y1, x2, y2))
            focused_person_clothing_label, focused_person_clothing_id = get_person_clothing_prediction(focused_person_crop)
            if focused_person_clothing_label is not None:
                found_focused_in_all = True
            else:
                print("CnnYolo.py: אזהרה: לא ניתן לסווג את האדם הממוקד.", file=sys.stderr)
            # אם מצאנו את האדם הממוקד, אין צורך להמשיך בלולאה
            break
    
    if not found_focused_in_all:
        print("CnnYolo.py: שגיאה: האדם הממוקד לא נמצא ברשימת כל האנשים או שלא ניתן לסווגו.", file=sys.stderr)
        print("RESULTS_START")
        print("clothing:Error: Focused person not found or classified.")
        print("RESULTS_END")
        sys.exit(1)

    # סיווג שאר האנשים
    for bbox in all_bboxes:
        if not is_bbox_within_tolerance(focused_bbox, bbox): # אם זה לא האדם הממוקד
            x1, y1, x2, y2 = bbox
            other_person_crop = full_img_pil.crop((x1, y1, x2, y2))
            label, _id = get_person_clothing_prediction(other_person_crop)
            if label is not None:
                other_persons_clothing_labels.append(label)
                other_persons_clothing_ids.append(_id)

    final_clothing_status = "אחר" # ברירת מחדל

    # לוגיקה: אם האדם הממוקד לובש מעיל ושאר האנשים אינם לובשים מעיל
    # (האדם הממוקד הוא 'מעיל' וכל השאר הם לא 'מעיל')
    if focused_person_clothing_label == 'מעיל':
        all_others_not_coat = True
        if not other_persons_clothing_ids: # אין אחרים
            all_others_not_coat = False # התנאי "ושאר האנשים אינם" לא מתקיים
        else:
            for other_id in other_persons_clothing_ids:
                if clothing_classes_map.get(other_id) == 'מעיל': # אם מישהו אחר לובש מעיל
                    all_others_not_coat = False
                    break
        if all_others_not_coat:
            final_clothing_status = "מעיל"
    
    # לוגיקה: אם האדם הממוקד אינו לובש חולצה לבנה וכל האנשים האחרים כן לובשים חולצה לבנה
    # (האדם הממוקד הוא לא 'חולצה לבנה' וכל השאר הם 'חולצה לבנה')
    elif focused_person_clothing_label != 'חולצה לבנה': 
        all_others_white_shirt = True
        if not other_persons_clothing_ids: # אין אחרים
            all_others_white_shirt = False # התנאי "וכל האנשים האחרים כן" לא מתקיים
        else:
            for other_id in other_persons_clothing_ids:
                if clothing_classes_map.get(other_id) != 'חולצה לבנה': # אם מישהו אחר לא לובש חולצה לבנה
                    all_others_white_shirt = False
                    break
        if all_others_white_shirt:
            final_clothing_status = "חולצה לבנה"

    # לוגיקה: אם האדם הממוקד אינו לובש שרוולים קצרים (כלומר, לובש ארוך) וכל האנשים האחרים כן לובשים שרוולים קצרים
    # (האדם הממוקד הוא 'שרוולים ארוכים' וכל השאר הם 'שרוולים קצרים')
    elif focused_person_clothing_label == 'שרוולים ארוכים': # האדם הממוקד לובש ארוך
        all_others_short_sleeves = True
        if not other_persons_clothing_ids: # אין אחרים
            all_others_short_sleeves = False # התנאי "וכל האנשים האחרים כן" לא מתקיים
        else:
            for other_id in other_persons_clothing_ids:
                if clothing_classes_map.get(other_id) != 'שרוולים קצרים': # אם מישהו אחר לא לובש קצר
                    all_others_short_sleeves = False
                    break
        if all_others_short_sleeves:
            final_clothing_status = "שרוולים קצרים" # השם מתייחס לתנאי שהתקיים אצל האחרים
    
    # הדפסת התוצאה בפורמט קריא ל-C++
    print("RESULTS_START")
    print(f"clothing:{final_clothing_status}")
    print("RESULTS_END")
    
    sys.exit(0)