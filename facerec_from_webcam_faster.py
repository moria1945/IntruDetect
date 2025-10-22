# ייבוא ספריות נדרשות
import face_recognition
import cv2 
import numpy as np 
from collections import defaultdict # יצירת מילון שבו ערך ברירת מחדל נוצר אוטומטית למפתח שאינו קיים
import pickle # טעינה ושמירה של אוביקטים
from sklearn.neighbors import KDTree # חיפוש שכנים קרובים
import sys #קידוד לעברית
import os # קבצים
import datetime # זמן
import argparse # עברת ארגמנתים ל-C++

sys.stdout.reconfigure(encoding='utf-8')
sys.stderr.reconfigure(encoding='utf-8')


# הגדרות ופרמטרים גלובליים

# קובץ המכיל את קידודי הפנים של הפושעים
ENCODED_FACES_FILE = "encoded_faces.pkl" 

# מוסכמת זיהוי פושעים
CRIMINAL_NAME_INDICATORS = ["פושע", "מחבל", "Criminal", "Terrorist"] 

# פרמטרים מרכזיים לזיהוי פנים
FACE_RECOGNITION_MODEL = "hog" # מודל זיהוי פנים
NUM_JITTERS = 5 # מס' חזרות לדיוק בזיהוי הפנים
TOLERANCE = 0.6 # סף התאמה
MIN_VISITS_FOR_ALERT = 5 # מנימום מופעים להתראה

# נתיב לתקיה לשמירת תמונות מיוצאות
EXPORT_FACES_DIR = "exported_faces"
# אם התקיה קיימת צא 
os.makedirs(EXPORT_FACES_DIR, exist_ok=True) 


# משתנים גלובליים לניהול מצב

known_face_encodings = None # מערך קידודי פנים ידועים שנטענו מ-ENCODED_FACES_FILE.
known_face_names = None # מערך שמות הפנים הידועות, מקביל ל-known_face_encodings.
known_face_tree = None # אובייקט KDTree לביצוע חיפוש
criminal_names_set = set() # קבוצה של שמות פושעים מזוהים
unknown_face_encodings = {} # מילון לאחסון קידודי פנים של אנשים לא ידועים שזוהו (בדרך כלל לא בשימוש ישיר בממשק C++).
added_unknown_faces = {} # פנים לא ידועות שנוספו באופן זמני (בדרך כלל לא בשימוש ישיר בממשק C++).
unknown_person_counter = 0 # מונה ליצירת שמות זמניים עבור אנשים לא ידועים (לדוגמה, "Unknown_1").
face_counts = defaultdict(int) # מילון לספירת ביקורים של כל פנים שזוהתה. יתאפס בכל הפעלת סקריפט.
last_export_time = defaultdict(lambda: datetime.datetime.min) # עוקב אחר זמן הייצוא האחרון של תמונת פנים ספציפית, למניעת ייצוא יתר.
EXPORT_COOLDOWN_SECONDS = 5 # הגדרת "זמן צינון" (cooldown) בשניות: לא לייצא את אותה פנים שוב בתוך X שניות.


# פונקציות עזר
# פונקציות המבצעות משימות ספציפיות כדי לשמור על קוד מודולרי ונקי.

# טוענת את כל נתוני הפנים הידועים (קידודים ושמות) מקובץ ייעודי וגם מזהה מי מבין השמות האלה מייצג פושע
def _load_known_faces_and_criminals():
    """
    טוען קידודי פנים ידועים מקובץ ה-pickle ומזהה פושעים מתוכם.
    פעולה זו מתבצעת פעם אחת בעת עליית הסקריפט.
    """
    global known_face_encodings, known_face_names, known_face_tree, criminal_names_set

    # ודא שקובץ הקידוד קיים לפני הניסיון לטעון
    if not os.path.exists(ENCODED_FACES_FILE):
        print(f"Error: Encoding file {ENCODED_FACES_FILE} not found. Please run the encoding script first.", file=sys.stderr)
        sys.exit(1) # יציאה עם קוד שגיאה

    try:
        # טען נתונים מקובץ הקידוד המכיל קידודים ושמות
        with open(ENCODED_FACES_FILE, 'rb') as f:
            encoded_data = pickle.load(f)
            known_face_encodings = encoded_data.get("encodings", np.empty((0, 128))) # אחזור הקידודים, ברירת מחדל למערך ריק אם לא נמצא.
            known_face_names = encoded_data.get("names", np.array([])) # אחזור השמות, ברירת מחדל למערך ריק אם לא נמצא.

        # אם קיימים קידודים במאגר, בנה KDTree לחיפוש מהיר ויעיל.
        if known_face_encodings.shape[0] > 0:
            known_face_tree = KDTree(known_face_encodings)
        else:
            known_face_tree = None
            print("Warning: Encoding file is empty or contains no faces. Database is empty.", file=sys.stderr)

        # זיהוי פושעים: סרוק את שמות הפנים הידועות והוסף שמות המכילים אינדיקטורים לרשימת הפושעים.
        criminal_names_set = set()
        for name in known_face_names:
            for indicator in CRIMINAL_NAME_INDICATORS:
                if indicator.lower() in name.lower(): # השוואה לא תלוית רישיות
                    criminal_names_set.add(name)
                    break # עבר לשם הבא לאחר זיהוי אינדיקטור אחד

        print(f"{len(known_face_names)} known faces loaded from {ENCODED_FACES_FILE}.", file=sys.stderr)
        if criminal_names_set:
            print(f"Identified {len(criminal_names_set)} criminals in the database: {criminal_names_set}", file=sys.stderr)
        else:
            print("Warning: No criminals identified in the database (ensure CRIMINAL_NAME_INDICATORS is set correctly).", file=sys.stderr)

    except Exception as e:
        print(f"Error loading encodings: {e}", file=sys.stderr)
        sys.exit(1) # יציאה עם קוד שגיאה במקרה של תקלה בטעינה
#  מקבלת קידוד פנים משתמשת בעץ חיפוש מהיר  כדי למצוא את הפנים הקרובות ביותר במאגר הידועים ומחזירה את שמו של האדם התואם אם המרחק קטן מסף מסוים, אחרת היא מחזירה לא ידוע 
def find_match_with_tree(face_encoding):
    """
    מוצא התאמה עבור קידוד פנים נתון במאגר הפנים הידועות, תוך שימוש ב-KDTree לחיפוש אופטימלי.
    מחזירה את שם האדם המזוהה ואת מרחק ההתאמה.
    """
    # אם אין מאגר פנים או שהוא ריק, כל פנים תיחשב "Unknown".
    if known_face_tree is None or known_face_encodings.shape[0] == 0:
        return "Unknown", np.inf 

    # שאילתה לעץ KDTree למציאת השכן הקרוב ביותר לקידוד הפנים הנתון (k=1).
    distances, indices = known_face_tree.query([face_encoding], k=1)
    
    # אם המרחק שנמצא קטן או שווה לסף הסבילות (TOLERANCE), זוהתה התאמה.
    if distances[0][0] <= TOLERANCE:
        return known_face_names[indices[0][0]], distances[0][0] # החזר את שם האדם המוכר והמרחק.
    return "Unknown", distances[0][0] # אחרת, החזר "Unknown" ואת המרחק.
# מוסיפה קידוד פנים חדש עם שם למאגר הפנים הידועות בזיכרון של התוכנית, ובמידת הצורך בונה מחדש את עץ החיפוש המהיר כדי לכלול את הפנים החדשות. 
def update_known_faces(new_encoding, new_name):
    """
    מוסיף קידוד פנים חדש למאגר הפנים הידועות בזיכרון בזמן ריצה.
    הערה: שינויים אלה אינם נשמרים באופן קבוע עקב אופי הפעלת הסקריפט ע"י C++ (איפוס מצב בכל הפעלה).
    """
    global known_face_encodings, known_face_names, known_face_tree

    # אם השם כבר קיים במאגר, אין צורך להוסיף שוב.
    if new_name in known_face_names:
        return

    # הוספת הקידוד והשם למערכים הגלובליים.
    if known_face_encodings.shape[0] == 0: # אם המאגר ריק, אתחל אותו עם הפנים החדשות.
        known_face_encodings = np.array([new_encoding])
        known_face_names = np.array([new_name])
    else: # אם המאגר קיים, הוסף אליו את הקידוד והשם.
        known_face_encodings = np.vstack([known_face_encodings, new_encoding])
        known_face_names = np.append(known_face_names, new_name)
    known_face_tree = KDTree(known_face_encodings) # בנה מחדש את ה-KDTree כדי שיכלול את הפנים החדשות.

# שומרת תמונת פנים שזוהתה מפריים נתון לתיקייה ייעודית, 
# תוך מניעת שמירה כפולה של אותה פנים בפרקי זמן קצרים כדי למנוע הצפת שטח אחסון. 
# היא חותכת את אזור הפנים מהתמונה המקורית,
# מוודאת שהקואורדינטות תקינות, ויוצרת שם קובץ ייחודי הכולל את שם האדם, סיבת השמירה וחותמת זמן.
def _export_face_image(frame, face_location_original_coords, name, reason):
    """
    שומר את תמונת הפנים שזוהתה לתיקיית הייצוא, תוך שמירה על "זמן צינון" למניעת כפילויות.
    """
    current_time = datetime.datetime.now()
    # בדוק אם עבר מספיק זמן מאז הייצוא האחרון של פנים אלה (לפי EXPORT_COOLDOWN_SECONDS).
    if (current_time - last_export_time[name]).total_seconds() < EXPORT_COOLDOWN_SECONDS:
        # print(f"Skipping export for {name} due to cooldown.", file=sys.stderr) # ניתן להפעיל להדפסת הודעות דיבוג
        return

    # חיתוך הפנים מהפריים המקורי לפי הקואורדינטות שזוהו.
    top, right, bottom, left = face_location_original_coords
    
    # ודא שהקואורדינטות בתוך גבולות התמונה כדי למנוע שגיאות.
    h, w, _ = frame.shape
    top = max(0, top)
    right = min(w, right)
    bottom = min(h, bottom)
    left = max(0, left)

    face_image = frame[top:bottom, left:right]

    # בדיקה אם תמונת הפנים החתוכה ריקה או לא תקינה (למשל, קואורדינטות לא תקינות).
    if face_image.shape[0] == 0 or face_image.shape[1] == 0:
        print(f"Warning: Empty or invalid face image for export for {name}.", file=sys.stderr)
        return

    # יצירת שם קובץ ייחודי הכולל חותמת זמן ואת סיבת הייצוא.
    timestamp = current_time.strftime("%Y%m%d_%H%M%S")
    # ניקוי השם מתווים שאינם חוקיים לשם קובץ.
    safe_name = "".join(c for c in name if c.isalnum() or c in (' ', '_')).rstrip()
    if not safe_name: safe_name = "unknown" # אם השם ריק אחרי הניקוי, השתמש ב"unknown".

    filename = os.path.join(EXPORT_FACES_DIR, f"{safe_name}_{reason}_{timestamp}.jpg")
    try:
        cv2.imwrite(filename, face_image) # שמור את התמונה לתיקייה.
        print(f"Face image saved: {filename} (Reason: {reason})", file=sys.stderr) # הדפסה ל-stderr לרישום לוגים.
        last_export_time[name] = current_time # עדכן את זמן הייצוא האחרון עבור פנים אלו.
    except Exception as e:
        print(f"Error saving image {filename}: {e}", file=sys.stderr)


# כלי עזר: יצירה/עדכון של מאגר קידודים (פונקציית הכנה)
# פונקציה זו נועדה להרצה חד פעמית (או בעת עדכון המאגר) ולא כחלק מלולאת הזיהוי בזמן אמת.

# סורקת תיקייה המכילה תמונות של אנשים (כששם הקובץ הוא שם האדם),
# מזהה את הפנים בכל תמונה, מקודדת אותן, ושומרת את כל הקידודים והשמות לקובץ 
def create_or_update_encoded_faces_file_from_folder(criminals_faces_dir, output_file=ENCODED_FACES_FILE):
    """
    יוצר או מעדכן את קובץ הקידוד `output_file` מתמונות הנמצאות בתיקייה `criminals_faces_dir`.
    כל תמונה בתיקייה צריכה להיקרא על שם האדם (ללא סיומת קובץ).
    לדוגמה: "אבי.jpg", "פושע_דוד.png".
    """
    print(f"\n--- Starting face encoding process from folder: {criminals_faces_dir} ---", file=sys.stderr)
    known_face_encodings_list = []
    known_face_names_list = []

    # ודא שתיקיית התמונות קיימת.
    if not os.path.exists(criminals_faces_dir):
        print(f"Error: Image folder '{criminals_faces_dir}' not found. Please create it and place images inside.", file=sys.stderr)
        return

    # עבור על כל הקבצים בתיקייה.
    for filename in os.listdir(criminals_faces_dir):
        # בדוק שזהו קובץ תמונה רלוונטי.
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.gif')):
            filepath = os.path.join(criminals_faces_dir, filename) # נתיב מלא לקובץ התמונה.
            image_name = os.path.splitext(filename)[0] # חלץ את שם הקובץ ללא סיומת (זה יהיה שם האדם).
            
            print(f"Encoding '{image_name}' from '{filepath}'...", file=sys.stderr)
            try:
                image = face_recognition.load_image_file(filepath) # טען את התמונה.
                # מצא את קידודי הפנים בתמונה (num_jitters משפיע על דיוק הקידוד).
                face_encodings_list = face_recognition.face_encodings(image, num_jitters=NUM_JITTERS) 

                if face_encodings_list:
                    face_encoding = face_encodings_list[0] # קח את הקידוד הראשון שזוהה (הנחה: תמונה אחת מכילה פנים אחת).
                    known_face_encodings_list.append(face_encoding)
                    known_face_names_list.append(image_name)
                else:
                    print(f"Warning: No face detected in image '{filename}'. Not encoding.", file=sys.stderr)

            except Exception as e:
                print(f"Error processing image '{filename}': {e}", file=sys.stderr)

    # המר את רשימות הקידודים והשמות למערכי NumPy.
    final_encodings = np.array(known_face_encodings_list)
    final_names = np.array(known_face_names_list)

    if final_encodings.shape[0] == 0:
        print("No faces were encoded. Encoding file not created or updated.", file=sys.stderr)
        return

    try:
        encoded_data = {"encodings": final_encodings, "names": final_names}
        with open(output_file, 'wb') as f:
            pickle.dump(encoded_data, f) # שמור את הקידודים והשמות לקובץ ה-pickle.
        print(f"SUCCESS: {len(final_names)} faces encoded and saved to '{output_file}' successfully.", file=sys.stderr)
    except Exception as e:
        print(f"Error saving encoding file to '{output_file}': {e}", file=sys.stderr)


# לולאת זיהוי פנים ראשית (נקודת כניסה לסקריפט)
# קוד זה מופעל כאשר הסקריפט מורץ ישירות (כמו על ידי תוכנית C++).


if __name__ == '__main__':
    # הגדרת ארגומנטים שניתן להעביר לסקריפט משורת הפקודה.
    parser = argparse.ArgumentParser(description="Face Recognition Processor")
    parser.add_argument("--image", required=False, help="Path to the image file to process.")
    parser.add_argument("--bbox", required=False, help="Bounding box of the focused person (x1,y1,x2,y2).")
    # ניתן להוסיף ארגומנטים נוספים אם C++ יעביר אותם בעתיד.

    args = parser.parse_args() # ניתוח הארגומנטים שהתקבלו.

    # טען את הפנים הידועות והפושעים ממאגר הנתונים (פעולה חד פעמית בהפעלת הסקריפט).
    _load_known_faces_and_criminals()

    # אתחול משתנים עבור תוצאות הפלט שישלחו ל-C++.
    person_name_output = "Unknown"
    is_criminal_output = "false"
    visits_exceeded_threshold_output = "false"
    total_visits_output = "0"
    
    # אם התוכנית הופעלה עם ארגומנט תמונה, בצע עיבוד של תמונה בודדת.
    if args.image:
        frame_path = args.image
        try:
            frame = cv2.imread(frame_path) # טען את התמונה.
            if frame is None:
                print(f"Error: Could not load image from {frame_path}", file=sys.stderr)
                sys.exit(1) # יציאה אם התמונה לא קיימת או לא ניתנת לטעינה.

            face_locations_to_process = []
            # אם סופק bbox (תיבת תוחמת) כארגומנט, השתמש בו למיקוד.
            if args.bbox and args.bbox != "None": 
                try:
                    x1, y1, x2, y2 = map(int, args.bbox.split(','))
                    # המרה מפורמט x1,y1,x2,y2 לפורמט של face_recognition (top, right, bottom, left).
                    face_locations_to_process = [(y1, x2, y2, x1)] 

                except ValueError:
                    print(f"Warning: Invalid bbox format: {args.bbox}. Attempting to detect faces in the full frame.", file=sys.stderr)
                    # לא תקין, חפש פנים בכל הפריים
                    small_frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5) 
                    rgb_small_frame = np.ascontiguousarray(small_frame[:, :, ::-1]) # המר לפורמט RGB הנדרש על ידי face_recognition.
                    face_locations_to_process = face_recognition.face_locations(rgb_small_frame, model=FACE_RECOGNITION_MODEL)
                    # המר בחזרה לקואורדינטות המקוריות (הכפלה ב-2 בגלל ההקטנה).
                    face_locations_to_process = [(top * 2, right * 2, bottom * 2, left * 2) for top, right, bottom, left in face_locations_to_process]
            else: # אם לא סופק bbox, חפש פנים בפריים המלא (עם הקטנה).
                small_frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
                rgb_small_frame = np.ascontiguousarray(small_frame[:, :, ::-1])
                face_locations_to_process = face_recognition.face_locations(rgb_small_frame, model=FACE_RECOGNITION_MODEL)
                face_locations_to_process = [(top * 2, right * 2, bottom * 2, left * 2) for top, right, bottom, left in face_locations_to_process]

            if face_locations_to_process:
                # קח רק את הפנים הראשונות שזוהו באזור הממוקד (או בפריים כולו).
                top_orig, right_orig, bottom_orig, left_orig = face_locations_to_process[0]
                face_crop = frame[top_orig:bottom_orig, left_orig:right_orig] # חתוך את אזור הפנים מהפריים המקורי.
                
                # ודא שהחיתוך תקף (לא ריק).
                if face_crop.shape[0] > 0 and face_crop.shape[1] > 0:
                    rgb_face_crop = np.ascontiguousarray(face_crop[:, :, ::-1]) # המר את חיתוך הפנים ל-RGB.
                    # מצא קידודים בתוך החיתוך (בהנחה של פנים אחת ב-bbox).
                    face_encodings_in_crop = face_recognition.face_encodings(rgb_face_crop, [(0, face_crop.shape[1], face_crop.shape[0], 0)], num_jitters=NUM_JITTERS) 
                    
                    if face_encodings_in_crop:
                        face_encoding = face_encodings_in_crop[0]
                        name, distance = find_match_with_tree(face_encoding) # מצא התאמה במאגר הפנים הידועות.

                        person_name_output = name # עדכן את שם האדם לפלט.
                        if name != "Unknown" and name in criminal_names_set:
                            is_criminal_output = "true" # סמן כפושע אם זוהה ומופיע ברשימת הפושעים.
                            _export_face_image(frame, (top_orig, right_orig, bottom_orig, left_orig), name, "criminal_single_frame") # ייצא תמונה של פושע.
                        
                        # מעקב ביקורים: ספירת ביקורים לאדם זה (יש לזכור שמונה זה מתאפס בכל הפעלה של הסקריפט).
                        face_counts[name] += 1
                        if face_counts[name] > MIN_VISITS_FOR_ALERT:
                            visits_exceeded_threshold_output = "true"
                            if not is_criminal_output == "true": # אם לא פושע, ורק מבקר תכוף, ייצא תמונה.
                                _export_face_image(frame, (top_orig, right_orig, bottom_orig, left_orig), name, "frequent_visitor_single_frame")
                        
                        total_visits_output = str(face_counts[name]) # ספירת ביקורים כוללת לפלט.
                    else:
                        print(f"Warning: No face detected in the focused crop from {frame_path}.", file=sys.stderr)
                else:
                    print(f"Warning: Empty or invalid crop from {frame_path}.", file=sys.stderr)
            else:
                print(f"Warning: No face detected in the provided frame/bbox from {frame_path}.", file=sys.stderr)

        except Exception as e:
            print(f"Error processing image {frame_path}: {e}", file=sys.stderr)

    # --- הדפסת תוצאות ל-stdout בפורמט ש-C++ מצפה לו ---
    # C++ יקרא שורות אלה וינתח אותן כדי לקבל את המידע.
    print("RESULTS_START")
    print(f"PersonName:{person_name_output}")
    print(f"IsCriminal:{is_criminal_output}")
    print(f"VisitsExceededThreshold:{visits_exceeded_threshold_output}")
    print(f"TotalVisits:{total_visits_output}")
    print("RESULTS_END")

    # אין צורך בלולאת וידאו בתוך קובץ זה כאשר מורץ על ידי C++, כיוון ש-C++ הוא המנהל הראשי של זרימת הווידאו.
    # שחרור משאבים (לדוגמה, video_capture.release() ו-cv2.destroyAllWindows()) אינו נדרש כאן כי לא נפתחו משאבי וידאו בתוך סקריפט זה.