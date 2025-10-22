import cv2 # ספריית OpenCV לעיבוד תמונות ווידאו
import numpy as np # ספריית NumPy לפעולות על מערכים (Arrays)
import os # ספרייה לפעולות מערכת הפעלה (נתיבים, תיקיות)
import subprocess # ספרייה להפעלת תהליכי משנה (Subprocesses)
import re # ספרייה לביטויים רגולריים (לא בשימוש ישיר בקוד זה כרגע אך שימושית)
import torch # ספריית PyTorch לטעינת מודלים וביצוע היסקים (Inference)
import random # ספרייה לפעולות אקראיות (לא בשימוש ישיר בקוד זה כרגע)
import shutil # ספרייה לפעולות קבצים ברמה גבוהה (כמו מחיקת תיקיות)
import sys # ספרייה לגישה למערכת ההפעלה, כולל קלט/פלט סטנדרטי
import datetime # הוסף ייבוא עבור חותמת זמן

# הגדרת קידוד פלט סטנדרטי ל-UTF-8.
# (Set output encoding for Python's own output to UTF-8.)
sys.stdout.reconfigure(encoding='utf-8')
sys.stderr.reconfigure(encoding='utf-8')

# --- הגדרות גלובליות ---
# (--- Global Configurations ---)

# טעינת מודל YOLOv5s.
# מודל זה נטען פעם אחת בלבד כשהסקריפט מתחיל, כדי למנוע טעינה חוזרת בכל פריים.
# (Loading YOLOv5s model.
# This model is loaded only once when the script starts, to prevent reloading it every frame.)
try:
    model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
    model.eval() # הגדרת המודל למצב הערכה (Evaluation mode)
except Exception as e:
    print(f"Error loading YOLOv5 model: {e}", file=sys.stderr)
    sys.exit(1) # יציאה אם טעינת המודל נכשלה

# נתיב לקובץ הווידאו
# (Path to the video file)
VIDEO_PATH = r"C:\\Users\\moiam\\Documents\\Project\\aldoritmem\\People Walking Past the Camera - Free Stock Footage For Commercial Projects.mp4"

# נתיב לקובץ ההרצה של C++. ודא שקובץ זה קיים וקומפל.
# (Path to your C++ executable. Ensure this file exists and is compiled.)

CPP_EXECUTABLE_PATH = r"C:\Users\moiam\Documents\Project\aldoritmem\cpp_m.exe"

#CPP_EXECUTABLE_PATH = r"C:\\Users\\moiam\\Documents\\Project\\aldoritmem\\cpp_m.cpp" 

# תיקייה זמנית לקבצים המועברים ל-C++
# (Temporary directory for files passed to C++)
TEMP_DIR = "temp_data_for_cpp" 

# מזהה אנשים בתוך תמונת וידאו (פריים) באמצעות מודל YOLOv5 ומחזירה את תיבות התוחמת של האנשים שזוהו.
def detect_persons(frame):
    """
    Detects persons in a frame using YOLOv5.
    מזהה אנשים בפריים באמצעות YOLOv5.
    """
    results = model(frame) # הפעלת המודל על הפריים
    # סינון עבור אובייקטים מסוג class 0 (person).
    # xyxy[0] מכיל את כל הזיהויים, [:, 5] הוא אינדקס הקלאס, == 0 מסנן לאנשים.
    # (Filter for class 0 (person).
    # xyxy[0] contains all detections, [:, 5] is the class index, == 0 filters for persons.)
    persons = results.xyxy[0][results.xyxy[0][:, 5] == 0]
    return persons

#מקבלת קואורדינטות של תיבת תוחמת ומחשבת את נקודת המרכז שלה.
def calculate_center(bbox):
    """
    Calculates the center point of a bounding box.
    מחשב את נקודת המרכז של תיבת תוחמת (Bounding Box).
    """
    x1, y1, x2, y2 = bbox
    return ((x1 + x2) // 2, (y1 + y2) // 2)
#  מחשבת את המרחק האוקלידי (הקו הישר) בין שתי נקודות נתונות
def distance(center1, center2):
    """
    Calculates Euclidean distance between two points.
    מחשב מרחק אוקלידי בין שתי נקודות.
    """
    return np.sqrt((center1[0] - center2[0])**2 + (center1[1] - center2[1])**2)

# מקבלת פריים, קואורדינטות של תיבת תוחמת, טקסט להצגה וצבע, ומציירת את המלבן והטקסט על הפריים,
#  תוך התאמת מיקום הטקסט למניעת יציאתו מגבולות התמונה.
def draw_person_bbox(frame, bbox, display_text, color, thickness=2):
    """
    Draws a bounding box and text on the frame.
    מצייר תיבת תוחמת וטקסט על הפריים.
    """
    x1, y1, x2, y2 = map(int, bbox)
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness) # ציור המלבן

    # חישוב מיקום הטקסט כדי שלא יצא מגבולות התמונה
    # (Calculate text position to prevent it from going out of image bounds)
    text_y_pos = y2 - 6
    if text_y_pos < y1 + 20: # אם אין מספיק מקום למטה, צייר למעלה
        text_y_pos = y1 + 25
        cv2.rectangle(frame, (x1, y1), (x2, y1 + 35), color, cv2.FILLED) # מלבן מילוי לרקע הטקסט
    else:
        cv2.rectangle(frame, (x1, y2 - 35), (x2, y2), color, cv2.FILLED) # מלבן מילוי לרקע הטקסט

    font = cv2.FONT_HERSHEY_DUPLEX
    cv2.putText(frame, display_text, (x1 + 6, text_y_pos), font, 0.8, (255, 255, 255), 1) # ציור הטקסט
    return frame
# מכינה נתונים ותמונות (פריים מלא וחיתוך של האדם הממוקד) לשליחה לתוכנית חיצונית ב-C++, 
# ושומרת אותם בתיקייה זמנית. היא מוודאת שתמונות החיתוך תקינות וגדולות מספיק,
#  ומעבירה את הקואורדינטות כמחרוזות.
def prepare_data_for_cpp(frame, focused_person_bbox_arr, previous_focused_bbox_arr, frame_count, temp_dir):
    """
    Prepares image files and data for the C++ process.
    Saves the full frame and the focused person crop, with integrity checks.
    מכין קבצי תמונה ונתונים עבור תהליך ה-C++.
    שומר את הפריים המלא ואת חיתוך הפנים הממוקדות, עם בדיקות תקינות.
    """
    prepared_data = {}

    # שמור את הפריים המלא (תמיד נסה לשמור אותו)
    # (Save the full frame (always try to save it))
    full_frame_path = os.path.join(temp_dir, f"full_frame_{frame_count}.jpg") 
    try:
        cv2.imwrite(full_frame_path, frame)
        prepared_data['full_frame_path'] = full_frame_path
        print(f"[Python Debug] Saved full frame to: {full_frame_path}", file=sys.stderr)
    except Exception as e:
        print(f"[Python Error] Could not save full frame {full_frame_path}: {e}", file=sys.stderr)
        prepared_data['full_frame_path'] = "none" # סמן כלא זמין

    # שמור את החיתוך כ-PNG כדי לנסות לפתור בעיות קריאה ב-C++
    # (Save the crop as PNG to try and resolve C++ read issues)
    focused_person_crop_path = os.path.join(temp_dir, f"focused_person_crop_{frame_count}.png")

    # הגדר סף מינימלי למידות החיתוך.
    # (Set minimum threshold for crop dimensions.)
    min_dim = 20

    if focused_person_bbox_arr is not None:
        x1, y1, x2, y2 = map(int, focused_person_bbox_arr)

        # ודא שקואורדינטות החיתוך בתוך גבולות הפריים
        # (Ensure crop coordinates are within frame boundaries)
        h_frame, w_frame, _ = frame.shape
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(w_frame, x2)
        y2 = min(h_frame, y2)

        # בדוק מידות מינימליות של החיתוך לפני שמירה
        # (Check minimum dimensions of the crop before saving)
        current_width = x2 - x1
        current_height = y2 - y1

        if current_width >= min_dim and current_height >= min_dim:
            # אם תיבת התוחמת תקפה וגדולה מספיק, חתוך ושמור
            # (If the bounding box is valid and large enough, crop and save)
            try:
                focused_person_crop = frame[y1:y2, x1:x2].copy()
                cv2.imwrite(focused_person_crop_path, focused_person_crop)
                prepared_data['focused_person_crop_path'] = focused_person_crop_path
                prepared_data['focused_person_bbox_str'] = f"{x1},{y1},{x2},{y2}" # העבר כמחרוזת
                print(f"[Python Debug] Saved focused person crop to: {focused_person_crop_path} with dimensions: {focused_person_crop.shape}", file=sys.stderr)
            except Exception as e:
                print(f"[Python Error] Could not save focused person crop {focused_person_crop_path}: {e}", file=sys.stderr)
                prepared_data['focused_person_crop_path'] = "none"
                prepared_data['focused_person_bbox_str'] = "none"
        else:
            # אם תיבת התוחמת לא תקפה או קטנה מדי
            # (If the bounding box is invalid or too small)
            print(f"[Python Debug] Skipping focused person crop save for frame {frame_count}: Bbox too small or invalid ({x1},{y1},{x2},{y2}). Expected min dimension {min_dim}.", file=sys.stderr)
            prepared_data['focused_person_crop_path'] = "none" # חשוב: שלח "none" אם החיתוך לא תקף
            prepared_data['focused_person_bbox_str'] = "none" # וה-bbox אינו רלוונטי
    else:
        # אם אין אדם ממוקד בכלל
        # (If no focused person at all)
        print(f"[Python Debug] Skipping focused person crop save for frame {frame_count}: No focused person detected.", file=sys.stderr)
        prepared_data['focused_person_crop_path'] = "none"
        prepared_data['focused_person_bbox_str'] = "none"

    # תיבת תוחמת קודמת (אם קיימת) עבור ניתוח תנועה
    # (Previous bounding box (if exists) for motion analysis)
    if previous_focused_bbox_arr is not None:
        prepared_data['previous_focused_bbox_str'] = ",".join(map(str, map(int, previous_focused_bbox_arr)))
    else:
        prepared_data['previous_focused_bbox_str'] = "none" # עדיף "none" על פני "None" עבור C++

    return prepared_data

# מוחקת את התיקייה הזמנית ששימשה לאחסון קבצים, כולל כל התוכן שלה, 
# כדי לשמור על סדר ולפנות מקום. היא כוללת טיפול בשגיאות למקרה שהתיקייה לא ניתנת למחיקה
def cleanup_temp_dir(temp_dir):
    """
    Deletes the temporary files directory.
    מוחק את תיקיית הקבצים הזמנית.
    """
    if os.path.exists(temp_dir):
        try:
            shutil.rmtree(temp_dir) # מחיקת התיקייה ותוכנה
            print(f"Temporary directory '{temp_dir}' cleaned up.", file=sys.stderr)
        except OSError as e:
            print(f"Warning: Could not clean up temporary directory '{temp_dir}': {e}. Files might be locked.", file=sys.stderr)

#  לולאת עיבוד ראשית 
# היא הפונקציה הראשית של התוכנית. היא פותחת את קובץ הווידאו, 
# עוברת פריים אחר פריים, מזהה ועוקבת אחרי אנשים  מכינה ושולחת נתונים לתוכנית חיצונית לצורך ניתוח נוסף 
#  מקבלת בחזרה את התוצאות, מציגה אותן על גבי המסך, ולבסוף מנקה את המשאבים.

def main():
    cap = cv2.VideoCapture(VIDEO_PATH) # פתיחת קובץ הווידאו
    if not cap.isOpened():
        print(f"Error: Could not open video file at path {VIDEO_PATH}", file=sys.stderr)
        sys.exit(1) # יציאה אם הווידאו לא נפתח

    frame_skip = 10 # עבד כל (frame_skip + 1) פריים
    frame_count = 0
    focused_person_bbox = None # תיבת תוחמת של האדם הממוקד הנוכחי (מערך numpy)
    previous_focused_bbox = None # תיבת תוחמת קודמת למעקב תנועה
    min_detection_dim = 20 # מימד מינימלי עבור אדם להיחשב 'תקף' לזיהוי ומעקב

    # נקה וצור מחדש תיקייה זמנית בהתחלה
    # (Clean and recreate temporary directory at the start)
    cleanup_temp_dir(TEMP_DIR)
    os.makedirs(TEMP_DIR, exist_ok=True) # צור את התיקייה

    while True:
        ret, frame = cap.read() # קרא פריים מהווידאו
        if not ret:
            break # סוף וידאו

        # הפעל זיהוי YOLO ועדכן את האדם הממוקד רק כל 'frame_skip' פריימים
        # (Run YOLO detection and update focused person only every 'frame_skip' frames)
        if frame_count % (frame_skip + 1) == 0:
            persons_in_current_frame = detect_persons(frame) # זיהוי אנשים בפריים הנוכחי

            valid_persons = []
            if persons_in_current_frame is not None:
                for person in persons_in_current_frame:
                    x1, y1, x2, y2 = map(int, person[:4].cpu().numpy()) # חילוץ קואורדינטות ה-bbox
                    # ודא שהקואורדינטות בתוך גבולות התמונה
                    # (Ensure coordinates are within image boundaries)
                    h_frame, w_frame, _ = frame.shape
                    x1 = max(0, x1)
                    y1 = max(0, y1)
                    x2 = min(w_frame, x2)
                    y2 = min(h_frame, y2)

                    current_width = x2 - x1
                    current_height = y2 - y1

                    # הוסף רק אנשים עם מימדים מספיקים לרשימת ה-valid_persons
                    # (Add only persons with sufficient dimensions to the valid_persons list)
                    if current_width >= min_detection_dim and current_height >= min_detection_dim:
                        valid_persons.append(person)

            if len(valid_persons) > 0:
                if focused_person_bbox is None:
                    # מיקוד התחלתי: בחר את האדם עם תיבת התוחמת הגדולה ביותר (שטח) מבין האנשים התקפים
                    # (Initial focus: select the person with the largest bounding box (area) among valid persons)
                    max_area = 0
                    selected_person = None
                    for person in valid_persons:
                        x1, y1, x2, y2 = person[:4].cpu().numpy()
                        area = (x2 - x1) * (y2 - y1)
                        if area > max_area:
                            max_area = area
                            selected_person = person

                    if selected_person is not None:
                        focused_person_bbox = selected_person[:4].cpu().numpy()
                        print(f"[Python Debug] Initial focus on largest person with bbox: {focused_person_bbox} at frame {frame_count}", file=sys.stderr)
                    else:
                        focused_person_bbox = None
                        print(f"[Python Debug] No valid person found for initial focus at frame {frame_count}.", file=sys.stderr)
                else:
                    # מעקב: מצא את האדם הקרוב ביותר למרכז האדם הממוקד הקודם, מבין האנשים התקפים
                    # (Tracking: find the person closest to the center of the previous focused person, among valid persons)
                    focused_person_center_prev = calculate_center(focused_person_bbox)
                    min_distance = float('inf')
                    closest_person = None

                    for person in valid_persons:
                        current_bbox = person[:4].cpu().numpy()
                        current_center = calculate_center(current_bbox)
                        dist = distance(focused_person_center_prev, current_center)
                        if dist < min_distance:
                            min_distance = dist
                            closest_person = person

                    if closest_person is not None:
                        focused_person_bbox = closest_person[:4].cpu().numpy()
                        print(f"[Python Debug] Tracking focused person to new bbox: {focused_person_bbox} at frame {frame_count}", file=sys.stderr)
                    else:
                        # אם לא נמצא אדם תקף קרוב, אפס את המיקוד
                        # (If no nearby valid person is found, reset focus)
                        focused_person_bbox = None
                        print(f"Warning: Focused person lost or all nearby persons are too small in frame {frame_count}. Resetting focus.", file=sys.stderr)
            else:
                # לא זוהו אנשים תקפים בפריים, אפס את המיקוד
                # (No valid persons detected in the frame, reset focus)
                focused_person_bbox = None
                print(f"No valid persons detected in frame {frame_count}. Resetting focus.", file=sys.stderr)

        frame_to_display = frame.copy() # צור עותק להצגה ויזואלית

        cpp_analysis_results = {} # מילון לאחסון תוצאות מ-C++

        # הכן נתונים (שמור פריימים) עבור C++ תמיד, אך C++ יקבל 'none' אם אין חיתוך תקף
        # (Prepare data (save frames) for C++ always, but C++ will receive 'none' if no valid crop)
        prepared_data = prepare_data_for_cpp(frame, focused_person_bbox, previous_focused_bbox, frame_count, TEMP_DIR)

        # המשך עם תקשורת C++ רק אם נתיב הפריים המלא זמין
        # (Continue with C++ communication only if full frame path is available)
        if prepared_data['full_frame_path'] != "none":
            # בנה את הפקודה עבור קובץ ההרצה של C++
            # (Build the command for the C++ executable)
            cpp_command = [
                CPP_EXECUTABLE_PATH,
                prepared_data['full_frame_path'],      # ארגומנט 1: נתיב לפריים המלא
                prepared_data['focused_person_crop_path'], # ארגומנט 2: נתיב לחיתוך האדם הממוקד (יכול להיות "none")
                prepared_data['focused_person_bbox_str'],  # ארגומנט 3: Bbox של האדם הממוקד (יכול להיות "none")
                prepared_data['previous_focused_bbox_str'] # ארגומנט 4: Bbox קודם (יכול להיות "none")
            ]
            
            try:
                # הדפסת הפקודה שרצה לדיבוג
                print(f"[Python Debug] Running C++ for frame {frame_count} with command: {cpp_command}", file=sys.stderr)

                # הפעל את קובץ ההרצה של C++ כתהליך משנה (subprocess)
                # השתמש ב-subprocess.run במקום Popen לניהול פלט קל יותר
                process = subprocess.run(
                    cpp_command,
                    capture_output=True, # ללכוד גם stdout וגם stderr
                    text=True,           # לקבל פלט כטקסט (במקום בתים)
                    encoding='utf-8',    # לקודד כ-UTF-8
                    errors='replace',    # טיפול בשגיאות קידוד (אם תווים לא חוקיים בפלט)
                    check=False,         # אל תזרוק חריגה עבור קוד יציאה שאינו אפס מיד, נטפל בזה ידנית
                    timeout=60           # מגביל זמן ל-60 שניות, למניעת תקיעות
                )
                
                # הדפס את stderr של ה-C++ ישירות לקונסול (sys.stderr)
                # זה יראה לך את הודעות הדיבוג של C++
                if process.stderr:
                    print(f"[CPP STDERR - Frame {frame_count}]:\n{process.stderr}", file=sys.stderr)
                
                # בדוק את קוד היציאה של ה-C++
                if process.returncode != 0:
                    print(f"[Python Error] C++ process for frame {frame_count} returned non-zero exit code {process.returncode}.", file=sys.stderr)
                    # אין צורך לצאת, פשוט תמשיך עם ברירות מחדל
                    # (No need to exit, just continue with defaults)

                # נתח את התוצאות מ-stdout של C++
                # (Parse results from C++ stdout)
                if process.stdout:
                    start_index = process.stdout.find("RESULTS_START")
                    end_index = process.stdout.find("RESULTS_END")
                    if start_index != -1 and end_index != -1:
                        output_block = process.stdout[start_index + len("RESULTS_START"):end_index]
                        for line in output_block.strip().split('\n'):
                            if ":" in line:
                                key, value = line.split(":", 1)
                                cpp_analysis_results[key.strip()] = value.strip()
                    print(f"Frame {frame_count} C++ consolidated results: {cpp_analysis_results}", file=sys.stderr)
                else:
                    print(f"[Python Warning] C++ process for frame {frame_count} produced no stdout.", file=sys.stderr)

            except subprocess.TimeoutExpired as e:
                # subprocess.run עם timeoutExpired תופס את הפלט החלקי ב-e.stdout וב-e.stderr
                print(f"[Python Error] C++ process for frame {frame_count} timed out after {e.timeout} seconds.", file=sys.stderr)
                if e.stdout:
                    print(f"[Python Error] C++ stdout (partial):\n{e.stdout}", file=sys.stderr)
                if e.stderr:
                    print(f"[Python Error] C++ stderr (partial):\n{e.stderr}", file=sys.stderr)
            except FileNotFoundError:
                print(f"[Python Error] Error: C++ executable not found at path '{CPP_EXECUTABLE_PATH}'. Please compile it and ensure the path is correct.", file=sys.stderr)
                sys.exit(1) # יציאה אם קובץ ההרצה לא נמצא, זה קריטי
            except Exception as e:
                print(f"[Python Error] An unexpected error occurred while running C++ process for frame {frame_count}: {e}", file=sys.stderr)
        else:
            print(f"Warning: Full frame path is unavailable for C++ in frame {frame_count}. Skipping C++ call.", file=sys.stderr)

        # --- הצגת תוצאות ---
        # (--- Display Results ---)
        # קבל תוצאות מ-C++ או ערכי ברירת מחדל אם קריאת C++ נכשלה/דלגה
        # (Get results from C++ or default values if C++ call failed/skipped)
        person_name = cpp_analysis_results.get('PersonName', 'Unknown')
        is_criminal = cpp_analysis_results.get('IsCriminal', 'false').lower() == 'true'
        visits_exceeded_threshold = cpp_analysis_results.get('VisitsExceededThreshold', 'false').lower() == 'true'
        total_visits = cpp_analysis_results.get('TotalVisits', 'N/A')

        display_text = person_name
        bbox_color = (0, 255, 0) # ירוק כברירת מחדל

        if is_criminal:
            bbox_color = (0, 0, 255) # אדום לפושע
            display_text = f"{person_name} (Criminal)"
        elif visits_exceeded_threshold:
            bbox_color = (255, 0, 0) # כחול למבקר תכוף (אם אינו פושע)
            display_text = f"{person_name} ({total_visits} visits)"
        elif person_name.startswith('P') or person_name == 'Unknown': # 'Unknown' או P# מ-C++
            bbox_color = (255, 165, 0) # כתום ללא ידוע
            display_text = f"Unknown ({person_name})"

        # צייר תיבת תוחמת וטקסט על הפריים
        # (Draw bounding box and text on the frame)
        if focused_person_bbox is not None:
            frame_to_display = draw_person_bbox(frame_to_display, focused_person_bbox, display_text, bbox_color, thickness=3)

        cv2.imshow('Focused Person Analysis', frame_to_display) # הצג את הפריים

        previous_focused_bbox = focused_person_bbox # שמור את ה-bbox הנוכחי לניתוח תנועה של הפריים הבא
        frame_count += 1 # קדם מונה פריימים

        if cv2.waitKey(1) & 0xFF == ord('q'): # יציאה בלחיצת 'q'
            break

    # שחרר משאבים ונקה
    # (Release resources and clean up)
    cap.release() # שחרר את אובייקט הווידאו
    cv2.destroyAllWindows() # סגור את כל חלונות OpenCV
    cleanup_temp_dir(TEMP_DIR) # נקה את התיקייה הזמנית


if __name__ == "__main__":
    main()