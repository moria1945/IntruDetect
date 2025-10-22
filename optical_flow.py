import numpy as np 
import cv2 as cv 
import collections # לספריית deque (רשימה דו-כיוונית עם גודל קבוע)
import argparse # ספרייה לניתוח ארגומנטים משורת הפקודה
import sys # ספרייה לגישה למערכת ההפעלה, כולל קלט/פלט סטנדרטי

sys.stdout.reconfigure(encoding='utf-8')
sys.stderr.reconfigure(encoding='utf-8')

# 1.  פרמטרים לזיהוי נקודות עניין
feature_params = dict(maxCorners=1000, # מקסימום נקודות לזיהוי
                      qualityLevel=0.01, # איכות מינימלית של הנקודה
                      minDistance=1, # מרחק מינימלי בין נקודות
                      blockSize=3) # גודל בלוק לחישובים

# פרמטרים לאלגוריתם לואקס קאנדה עצמו
lk_params = dict(winSize=(15, 15), # גודל חלון החיפוש
                 maxLevel=2, # רמת הפירמידה
                 criteria=(cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 0.03)) # קריטריוני סיום איטרציות

# 2. הגדרת ספים 
FAST_SPEED_THRESHOLD = 9 # סף מהירות לתנועה מהירה 
SHARP_CHANGE_THRESHOLD = 30.0 # סף להבדל ממוצע במהירות
MIN_SPEED_FOR_TURN_DETECTION = 5 # מהירות מינימלית הנדרשת לזיהוי פנייה חדה
SHARP_TURN_ANGLE_DEGREE = 45 # זווית (במעלות) לזיהוי פנייה חדה

# 3. משתנים גלובליים 

_prev_gray_frame = None # הפריים הקודם בשחור-לבן
_prev_points = None # הנקודות הקודמות שזוהו
_older_points_history = collections.deque(maxlen=10) # יאחסן עד 10 סטים של נקודות ישנות יותר, לצורך חישוב זוויות
_frame_speed_history = collections.deque(maxlen=10) # יאחסן עד 10 ממוצעי מהירות של פריימים, לצורך שינוי מהירות חד
_frame_count = 0 # מונה פריימים (מאז האתחול)

# מאפסת את כל נתוני המעקב של תנועת אובייקטים, ובכך מכינה את המערכת לתחילת ניתוח חדש של סרטון וידאו
def reset_optical_flow_tracker():
    """
    Resets the tracker's state (should be called when starting a new video).
    מאפס את מצב המעקב (יש לקרוא לפונקציה זו בעת התחלת סרטון חדש).
    """
    global _prev_gray_frame, _prev_points, _older_points_history, _frame_speed_history, _frame_count
    _prev_gray_frame = None
    _prev_points = None
    _older_points_history.clear() # נקה את היסטוריית הנקודות
    _frame_speed_history.clear() # נקה את היסטוריית המהירות
    _frame_count = 0
    print("Optical Flow Tracker Reset.", file=sys.stderr) # הדפסת הודעת איפוס (ל-stderr)

# מנתחת את התנועה בפריים וידאו יחיד, ממירה אותו לאפור,
# ומעדכנת מונה פריימים גלובלי
def analyze_motion_in_frame(frame: np.ndarray) -> dict:
    """
    Analyzes motion in a single frame and determines if there is fast motion, sharp speed change, or a sharp turn.
    The function aims to maintain its state between calls if the state is not reset (but it will be reset with each script run
    due to how C++ likely calls it).

    Args:
        frame: The current frame image (OpenCV BGR format).

    Returns:
        A dictionary containing the motion status:
        {
            'is_fast_motion': bool,
            'is_sharp_speed_change': bool,
            'is_sharp_turn': bool,
            'avg_speed': float,
            'sharp_speed_change_magnitude': float (if occurred),
            'sharp_turn_angles_degrees': list of floats (if occurred)
        }
    """
    global _prev_gray_frame, _prev_points, _older_points_history, _frame_speed_history, _frame_count

    _frame_count += 1 # הגדל את מונה הפריימים
    current_gray_frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY) # המר את הפריים הנוכחי לאפור

    # אתחול מילון התוצאות
    results = {
        'is_fast_motion': False,
        'is_sharp_speed_change': False,
        'is_sharp_turn': False,
        'avg_speed': 0.0,
        'sharp_speed_change_magnitude': 0.0,
        'sharp_turn_angles_degrees': []
    }

    # אם זה הפריים הראשון (או שהמעקב אופס) - אתחל נקודות למעקב
    # (If this is the first frame (or tracker was reset) - initialize points for tracking)
    if _prev_gray_frame is None or _prev_points is None:
        _prev_points = cv.goodFeaturesToTrack(current_gray_frame, mask=None, **feature_params)
        _prev_gray_frame = current_gray_frame.copy()
        # כדי לאפשר חישוב פנייה, ה-deque צריך להכיל לפחות None אחד בהתחלה
        # (To enable turn calculation, we need the deque to contain at least one None initially)
        _older_points_history.append(None) # הוסף None להיסטוריה עבור הפריים הראשון
        _frame_speed_history.append(0.0) # הוסף מהירות אפסית להיסטוריה
        # print("Optical Flow: Initializing points for first frame.", file=sys.stderr)
        return results

    # 4. חישוב Optical Flow
    # (Calculate Optical Flow)
    # p1: הנקודות החדשות (בפריים הנוכחי)
    # st: סטטוס (1 אם הנקודה נמצאה, 0 אם לא)
    # err: שגיאת חישוב
    p1, st, err = cv.calcOpticalFlowPyrLK(_prev_gray_frame, current_gray_frame, _prev_points, None, **lk_params)

    # וודא שיש מספיק נקודות תקפות
    # (Ensure there are enough valid points)
    if p1 is not None and st is not None and np.count_nonzero(st) > 0:
        # סנן רק את הנקודות שעוקבו בהצלחה
        # (Filter only the points that were successfully tracked)
        good_new = p1[st == 1]
        good_old = _prev_points[st == 1]

        # 5. חישוב מהירות ממוצעת
        # (Calculate average speed)
        motion_vectors = good_new - good_old # וקטורי תנועה
        speeds = np.linalg.norm(motion_vectors, axis=1) # גודל וקטורי התנועה (מהירות)
        avg_speed = np.mean(speeds) if speeds.size > 0 else 0.0 # מהירות ממוצעת
        results['avg_speed'] = avg_speed # עדכן את מילון התוצאות בטירחה

        # שמור את המהירות ב-deque (היסטוריית מהירויות)
        # (Save speed to deque)
        _frame_speed_history.append(avg_speed)

        # 6. זיהוי תנועה מהירה
        # (1. Fast Motion)
        if avg_speed > FAST_SPEED_THRESHOLD:
            results['is_fast_motion'] = True
            # print(f"Optical Flow (Frame {_frame_count}): Average speed: {avg_speed:.2f} - Fast motion!", file=sys.stderr)

        # 7. זיהוי שינוי חד במהירות ממוצעת (לאורך 10 פריימים)
        # (2. Sharp change in average speed (over 10 frames))
        # הערה: עקב איפוס המצב בכל קריאה לסקריפט, קטע קוד זה יהיה לא יעיל
        # (Note: Due to state reset on each script call, this code section will be ineffective)
        if len(_frame_speed_history) >= 2: # יש לפחות 2 ערכים בהיסטוריה
            history_speeds = list(_frame_speed_history)[:-1] # היסטוריית המהירויות למעט האחרון
            if history_speeds:
                prev_avg_speed_history = np.mean(history_speeds) # ממוצע מהירות קודם
                if abs(avg_speed - prev_avg_speed_history) > SHARP_CHANGE_THRESHOLD:
                    results['is_sharp_speed_change'] = True
                    results['sharp_speed_change_magnitude'] = abs(avg_speed - prev_avg_speed_history)
                    # print(f"Optical Flow (Frame {_frame_count}): Sharp speed change! Current: {avg_speed:.2f}, Previous average: {prev_avg_speed_history:.2f}", file=sys.stderr)

        # 8. זיהוי פניות חדות
        # (3. Detect sharp turns)
        # כדי לזהות פנייה חדה, אנו זקוקים לנקודות מעקב אמינות מ-T-2, T-1, T.
        # מכיוון ש-Lucas-Kanade לא שומר Track ID בין פריימים, קשה לחשב זווית סיבוב מדויקת.
        # הדרך הפשוטה ביותר היא להשתמש בנקודות קודמות (older_points) שמשמעותן good_old מהפריים הקודם,
        # ו-good_new הן הנקודות הנוכחיות.
        # עם זאת, המשתנים הגלובליים מתאפסים עם כל הרצת סקריפט, מה שפוגע ביכולת לשמור היסטוריה ארוכה.
        # כדי לאפשר חישוב פנייה בסיסי, נשמור את good_old מהפריים הנוכחי בהיסטוריה.
        # (This is still very problematic without Track IDs and state persistence between calls.)
        _older_points_history.append(good_old)
        
        # אם יש לנו לפחות נקודות מ-T-2 (older_points_history[0])
        # (If we have at least points from T-2 (older_points_history[0]))
        if len(_older_points_history) >= 2: # יש לנו לפחות good_old של הפריים הנוכחי, ו-good_old של הפריים הקודם.
            # נניח ש-older_pts_from_history הוא good_old מהפריים הקודם ששרד.
            older_pts_from_history = _older_points_history[0] 
            
            if older_pts_from_history is not None and len(good_new) > 0 and len(good_old) > 0:
                # חשב את זוויות וקטורי התנועה של כל נקודה
                # (Calculate angles of motion vectors for each point)
                angles = np.degrees(np.arctan2(motion_vectors[:, 1], motion_vectors[:, 0]))
                
                # אם טווח הזוויות גדול, כלומר נקודות נעות בכיוונים שונים, זה יכול להצביע על סיבוב.
                # (If the range of angles is large, meaning points are moving in different directions, it can indicate rotation.)
                angle_range = np.max(angles) - np.min(angles)
                
                if avg_speed > MIN_SPEED_FOR_TURN_DETECTION and angle_range > SHARP_TURN_ANGLE_DEGREE:
                    results['is_sharp_turn'] = True
                    results['sharp_turn_angles_degrees'] = angles.tolist() # החזר את כל הזוויות
                    # print(f"Optical Flow (Frame {_frame_count}): Sharp turn detected! Angle range: {angle_range:.2f} degrees.", file=sys.stderr)

    else: # אין מספיק נקודות Optical Flow תקפות
        # (Not enough valid Optical Flow points)
        # בחישוב של פריים בודד, זה יקרה לעתים קרובות.
        # כדי ש-Optical Flow יעבוד, הוא זקוק לנקודות קודמות.
        # נצטרך לאתחל מחדש בפריים הבא.
        _prev_points = cv.goodFeaturesToTrack(current_gray_frame, mask=None, **feature_params)
        _prev_gray_frame = current_gray_frame.copy()
        _older_points_history.append(None) # כדי לשמר את גודל ה-deque
        _frame_speed_history.append(0.0) # כדי לשמר את גודל ה-deque
        # print(f"Optical Flow (Frame {_frame_count}): No valid Optical Flow points. Reinitializing.", file=sys.stderr)

    # עדכן למעקב (עבור הפריים הבא, אם נשמור מצב)
    # (Update for tracking (for the next frame, if we maintain state))
    # זה לא ממש עוזר עם האופן שבו C++ קורא לסקריפט (מתחיל מחדש בכל פעם).
    _prev_points = good_new.reshape(-1, 1, 2) if 'good_new' in locals() and good_new.size > 0 else None
    _prev_gray_frame = current_gray_frame.copy()

    return results

# בלוק הרצה ראשי 
# הוא מקבל ארגומנטים מתוכנית C++ (כמו נתיב תמונה וקואורדינטות תיבה),
#  טוען את התמונה, מאפס את מערכת מעקב התנועה
# מנתח את התנועה בפריים, ומדפיס את התוצאות בפלט מובנה שתוכנית ה-C++ יכולה לקרוא.
if __name__ == '__main__':
    # הגדרת ארגומנטים המועברים על ידי C++
    # (Define arguments passed by C++)
    parser = argparse.ArgumentParser(description='Lucas-Kanade Optical Flow Analyzer for a single frame.')
    parser.add_argument('--current_frame', type=str, required=True, help='Path to the current frame image.')
    parser.add_argument('--current_bbox', type=str, required=True, help='Current bounding box as "x,y,w,h".')
    parser.add_argument('--previous_bbox', type=str, default='None', help='Previous bounding box as "x,y,w,h" or "None".')
    
    args = parser.parse_args()

    # טען את הפריים
    # (Load the frame)
    frame = cv.imread(args.current_frame)
    if frame is None:
        print(f"Error: Could not load current frame from path: {args.current_frame}", file=sys.stderr)
        print("RESULTS_START")
        print("motion_tracking:Error: Current frame not found.")
        print("RESULTS_END")
        sys.exit(1)

    # אתחל את המעקב. הערה: בכל קריאה, המעקב מאותחל מחדש
    # מכיוון שסקריפט הפייתון מתחיל מחדש בכל פעם ש-C++ מריץ אותו.
    # המשמעות היא שאין "היסטוריה" בפועל בין קריאות.
    # כדי לפתור זאת, תצטרך לשמור את _prev_gray_frame ואת _prev_points ב-C++
    # ולהעביר אותם כארגומנטים (לדוגמה, נתיב לקובץ זמני עם נקודות).
    # לחלופין, לממש את לוגיקת ה-Optical Flow כולה ב-C++ עצמו.
    # לעת עתה, נמשיך עם המודל הנוכחי.
    reset_optical_flow_tracker()
    # (הערה: איפוס זה הופך למעשה את analyze_motion_in_frame לחסר מצב (stateless))

    motion_analysis_results = analyze_motion_in_frame(frame)
    
    # הדפס את התוצאה בפורמט קריא עבור יישום C++
    # (Print the result in a C++ readable format)
    print("RESULTS_START")
    print(f"is_fast_motion:{'true' if motion_analysis_results['is_fast_motion'] else 'false'}")
    print(f"avg_speed:{motion_analysis_results['avg_speed']:.2f}")
    print(f"is_sharp_speed_change:{'true' if motion_analysis_results['is_sharp_speed_change'] else 'false'}")
    if motion_analysis_results['is_sharp_speed_change']:
        print(f"sharp_speed_change_magnitude:{motion_analysis_results['sharp_speed_change_magnitude']:.2f}")
    
    # פנייה חדה:
    # מבוסס על סטיית תקן של זוויות תנועה. זה לא אידיאלי ללא Track IDs ושימור מצב.
    # נחזיר את התוצאה הבסיסית.
    # (Sharp Turn: Based on standard deviation of motion angles. This is not ideal without Track IDs. We will return the basic result.)
    print(f"is_sharp_turn:{'true' if motion_analysis_results['is_sharp_turn'] else 'false'}")
    if motion_analysis_results['is_sharp_turn']:
        # הדפס את sharp_turn_angles_degrees כמחרוזת רשימה מופרדת בפסיקים
        # (Print angles_degrees as a comma-separated string list)
        angles_str = ','.join([f"{a:.2f}" for a in motion_analysis_results['sharp_turn_angles_degrees']])
        print(f"sharp_turn_angles_degrees:[{angles_str}]") # פורמט עבור C++

    print("RESULTS_END")
    
    sys.exit(0) # יציאה מוצלחת (Successful exit)