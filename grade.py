# מחשבת ציון חשד עבור אדם על ידי סיכום נקודות המוקצות
# למאפיינים חשודים שונים שנצפו בו, כגון זיהוי כלי נשק או פנים מכוסות
def calculate_suspicion_score(
    is_identified_as_suspicious=False,
    has_weapon=False,
    has_burglary_tool=False,
    face_covered=False,
    repeated_pattern=False,
    fast_walk=False,
    inappropriate_clothing=False,
    sharp_movements=False
):
    """
    מחשב ציון חשד על בסיס מאפיינים שונים של התנהגות/מראה.

    :param is_identified_as_suspicious: האם זוהה כחשוד באופן כללי (לדוגמה, ע"י AI מתקדם או ניתוח נוסף).
    :param has_weapon: האם זוהה כלי נשק.
    :param has_burglary_tool: האם זוהה כלי פריצה.
    :param face_covered: האם הפנים מכוסות.
    :param repeated_pattern: האם נצפה דפוס חוזר (לדוגמה, הליכה במעגלים, תצפית חוזרת).
    :param fast_walk: האם האדם הולך במהירות.
    :param inappropriate_clothing: האם הלבוש לא תואם (לדוגמה, מעיל כבד בקיץ).
    :param sharp_movements: האם נצפו תנועות חדות.
    :return: ציון החשד הכולל.
    """

    # הגדרת הניקוד לכל מאפיין
    scoring_factors = {
        "מזהה כחשוד": 30,
        "זיהוי כלי נשק": 20,
        "זיהוי כלי פריצה": 5,
        "פנים מכוסות": 15,
        "חזרה על אופן": 15,
        "הליכה מהירה": 2,
        "לבוש לא תואם": 10,
        "תנועות חדות": 3
    }

    current_score = 0

    # הוספת נקודות אם המאפיין קיים
    if is_identified_as_suspicious:
        current_score += scoring_factors["מזהה כחשוד"]
    if has_weapon:
        current_score += scoring_factors["זיהוי כלי נשק"]
    if has_burglary_tool:
        current_score += scoring_factors["זיהוי כלי פריצה"]
    if face_covered:
        current_score += scoring_factors["פנים מכוסות"]
    if repeated_pattern:
        current_score += scoring_factors["חזרה על אופן"]
    if fast_walk:
        current_score += scoring_factors["הליכה מהירה"]
    if inappropriate_clothing:
        current_score += scoring_factors["לבוש לא תואם"]
    if sharp_movements:
        current_score += scoring_factors["תנועות חדות"]

    return current_score


# דוגמה 1: חשוד קלאסי 100%
score1 = calculate_suspicion_score(
    is_identified_as_suspicious=True,
    has_weapon=True,
    has_burglary_tool=True,
    face_covered=True,
    repeated_pattern=True,
    fast_walk=True,
    inappropriate_clothing=True,
    sharp_movements=True
)
print(f"ציון חשד לדוגמה 1 (כל המאפיינים): {score1}") # אמור להיות 100

# דוגמה 2: אדם עם נשק בלבד 20%
score2 = calculate_suspicion_score(
    has_weapon=True
)
print(f"ציון חשד לדוגמה 2 (רק נשק): {score2}") # אמור להיות 20

# דוגמה 3: אדם עם פנים מכוסות ותנועות חדות
score3 = calculate_suspicion_score(
    face_covered=True,
    sharp_movements=True
)
print(f"ציון חשד לדוגמה 3 (פנים מכוסות ותנועות חדות): {score3}") # אמור להיות 15 + 3 = 18

# דוגמה 4: מישהו שמכין פריצה (כלי פריצה, פנים מכוסות, חזרה על אופן)
score4 = calculate_suspicion_score(
    has_burglary_tool=True,
    face_covered=True,
    repeated_pattern=True
)
print(f"ציון חשד לדוגמה 4 (הכנת פריצה): {score4}") # אמור להיות 5 + 15 + 15 = 35

# דוגמה 5: אדם עם לבוש לא תואם והולך מהר
score5 = calculate_suspicion_score(
    inappropriate_clothing=True,
    fast_walk=True
)
print(f"ציון חשד לדוגמה 5 (לבוש לא תואם והליכה מהירה): {score5}") # אמור להיות 10 + 2 = 12