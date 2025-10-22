import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import os # כדי לטעון משתני סביבה

# מאפשרת לשלוח מיילים מחשבון Gmail באופן אוטומטי,
# על ידי התחברות לשרת ה-SMTP של ג'ימייל ושימוש בסיסמת אפליקציה מיוחדת, והיא כוללת טיפול בשגיאות
def send_gmail(sender_email, sender_app_password, receiver_email, subject, body):
    """
    שולח מייל מ-Gmail באמצעות שרת SMTP וסיסמת אפליקציה.

    :param sender_email: כתובת המייל השולחת (חשבון Gmail).
    :param sender_app_password: סיסמת האפליקציה שיצרת עבור Gmail.
    :param receiver_email: כתובת המייל המקבלת.
    :param subject: נושא המייל.
    :param body: תוכן המייל.
    """
    # הגדרות שרת SMTP של Gmail
    smtp_server = "smtp.gmail.com"
    smtp_port = 587  # פורט ל-TLS

    # יצירת אובייקט הודעה
    msg = MIMEMultipart()
    msg['From'] = sender_email
    msg['To'] = receiver_email
    msg['Subject'] = subject

    # צירוף תוכן המייל (אפשר גם 'html' במקום 'plain' אם רוצים)
    msg.attach(MIMEText(body, 'plain', 'utf-8')) # הוספתי utf-8 לטיפול טוב יותר בעברית

    try:
        # התחברות לשרת ה-SMTP של Gmail
        server = smtplib.SMTP(smtp_server, smtp_port)
        server.starttls()  # הפעלת הצפנת TLS
        server.login(sender_email, sender_app_password)

        # שליחת המייל
        server.send_message(msg)
        print(f"המייל נשלח בהצלחה מ- {sender_email} אל {receiver_email}!")

    except smtplib.SMTPAuthenticationError:
        print("שגיאת אימות: ודא שכתובת המייל וסיסמת האפליקציה נכונות, ושאימות דו-שלבי פעיל.")
    except Exception as e:
        print(f"שגיאה כללית בשליחת המייל: {e}")
    finally:
        if 'server' in locals() and server:
            server.quit() # סגירת החיבור לשרת

# כדי לשלוח מייל בפועל, על ידי הגדרת פרטי חשבון השולח  פרטי הנמען, נושא המייל ותוכנו
# ולאחר מכן הפעלת הפונקציה עם נתונים אלו
if __name__ == "__main__":
    # החלפה זו בפרטי החשבון שלך ובסיסמת האפליקציה!
    # מומלץ בחום להשתמש במשתני סביבה ולא לקודד ישירות כאן
    my_gmail_user = "moriam1945@gmail.com" # החלף בכתובת המייל המלאה שלך
    my_gmail_app_password = "czfi wwpg hgtv quoe" # החלף בסיסמת האפליקציה שקיבלת

    recipient = "moriam1945@gmail.com" # שנה לכתובת המייל של המקבל
    subject = "נושא המייל מהפרויקט שלי"
    body = "שלום רב,\n\nזהו מייל שנשלח מפרויקט הפייתון שלי באמצעות חשבון Gmail וסיסמת אפליקציה.\n\nבברכה,\nהפרויקט שלך"

    send_gmail(my_gmail_user, my_gmail_app_password, recipient, subject, body)