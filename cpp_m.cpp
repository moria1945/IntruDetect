#include <iostream> 
#include <thread>  
#include <vector>  
#include <string>  
#include <sstream> 
#include <fstream>  
#include <future>   
#include <map>      
#include <stdexcept> 
#include <algorithm> 
#include <array>     

//הכללות של ספריות OpenCV 
#include <opencv2/opencv.hpp>
#include <opencv2/imgcodecs.hpp> 
#include <opencv2/imgproc.hpp>   // עבור פונקציות עיבוד תמונה
#include <opencv2/highgui.hpp>   // עבור פונקציות תצוגה אם נדרש לצורך דיבוג


#ifdef _WIN32
#include <windows.h> // עבור פונקציות API של Windows
#include <fcntl.h>   
#include <io.h>     
#else
#include <unistd.h>    
#include <sys/wait.h> 
#endif

//  הגדרות נתיבים גלובליות 
// אנא עדכן נתיבים אלו לנתיבים הנכונים במערכת שלך
const std::string PYTHON_EXE_PATH = "C:\\Users\\moiam\\AppData\\Local\\Programs\\Python\\Python38\\python.exe";
const std::string PYTHON_SCRIPTS_BASE_PATH = "C:\\Users\\moiam\\Documents\\Project\\aldoritmem\\";
// -------------------------------

// מבנה לאחסון תוצאה מכל תהליכון
struct ThreadResult {
    std::string name;       // שם התהליכון 
    std::string raw_output; // יחזיק את פלט המחרוזת הגולמי מ-Python
};

// פונקציה להרצת סקריפט Python עם ארגומנטים וקבלת הפלט שלו
std::string runPythonScript(const std::string& scriptFileName, const std::vector<std::string>& args) {
    std::string scriptFullPath = PYTHON_SCRIPTS_BASE_PATH + scriptFileName;

    // בנה את הפקודה המלאה: נתיב ל-python.exe + נתיב לסקריפט + ארגומנטים
    std::string command = "\"" + PYTHON_EXE_PATH + "\" \"" + scriptFullPath + "\""; // עטוף נתיב Python במירכאות
    for (const auto& arg : args) {
        command += " \"" + arg + "\""; // עטוף כל ארגומנט במירכאות למקרה שיש רווחים בנתיבים
    }

    std::cerr << "[CPP Debug] Python command to execute: " << command << std::endl; // הדפס את הפקודה ל-stderr

    std::string output = "";

#ifdef _WIN32
    // הגדרות אבטחה לצינור
    SECURITY_ATTRIBUTES saAttr;
    saAttr.nLength = sizeof(SECURITY_ATTRIBUTES);
    saAttr.bInheritHandle = TRUE; // ידית הצינור ניתנת להורשה
    saAttr.lpSecurityDescriptor = NULL;

    HANDLE hReadPipe, hWritePipe;
    // יצירת צינור עבור stdout/stderr של תהליך הילד
    if (!CreatePipe(&hReadPipe, &hWritePipe, &saAttr, 0)) {
        std::cerr << "[CPP Error] CreatePipe failed (" << GetLastError() << ")." << std::endl;
        return "";
    }
    // חשוב: ודא ש-hReadPipe לא עובר בירושה, כך שתהליך האב לא יקרא מעצמו
    SetHandleInformation(hReadPipe, HANDLE_FLAG_INHERIT, 0);

    STARTUPINFOA si; // מבנה STARTUPINFOA עבור מידע על תהליך ההפעלה
    PROCESS_INFORMATION pi; // מבנה PROCESS_INFORMATION עבור מידע על התהליך שנוצר
    ZeroMemory(&si, sizeof(si)); // אפס את המבנים
    si.cb = sizeof(si);
    si.hStdError = hWritePipe; // הפנה את stderr לצינור
    si.hStdOutput = hWritePipe; // הפנה את stdout לצינור
    si.dwFlags |= STARTF_USESTDHANDLES; // השתמש בידיות סטנדרטיות שהוגדרו

    ZeroMemory(&pi, sizeof(pi));

    // יצירת תהליך הפייתון
    if (!CreateProcessA(NULL,             // שם המודול הניתן להרצה 
                        const_cast<char*>(command.c_str()), // שורת הפקודה
                        NULL,             // מאפייני אבטחה של התהליך
                        NULL,             // מאפייני אבטחה של התהליכון הראשי
                        TRUE,             // האם הידיות ניתנות להורשה? (חשוב עבור הצינור)
                        0,                // דגלי יצירה
                        NULL,             // סביבת תהליך (ברירת מחדל: סביבת האב)
                        NULL,             // ספריית עבודה נוכחית (ברירת מחדל: ספריית האב)
                        &si,              // מצביע למבנה STARTUPINFO
                        &pi))             // מצביע למבנה PROCESS_INFORMATION
    {
        CloseHandle(hReadPipe);  // סגור את הידיות במקרה של כשל
        CloseHandle(hWritePipe);
        std::cerr << "[CPP Error] CreateProcess failed for command: " << command << ". Error: " << GetLastError() << std::endl;
        return "";
    }

    CloseHandle(hWritePipe); // סגור את קצה הכתיבה של הצינור בתהליך האב
                             

    std::stringstream ss; // streambuffer לאיסוף הפלט
    char buffer[4096]; // באפר לקריאה
    DWORD bytesRead;   // מספר הבתים שנקראו
    // קרא מצינור הקריאה עד שתהליך הילד נסגר או מתנתק
    while (ReadFile(hReadPipe, buffer, sizeof(buffer) - 1, &bytesRead, NULL) && bytesRead > 0) {
        buffer[bytesRead] = '\0'; // ודא סיום null
        ss << buffer;             // הוסף לבאפר הסטרינג
    }
    output = ss.str(); // קבל את הפלט השלם

    CloseHandle(hReadPipe); // סגור את הידיות
    WaitForSingleObject(pi.hProcess, INFINITE); // המתן עד שתהליך הילד יסתיים
    CloseHandle(pi.hProcess); // סגור ידיות תהליך ותהליכון
    CloseHandle(pi.hThread);
#else 
#endif
    return output;
}

// פונקציה לניתוח התוצאות ממחרוזת פלט ה-Python
std::map<std::string, std::string> parsePythonOutput(const std::string& py_raw_output) {
    std::map<std::string, std::string> results;
    size_t start_pos = py_raw_output.find("RESULTS_START");
    size_t end_pos = py_raw_output.find("RESULTS_END");

    if (start_pos != std::string::npos && end_pos != std::string::npos && end_pos > start_pos) {
        // חלץ את בלוק התוצאות
        std::string output_block = py_raw_output.substr(start_pos + std::string("RESULTS_START").length(), end_pos - (start_pos + std::string("RESULTS_START").length()));
        std::stringstream ss(output_block);
        std::string line;
        while (std::getline(ss, line)) {
            // הסר רווחים מובילים/סופיים
            line.erase(0, line.find_first_not_of(" \t\n\r\f\v"));
            line.erase(line.find_last_not_of(" \t\n\r\f\v") + 1);

            size_t colon_pos = line.find(":");
            if (colon_pos != std::string::npos) {
                std::string key = line.substr(0, colon_pos);
                std::string value = line.substr(colon_pos + 1);
                // הסר רווחים מובילים/סופיים מהמפתח ומהערך
                key.erase(0, key.find_first_not_of(" \t\n\r\f\v"));
                key.erase(key.find_last_not_of(" \t\n\r\f\v") + 1);
                value.erase(0, value.find_first_not_of(" \t\n\r\f\v"));
                value.erase(value.find_last_not_of(" \t\n\r\f\v") + 1);
                results[key] = value;
            } else if (!line.empty()) { // אופציונלי: רשום שורות שאינן תואמות לפורמט Key:Value
                std::cerr << "[CPP Warning - parsePythonOutput] Line not compatible with Key:Value format: '" << line << "'" << std::endl;
            }
        }
    } else {
        // הדפס אזהרה זו רק אם הפלט אינו רק "RESULTS_START\nRESULTS_END" שזו תוצאה ריקה תקינה
        if (py_raw_output.find("RESULTS_START\nRESULTS_END") == std::string::npos &&
            py_raw_output.find("RESULTS_START\nRESULTS_END\n") == std::string::npos) { // בדוק גם עבור תו שורה חדשה בסוף
            std::cerr << "[CPP Warning] No distinctive 'RESULTS_START'/'RESULTS_END' found in Python output. Raw output:\n" << py_raw_output << std::endl;
        }
    }
    return results;
}

// פונקציה לבדיקה אם מחרוזת היא "none" (לא תלוי רישיות)
bool isNone(const std::string& s) {
    std::string lower_s = s;
    std::transform(lower_s.begin(), lower_s.end(), lower_s.begin(), ::tolower);
    return lower_s == "none";
}

// === פונקציות תהליכונים (Thread Functions) ===

void runFaceRecognition(const std::string& focusedPersonCropPath, const std::string& focusedPersonBboxStr, std::promise<ThreadResult> resultPromise) {
    std::cerr << "[CPP Debug - FaceRecognition] Starting face recognition thread." << std::endl;
    std::cerr << "[CPP Debug - FaceRecognition] Focused person crop path: " << focusedPersonCropPath << std::endl;
    std::cerr << "[CPP Debug - FaceRecognition] Focused person bbox string: " << focusedPersonBboxStr << std::endl;

    if (isNone(focusedPersonCropPath) || isNone(focusedPersonBboxStr)) {
        std::cerr << "[CPP Debug - FaceRecognition] Skipping: Focused person crop path or bbox string is 'none'." << std::endl;
        // החזר תוצאת ברירת מחדל אם אין נתונים תקפים
        resultPromise.set_value({"FaceRecognition", "RESULTS_START\nPersonName:Unknown\nRESULTS_END"});
        return;
    }

    cv::Mat crop_img = cv::imread(focusedPersonCropPath, cv::IMREAD_COLOR);
    if (crop_img.empty()) {
        std::cerr << "[CPP Error - FaceRecognition] Failed to load crop image: " << focusedPersonCropPath << ". Image is empty." << std::endl;
        resultPromise.set_value({"FaceRecognition", "RESULTS_START\nPersonName:Unknown\nRESULTS_END"});
        return;
    }
    std::cerr << "[CPP Debug - FaceRecognition] Successfully loaded crop image: " << focusedPersonCropPath << " Dimensions: " << crop_img.cols << "x" << crop_img.rows << std::endl;
    cv::imwrite("debug_facerec_crop_output.png", crop_img); 

    std::string rawOutput = runPythonScript("facerec_from_webcam_faster.py", {"--image", focusedPersonCropPath, "--bbox", focusedPersonBboxStr});
    resultPromise.set_value({"FaceRecognition", rawOutput});
    std::cerr << "[CPP Debug - FaceRecognition] Face recognition thread finished." << std::endl;
}

void runMotionTracking(const std::string& fullFramePath, const std::string& currentBboxStr, const std::string& previousBboxStr, std::promise<ThreadResult> resultPromise) {
    std::cerr << "[CPP Debug - MotionTracking] Starting motion tracking thread." << std::endl;
    std::cerr << "[CPP Debug - MotionTracking] Full frame path: " << fullFramePath << std::endl;
    std::cerr << "[CPP Debug - MotionTracking] Current bbox string: " << currentBboxStr << std::endl;
    std::cerr << "[CPP Debug - MotionTracking] Previous bbox string: " << previousBboxStr << std::endl;

    if (isNone(fullFramePath) || isNone(currentBboxStr)) { 
        std::cerr << "[CPP Debug - MotionTracking] Skipping: Full frame path or current bbox string is 'none'." << std::endl;
        resultPromise.set_value({"MotionTracking", "RESULTS_START\nMovement:Stationary\nRESULTS_END"}); // ברירת מחדל
        return;
    }

    cv::Mat full_frame_img = cv::imread(fullFramePath, cv::IMREAD_COLOR);
    if (full_frame_img.empty()) {
        std::cerr << "[CPP Error - MotionTracking] Failed to load full frame image: " << fullFramePath << ". Image is empty." << std::endl;
        resultPromise.set_value({"MotionTracking", "RESULTS_START\nMovement:Stationary\nRESULTS_END"});
        return;
    }
    std::cerr << "[CPP Debug - MotionTracking] Successfully loaded full frame image: " << fullFramePath << " Dimensions: " << full_frame_img.cols << "x" << full_frame_img.rows << std::endl;
    cv::imwrite("debug_motion_full_frame_output.png", full_frame_img);

    std::string rawOutput = runPythonScript("optical_flow.py", {"--current_frame", fullFramePath, "--current_bbox", currentBboxStr, "--previous_bbox", previousBboxStr});
    resultPromise.set_value({"MotionTracking", rawOutput});
    std::cerr << "[CPP Debug - MotionTracking] Motion tracking thread finished." << std::endl;
}

void runWeaponDetection(const std::string& focusedPersonCropPath, std::promise<ThreadResult> resultPromise) {
    std::cerr << "[CPP Debug - WeaponDetection] Starting weapon detection thread." << std::endl;
    std::cerr << "[CPP Debug - WeaponDetection] Focused person crop path: " << focusedPersonCropPath << std::endl;

    if (isNone(focusedPersonCropPath)) {
        std::cerr << "[CPP Debug - WeaponDetection] Skipping: Focused person crop path is 'none'." << std::endl;
        resultPromise.set_value({"WeaponDetection", "RESULTS_START\nWeaponDetected:false\nRESULTS_END"}); // ברירת מחדל
        return;
    }

    cv::Mat crop_img = cv::imread(focusedPersonCropPath, cv::IMREAD_COLOR);
    if (crop_img.empty()) {
        std::cerr << "[CPP Error - WeaponDetection] Failed to load crop image: " << focusedPersonCropPath << ". Image is empty." << std::endl;
        resultPromise.set_value({"WeaponDetection", "RESULTS_START\nWeaponDetected:false\nRESULTS_END"});
        return;
    }
    std::cerr << "[CPP Debug - WeaponDetection] Successfully loaded crop image: " << focusedPersonCropPath << " Dimensions: " << crop_img.cols << "x" << crop_img.rows << std::endl;
    cv::imwrite("debug_weapon_crop_output.png", crop_img); 

    std::string rawOutput = runPythonScript("Faster_R-CNN.py", {"--image", focusedPersonCropPath});
    resultPromise.set_value({"WeaponDetection", rawOutput});
    std::cerr << "[CPP Debug - WeaponDetection] Weapon detection thread finished." << std::endl;
}

// פונקציית runClothingStyle מושבתת כרגע ולא תבוצע.
// כאשר תרצה להפעיל אותה, ודא שהמודל אומן ושארגומנטים (fullFramePath, focusedPersonBboxStr, allPersonsBboxesStr)
// מועברים כראוי מתהליך main.py
/*
void runClothingStyle(const std::string& fullFramePath, const std::string& focusedPersonBboxStr, const std::string& allPersonsBboxesStr, std::promise<ThreadResult> resultPromise) {
    std::cerr << "[CPP Debug - ClothingStyle] Starting clothing style analysis thread." << std::endl;
    if (isNone(fullFramePath) || isNone(focusedPersonBboxStr) || isNone(allPersonsBboxesStr)) {
        std::cerr << "[CPP Debug - ClothingStyle] Skipping: One or more arguments are 'none'." << std::endl;
        resultPromise.set_value({"ClothingStyle", "RESULTS_START\nClothingStyle:N/A\nRESULTS_END"});
        return;
    }
    std::string rawOutput = runPythonScript("CnnYolo.py", {
        "--full_frame", fullFramePath,
        "--focused_bbox", focusedPersonBboxStr,
        "--all_bboxes", allPersonsBboxesStr
    });
    resultPromise.set_value({"ClothingStyle", rawOutput});
    std::cerr << "[CPP Debug - ClothingStyle] Clothing style analysis thread finished." << std::endl;
}
*/

int main(int argc, char *argv[]) {
#ifdef _WIN32
    // הגדר את stdout ו-stderr ל-UTF-8 ב-Windows
    _setmode(_fileno(stdout), _O_U8TEXT);
    _setmode(_fileno(stderr), _O_U8TEXT);
#endif

    std::cerr << "[CPP Debug] cpp_m.exe started." << std::endl;
    std::cerr << "[CPP Debug] Number of arguments (argc): " << argc << std::endl;
    for (int i = 0; i < argc; ++i) {
        std::cerr << "[CPP Debug] argv[" << i << "]: " << argv[i] << std::endl;
    }

    // התאמת מספר הארגומנטים המגיעים מ-main_process.py:
    // argv[0]: שם התוכנית (cpp_m.exe)
    // argv[1]: full_frame_path
    // argv[2]: focused_person_crop_path
    // argv[3]: focused_person_bbox_str
    // argv[4]: previous_focused_bbox_str
    if (argc < 5) {
        std::cerr << "[CPP Error] Usage: " << argv[0] << " <full_frame_path> <focused_person_crop_path> <focused_person_bbox_str> <previous_focused_bbox_str>" << std::endl;
        return 1;
    }

    std::string fullFramePath = argv[1];
    std::string focusedPersonCropPath = argv[2];
    std::string focusedPersonBboxStr = argv[3];
    std::string previousFocusedBboxStr = argv[4];

    std::cerr << "[CPP Debug] Parsed arguments:" << std::endl;
    std::cerr << "   fullFramePath: " << fullFramePath << std::endl;
    std::cerr << "   focusedPersonCropPath: " << focusedPersonCropPath << std::endl;
    std::cerr << "   focusedPersonBboxStr: " << focusedPersonBboxStr << std::endl;
    std::cerr << "   previousFocusedBboxStr: " << previousFocusedBboxStr << std::endl;

    std::vector<std::future<ThreadResult>> futures;
    std::vector<std::thread> threads;

    // הפעל תהליכון לזיהוי פנים
    std::promise<ThreadResult> facePromise;
    futures.push_back(facePromise.get_future());
    threads.emplace_back(runFaceRecognition, focusedPersonCropPath, focusedPersonBboxStr, std::move(facePromise));

    // הפעל תהליכון למעקב תנועה
    std::promise<ThreadResult> motionPromise;
    futures.push_back(motionPromise.get_future());
    threads.emplace_back(runMotionTracking, fullFramePath, focusedPersonBboxStr, previousFocusedBboxStr, std::move(motionPromise));

    // הפעל תהליכון לזיהוי נשק
    std::promise<ThreadResult> weaponPromise;
    futures.push_back(weaponPromise.get_future());
    threads.emplace_back(runWeaponDetection, focusedPersonCropPath, std::move(weaponPromise));

    // המתן לכל התהליכונים שיסיימו וקבל את התוצאות
    std::map<std::string, std::string> consolidatedResults;

    for (auto& future : futures) {
        ThreadResult rawResult = future.get();
        std::cerr << "[CPP Debug] Received raw output from " << rawResult.name << ":\n" << rawResult.raw_output << std::endl;
        // נתח את הפלט הגולמי של כל מודול Python
        std::map<std::string, std::string> moduleResults = parsePythonOutput(rawResult.raw_output);

        // שלב את התוצאות למפה המאוחדת
        for (const auto& pair : moduleResults) {
            consolidatedResults[pair.first] = pair.second;
        }
    }

    // הדפס את כל התוצאות המאוחדות ל-Python (main_process.py) באמצעות stdout
    std::cout << "RESULTS_START" << std::endl;
    for (const auto& pair : consolidatedResults) {
        std::cout << pair.first << ":" << pair.second << std::endl;
    }
    std::cout << "RESULTS_END" << std::endl;

    // צרף את כל התהליכונים
    for (auto& thread : threads) {
        if (thread.joinable()) {
            thread.join();
        }
    }
    std::cerr << "[CPP Debug] cpp_m.exe finished." << std::endl;

    return 0;
}