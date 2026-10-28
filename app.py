import cv2
import mediapipe as mp
import time
import threading
import base64
import os

from mediapipe.python._framework_bindings import resource_util
from flask import Flask, render_template, jsonify

# =========================================================
# MEDIAPIPE ФАЙЛ ЖОЛЫН ТҮЗЕТУ
# =========================================================
mp_dir = os.path.dirname(mp.__file__)
resource_util.set_resource_dir(mp_dir)

# =========================================================
# FLASK
# =========================================================
app = Flask(__name__)

# =========================================================
# MEDIAPIPE POSE
# =========================================================
mp_pose = mp.solutions.pose

pose = mp_pose.Pose(
    static_image_mode=False,
    model_complexity=1,
    smooth_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)
# =========================================================
# КАМЕРА
# =========================================================
camera = cv2.VideoCapture(0)

if not camera.isOpened():
    raise RuntimeError("Камера ашылмады!")

# =========================================================
# LANDMARKS
# =========================================================
LANDMARKS = {
    "nose": mp_pose.PoseLandmark.NOSE,
    "left_shoulder": mp_pose.PoseLandmark.LEFT_SHOULDER,
    "right_shoulder": mp_pose.PoseLandmark.RIGHT_SHOULDER,
    "left_elbow": mp_pose.PoseLandmark.LEFT_ELBOW,
    "right_elbow": mp_pose.PoseLandmark.RIGHT_ELBOW,
    "left_wrist": mp_pose.PoseLandmark.LEFT_WRIST,
    "right_wrist": mp_pose.PoseLandmark.RIGHT_WRIST,
    "left_hip": mp_pose.PoseLandmark.LEFT_HIP,
    "right_hip": mp_pose.PoseLandmark.RIGHT_HIP,
    "left_knee": mp_pose.PoseLandmark.LEFT_KNEE,
    "right_knee": mp_pose.PoseLandmark.RIGHT_KNEE,
    "left_ankle": mp_pose.PoseLandmark.LEFT_ANKLE,
    "right_ankle": mp_pose.PoseLandmark.RIGHT_ANKLE
}

# =========================================================
# ОЙЫН КҮЙІ
# =========================================================
state = {
    "level": 1,
    "started": False,
    "countdown": False,
    "countdown_value": 0,

    "message": "Бастау батырмасын басыңыз",
    "message_type": "info",

    "progress": 0,
    "score": 0,
    "level_score": 0,

    "camera_ok": False,
    "points_visible": 0,
    "required_points_count": 4,

    "level_complete": False,
    "game_complete": False,

    "shoulder_difference": 0,
    "left_angle": 0,
    "right_angle": 0,

    "camera_image": "",

    "rocket_launch_id": 0
}

# =========================================================
# ПАРАМЕТРЛЕР
# =========================================================
LEVEL_HOLD_TIME = 3.0  # Қолыңызды/деңені түзу 3 секунд ұстап тұру керек
VISIBILITY = 0.35

hold_start = None


# =========================================================
# НҮКТЕНІ АЛУ
# =========================================================
def get_point(landmarks, name):
    landmark = landmarks[LANDMARKS[name]]

    if landmark.visibility < VISIBILITY:
        return None

    return [landmark.x, landmark.y, landmark.z]


# =========================================================
# ҚАЖЕТТІ НҮКТЕЛЕР (3 КЕЗЕҢ)
# =========================================================
def check_required_points(points, level):
    # 1 - денені түзу ұстау
    if level == 1:
        req_keys = [
            "left_shoulder",
            "right_shoulder"
        ]

    # 2 - қолды бас деңгейінде ұстау
    elif level == 2:
        req_keys = [
            "left_shoulder",
            "right_shoulder",
            "left_wrist",
            "right_wrist",
            "nose"
        ]

    # 3 - екі қолды жоғары көтеру
    elif level == 3:
        req_keys = [
            "left_shoulder",
            "right_shoulder",
            "left_wrist",
            "right_wrist",
            "nose"
        ]

    else:
        req_keys = []

    state["required_points_count"] = len(req_keys)

    for key in req_keys:
        if points.get(key) is None:
            return False

    return True


# =========================================================
# 1-КЕЗЕҢ: ТҮЗУ ТҰРУ
# =========================================================
def check_level_1(points):
    if not check_required_points(points, 1):
        state["camera_ok"] = False
        state["message"] = "Камераға толық көрініңіз"
        state["message_type"] = "warning"
        return False

    state["camera_ok"] = True

    left = points["left_shoulder"]
    right = points["right_shoulder"]

    difference = abs(left[1] - right[1])
    state["shoulder_difference"] = round(difference, 3)

    if difference <= 0.06:
        return True

    state["message"] = "Денеңізді түзу ұстаңыз"
    state["message_type"] = "warning"
    return False


# =========================================================
# 2-КЕЗЕҢ: ҚОЛДЫ БАС ДЕҢГЕЙІНДЕ ҰСТАУ
# =========================================================
def check_level_2(points):
    if not check_required_points(points, 2):
        state["camera_ok"] = False
        state["message"] = "Басыңыз бен екі қолыңыз камерада көрінсін"
        state["message_type"] = "warning"
        return False

    state["camera_ok"] = True

    nose_y = points["nose"][1]

    left_wrist_y = points["left_wrist"][1]
    right_wrist_y = points["right_wrist"][1]

    left_shoulder_y = points["left_shoulder"][1]
    right_shoulder_y = points["right_shoulder"][1]

    left_ok = (
        left_wrist_y < left_shoulder_y - 0.03
        and left_wrist_y >= nose_y - 0.12
        and left_wrist_y <= nose_y + 0.15
    )

    right_ok = (
        right_wrist_y < right_shoulder_y - 0.03
        and right_wrist_y >= nose_y - 0.12
        and right_wrist_y <= nose_y + 0.15
    )

    if left_ok and right_ok:
        return True

    state["message"] = "Екі қолды бас деңгейінде ұстаңыз"
    state["message_type"] = "warning"
    return False


# =========================================================
# 3-КЕЗЕҢ: ЕКІ ҚОЛДЫ ЖОҒАРЫ КӨТЕРУ
# =========================================================
def check_level_3(points):
    if not check_required_points(points, 3):
        state["camera_ok"] = False
        state["message"] = "Екі қолыңыз бен басыңыз көрінсін"
        state["message_type"] = "warning"
        return False

    state["camera_ok"] = True

    left_wrist_y = points["left_wrist"][1]
    right_wrist_y = points["right_wrist"][1]

    left_shoulder_y = points["left_shoulder"][1]
    right_shoulder_y = points["right_shoulder"][1]

    left_ok = left_wrist_y < left_shoulder_y - 0.12
    right_ok = right_wrist_y < right_shoulder_y - 0.12

    if left_ok and right_ok:
        return True

    state["message"] = "Екі қолды жоғары көтеріңіз"
    state["message_type"] = "warning"
    return False


# =========================================================
# АҒЫМДАҒЫ КЕЗЕҢДІ ТЕКСЕРУ
# =========================================================
def check_current_level(points):
    level = state["level"]

    if level == 1:
        return check_level_1(points)

    if level == 2:
        return check_level_2(points)

    if level == 3:
        return check_level_3(points)

    return False


# =========================================================
# КЕЛЕСІ КЕЗЕҢДІ АВТОМАТТЫ ТҮРДЕ БІРДЕН БАСТАУ
# =========================================================
def auto_start_next_level():
    global hold_start

    if state["game_complete"]:
        return

    state["level"] += 1
    state["level_complete"] = False
    state["level_score"] = 0
    state["progress"] = 0

    # Кері санақсыз, бірден бастау
    state["countdown"] = False
    state["countdown_value"] = 0
    state["started"] = True
    state["message"] = f"{state['level']}-кезең басталды!"
    state["message_type"] = "success"

    hold_start = None


# =========================================================
# КЕЗЕҢДІ ӨҢДЕУ
# =========================================================
def process_level(points):
    global hold_start

    if not state["started"]:
        return

    correct = check_current_level(points)

    if correct:
        if hold_start is None:
            hold_start = time.time()

        elapsed = time.time() - hold_start
        progress = min(elapsed / LEVEL_HOLD_TIME, 1.0)

        state["progress"] = progress * 100
        state["message"] = "Дұрыс! Осы қалыпты 3 секунд ұстап тұрыңыз"
        state["message_type"] = "success"

        # 3 секунд ұстап тұрғаннан кейін ғана өтеді
        if elapsed >= LEVEL_HOLD_TIME:
            state["level_score"] = 100
            state["score"] += 100
            state["level_complete"] = True
            state["started"] = False
            state["progress"] = 100

            state["rocket_launch_id"] += 1
            hold_start = None

            if state["level"] < 3:
                state["message"] = f"{state['level']}-кезең орындалды! +100 ұпай"
                state["message_type"] = "level-complete"

                # Келесі кезеңге ешқандай кідіріссіз бірден көшу
                threading.Thread(
                    target=auto_start_next_level,
                    daemon=True
                ).start()

            else:
                state["game_complete"] = True
                state["message"] = "Барлық 3 кезең сәтті орындалды! +100 ұпай"
                state["message_type"] = "game-complete"

    else:
        hold_start = None
        state["progress"] = 0


# =========================================================
# КАМЕРА THREAD
# =========================================================
def camera_loop():
    while True:
        success, frame = camera.read()

        if not success:
            continue

        frame = cv2.flip(frame, 1)

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(rgb)

        points = {}
        visible = 0

        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark

            for name in LANDMARKS:
                point = get_point(landmarks, name)
                points[name] = point

                if point is not None:
                    visible += 1

                    x = int(point[0] * frame.shape[1])
                    y = int(point[1] * frame.shape[0])

                    cv2.circle(
                        frame,
                        (x, y),
                        6,
                        (0, 255, 255),
                        -1
                    )

            connections = [
                ("left_shoulder", "left_elbow"),
                ("left_elbow", "left_wrist"),
                ("right_shoulder", "right_elbow"),
                ("right_elbow", "right_wrist"),

                ("left_hip", "left_knee"),
                ("left_knee", "left_ankle"),
                ("right_hip", "right_knee"),
                ("right_knee", "right_ankle"),

                ("left_shoulder", "left_hip"),
                ("right_shoulder", "right_hip"),
                ("left_shoulder", "right_shoulder"),
                ("left_hip", "right_hip")
            ]

            for a, b in connections:
                if points.get(a) is not None and points.get(b) is not None:
                    p1 = points[a]
                    p2 = points[b]

                    x1 = int(p1[0] * frame.shape[1])
                    y1 = int(p1[1] * frame.shape[0])

                    x2 = int(p2[0] * frame.shape[1])
                    y2 = int(p2[2] * frame.shape[0]) if len(p2) > 2 and False else int(p2[1] * frame.shape[0])

                    cv2.line(
                        frame,
                        (x1, y1),
                        (x2, y2),
                        (255, 255, 255),
                        2
                    )

        state["points_visible"] = visible

        process_level(points)

        _, buffer = cv2.imencode(".jpg", frame)
        jpg_as_text = base64.b64encode(buffer).decode("utf-8")

        state["camera_image"] = (
            "data:image/jpeg;base64," + jpg_as_text
        )

        time.sleep(0.01)


# =========================================================
# ROUTES
# =========================================================
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/state")
def get_state():
    return jsonify(dict(state))


# =========================================================
# 1-КЕЗЕҢДІ БАСТАУ (Дайындалусыз, бірден басталады)
# =========================================================
@app.route("/start", methods=["POST"])
def start_level():
    global hold_start

    if state["game_complete"]:
        return jsonify({"success": False})

    if state["started"]:
        return jsonify({
            "success": False,
            "message": "Ойын жүріп жатыр"
        })

    if state["level"] != 1 or state["level_complete"]:
        return jsonify({
            "success": False,
            "message": "Бұл кезең автоматты түрде басталады"
        })

    # 3, 2, 1 санауды айналып өтіп, бірден бастаймыз
    state["countdown"] = False
    state["countdown_value"] = 0
    state["started"] = True
    state["progress"] = 0
    state["message"] = "Бастаңыз!"
    state["message_type"] = "success"

    hold_start = None

    return jsonify({
        "success": True,
        "level": 1
    })


# =========================================================
# RESET
# =========================================================
@app.route("/reset", methods=["POST"])
def reset_game():
    global hold_start

    state["level"] = 1
    state["started"] = False
    state["countdown"] = False
    state["countdown_value"] = 0

    state["level_complete"] = False
    state["game_complete"] = False

    state["progress"] = 0
    state["score"] = 0
    state["level_score"] = 0

    state["rocket_launch_id"] = 0

    state["message"] = "Бастау батырмасын басыңыз"
    state["message_type"] = "info"

    hold_start = None

    return jsonify({"success": True})


# =========================================================
# КАМЕРА ІСКЕ ҚОСУ
# =========================================================
threading.Thread(
    target=camera_loop,
    daemon=True
).start()


# =========================================================
# SERVER
# =========================================================
if __name__ == "__main__":
    app.run(
        host="127.0.0.1",
        port=5000,
        debug=False,
        threaded=True
    )