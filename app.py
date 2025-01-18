from flask import Flask, render_template, Response
import cv2
import mediapipe as mp
import time
import math

app = Flask(__name__)

# 初始化 MediaPipe Pose 模型
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

# 全局变量
push_up_count = 0
is_up = False
last_elbow_y = None
push_up_start_time = None
# 新增跳绳和蹲起的全局变量
jump_rope_count = 0
last_jump_time = None
last_ankle_y = None
squat_count = 0
last_squat_time = None
last_hip_y = None
is_squat_down = False
jumpingjack_count=0
situp_count=0

# 计算两个点之间的欧几里得距离
def euclidean_distance(point1, point2):
    return math.sqrt((point1.x - point2.x) ** 2 + (point1.y - point2.y) ** 2)


# 计算三点之间的角度
def calculate_angle(point1, point2, point3):
    a = euclidean_distance(point1, point2)
    b = euclidean_distance(point2, point3)
    c = euclidean_distance(point1, point3)
    if a == 0 or b == 0 or c == 0:
        return 0
    cos_angle = (a ** 2 + b ** 2 - c ** 2) / (2 * a * b)
    return math.degrees(math.acos(cos_angle))


# 俯卧撑评分标准
def calculate_push_up_score(elbow_y, shoulder_y, body_angle, speed, stability):
    score = 0
    # 深度评分
    if elbow_y < shoulder_y - 0.1:  # 深度阈值可以调整
        score += 40
    else:
        score += 20

    # 平行性评分（body_angle 是身体角度的反映）
    if body_angle < 10:  # 设定一个合理的角度范围
        score += 30
    else:
        score += 15

    # 动作速度评分
    if 0.15 <= speed <= 0.5:
        score += 20
    else:
        score += 10

    # 稳定性评分
    if stability < 5:  # 如果肩膀与肘部稳定性较好
        score += 20
    else:
        score += 10

    return score


def calculate_jump_rope_score(ankle1, ankle2, knee1, knee2, wrist1, wrist2, speed, body_amplitude):
    score = 0

    # 腿部摆动幅度评分（增强权重）
    ankle_amplitude = abs(ankle1.y - ankle2.y)
    if 0.4 <= ankle_amplitude <= 0.5:
        score += 50  # 增加腿部幅度的权重
    elif 0.3 <= ankle_amplitude < 0.4 or 0.5 < ankle_amplitude <= 0.55:
        score += 40
    elif 0.2 <= ankle_amplitude < 0.3 or 0.55 < ankle_amplitude <= 0.6:
        score += 30
    else:
        score += 0

    # 手臂摆动幅度评分（增强权重）
    arm_amplitude = abs(wrist1.y - wrist2.y)
    if 0.25 <= arm_amplitude <= 0.35:
        score += 40  # 增加手臂幅度的权重
    elif 0.2 <= arm_amplitude < 0.25 or 0.35 < arm_amplitude <= 0.4:
        score += 30
    elif 0.15 <= arm_amplitude < 0.2 or 0.4 < arm_amplitude <= 0.45:
        score += 20
    else:
        score += 0

    # 速度评分（减少权重）
    if 1.5 <= speed <= 1.8:
        score += 15
    elif 1.2 <= speed < 1.5 or 1.8 < speed <= 2.0:
        score += 10
    elif 1.0 <= speed < 1.2 or 2.0 < speed <= 2.3:
        score += 5
    else:
        score += 0

    # 身体上下移动幅度评分（减少权重）
    if 0.1 <= body_amplitude <= 0.25:
        score += 10
    elif 0.05 <= body_amplitude < 0.1 or 0.25 < body_amplitude <= 0.3:
        score += 5
    elif 0.03 <= body_amplitude < 0.05 or 0.3 < body_amplitude <= 0.35:
        score += 3
    else:
        score += 0

    return score

def calculate_situp_score(shoulder, hip, knee, ankle, speed, body_amplitude, stability):
    score = 0

    # 身体弯曲角度评分
    # 假设通过肩膀与臀部之间的角度来判断仰卧起坐的深度
    body_angle = abs(shoulder.y - hip.y)  # 用 y 坐标的差值来简化计算
    if 0.2 <= body_angle <= 0.3:  # 假设合适的角度范围
        score += 50
    elif 0.3 < body_angle <= 0.4:
        score += 40
    elif 0.4 < body_angle <= 0.5:
        score += 30
    else:
        score += 10

    # 速度评分（评估仰卧起坐的速度）
    if 1.5 <= speed <= 2.0:  # 假设速度区间适合仰卧起坐的正常速度
        score += 20
    elif 1.0 <= speed < 1.5 or 2.0 < speed <= 2.5:
        score += 15
    elif 0.8 <= speed < 1.0 or 2.5 < speed <= 3.0:
        score += 10
    else:
        score += 5

    # 身体上下移动幅度评分（评估仰卧起坐的身体起伏）
    if 0.1 <= body_amplitude <= 0.25:
        score += 15
    elif 0.05 <= body_amplitude < 0.1 or 0.25 < body_amplitude <= 0.3:
        score += 10
    elif 0.03 <= body_amplitude < 0.05 or 0.3 < body_amplitude <= 0.35:
        score += 5
    else:
        score += 0

    # 稳定性评分（肩膀、膝盖和脚踝的稳定性）
    if stability < 0.05:  # 稳定性阈值（较低值表示更稳定）
        score += 20
    elif 0.05 <= stability < 0.1:
        score += 15
    else:
        score += 10

    return score


# 蹲起评分标准
def calculate_squat_score(hip1, hip2, knee1, knee2, ankle1, ankle2, speed, squat_stability, body_balance):
    score = 0
    # 深度评分：考虑膝盖、臀部和脚踝的位置关系
    depth = euclidean_distance(hip1, ankle1) + euclidean_distance(hip2, ankle2)
    if 0.4 <= depth <= 0.6:  # 深度范围可调整
        score += 30
    elif 0.3 <= depth < 0.4 or 0.6 < depth <= 0.7:
        score += 20
    else:
        score += 10
    # 速度评分
    if 0.5 <= speed <= 1.0:  # 速度范围可调整
        score += 30
    elif 1.0 < speed <= 1.5:
        score += 20
    else:
        score += 10
    # 稳定性评分
    if squat_stability < 0.05:  # 稳定性阈值可调整
        score += 20
    elif squat_stability < 0.1:
        score += 10
    else:
        score += 5
    # 身体平衡评分
    if body_balance < 0.05:  # 平衡阈值可调整
        score += 20
    elif body_balance < 0.1:
        score += 10
    else:
        score += 5
    return score

def generate_situp_stream():
    global situp_count, last_situp_time, last_hip_y, is_situp_down
    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(image)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        if results.pose_landmarks:
            mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

            landmarks = results.pose_landmarks.landmark
            shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value]
            hip = landmarks[mp_pose.PoseLandmark.LEFT_HIP.value]
            knee = landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value]
            ankle = landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value]

            # 计算身体弯曲角度
            body_angle = abs(shoulder.y - hip.y)

            # 计算仰卧起坐的速度（使用臀部的变化速度）
            if last_hip_y is not None:
                speed = abs(hip.y - last_hip_y) / (time.time() - last_situp_time) if last_situp_time else 0
            else:
                speed = 0
            last_hip_y = hip.y
            last_situp_time = time.time()

            # 身体上下移动幅度（例如臀部的移动幅度）
            body_amplitude = abs(hip.y - ankle.y)

            # 稳定性评分（评估稳定性）
            stability = abs(knee.x - ankle.x) + abs(knee.y - ankle.y)

            # 调用评分函数
            score = calculate_situp_score(shoulder, hip, knee, ankle, speed, body_amplitude, stability)

            # 判断是否为合格仰卧起坐
            if score < 40:
                cv2.putText(image, 'Sit-up: Not Qualified', (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            else:
                cv2.putText(image, f'Sit-up: {score} points', (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            # 检测仰卧起坐完成
            if hip.y < knee.y and not is_situp_down:
                situp_count += 1
                is_situp_down = True
            if hip.y > knee.y:
                is_situp_down = False

            # 显示计数
            cv2.putText(image, f'Sit-ups: {situp_count}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # 将图像转换为JPEG流
        ret, jpeg = cv2.imencode('.jpg', image)
        if not ret:
            break
        frame = jpeg.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

    cap.release()


def generate_jumpingjack_stream():
    global jumpingjack_count, last_jump_time, last_ankle_y
    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(image)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        if results.pose_landmarks:
            mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

            landmarks = results.pose_landmarks.landmark
            ankle1 = landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value]
            ankle2 = landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value]
            knee1 = landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value]
            knee2 = landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value]
            wrist1 = landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value]
            wrist2 = landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value]
            hip = landmarks[mp_pose.PoseLandmark.LEFT_HIP.value]

            # 计算速度：脚踝y轴变化的速度
            if last_ankle_y is not None:
                speed = abs(ankle1.y - last_ankle_y) / (time.time() - last_jump_time) if last_jump_time else 0
            else:
                speed = 0
            last_ankle_y = ankle1.y
            last_jump_time = time.time()

            # 身体上下移动幅度
            body_amplitude = abs(hip.y - ankle1.y)
            # 评分
            score = calculate_jump_jack_score(ankle1, ankle2, knee1, knee2, wrist1, wrist2, speed, body_amplitude)

            # 判断是否为合格开合跳
            if score < 80:
                cv2.putText(image, 'Jumping Jack: Not Qualified', (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            else:
                cv2.putText(image, f'Jumping Jack: {score} points', (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            # 检测开合跳完成
            if speed > 1.0:  # 速度阈值可调整
                jumpingjack_count += 1

            # 显示计数
            cv2.putText(image, f'Jumping Jacks: {jumpingjack_count}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # 将图像转换为JPEG流
        ret, jpeg = cv2.imencode('.jpg', image)
        if not ret:
            break
        frame = jpeg.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

    cap.release()



#开合跳
def calculate_jump_jack_score(ankle1, ankle2, knee1, knee2, wrist1, wrist2, speed, body_amplitude):
    score = 0

    # 速度评分
    if 1.5 <= speed <= 1.8:  # 速度范围严格，接近理想的开合跳速度
        score += 30
    elif 1.2 <= speed < 1.5 or 1.8 < speed <= 2.0:
        score += 20
    elif 1.0 <= speed < 1.2 or 2.0 < speed <= 2.3:
        score += 10
    else:
        score += 0  # 非常低或过高的速度减少分数

    # 腿部摆动幅度评分（进一步缩小幅度范围，要求腿部动作更标准）
    ankle_amplitude = euclidean_distance(ankle1, ankle2)
    if 0.4 <= ankle_amplitude <= 0.5:  # 严格的腿部摆动幅度范围
        score += 30
    elif 0.3 <= ankle_amplitude < 0.4 or 0.5 < ankle_amplitude <= 0.55:
        score += 20
    elif 0.2 <= ankle_amplitude < 0.3 or 0.55 < ankle_amplitude <= 0.6:
        score += 10
    else:
        score += 0  # 非标准幅度的腿部动作较低分

    # 手臂摆动幅度评分（严格要求手臂动作更自然）
    arm_amplitude = euclidean_distance(wrist1, wrist2)
    if 0.25 <= arm_amplitude <= 0.35:  # 严格的手臂摆动幅度范围
        score += 20
    elif 0.2 <= arm_amplitude < 0.25 or 0.35 < arm_amplitude <= 0.4:
        score += 10
    elif 0.15 <= arm_amplitude < 0.2 or 0.4 < arm_amplitude <= 0.45:
        score += 5
    else:
        score += 0  # 不自然的手臂动作扣分

    # 身体上下移动幅度评分（要求跳跃动作更加稳定）
    if 0.1 <= body_amplitude <= 0.25:  # 严格的身体上下移动幅度范围
        score += 20
    elif 0.05 <= body_amplitude < 0.1 or 0.25 < body_amplitude <= 0.3:
        score += 10
    elif 0.03 <= body_amplitude < 0.05 or 0.3 < body_amplitude <= 0.35:
        score += 5
    else:
        score += 0  # 不符合标准的上下移动幅度扣分

    return score


# 处理视频流和俯卧撑检测
def generate_video_stream(detection_type):
    global push_up_count, is_up, last_elbow_y, push_up_start_time

    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # 姿态估计
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(image)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        if results.pose_landmarks:
            mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

            landmarks = results.pose_landmarks.landmark
            shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value]
            elbow = landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value]
            hip = landmarks[mp_pose.PoseLandmark.LEFT_HIP.value]
            wrist = landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value]

            # 深度（肘部低于肩膀）
            elbow_y = elbow.y
            shoulder_y = shoulder.y

            # 计算平行性（假设通过肩膀、臀部和膝盖之间的角度来判断）
            body_angle = abs(shoulder_y - hip.y) * 90  # 角度计算简化示例

            # 计算速度：肘部y轴变化的速度
            if last_elbow_y is not None:
                speed = abs(elbow_y - last_elbow_y) / (time.time() - push_up_start_time) if push_up_start_time else 0
            else:
                speed = 0
            last_elbow_y = elbow_y
            push_up_start_time = time.time()

            # 稳定性（肩膀和肘部的稳定性）
            stability = abs(elbow.x - shoulder.x) + abs(elbow.y - shoulder.y)

            # 评分
            score = calculate_push_up_score(elbow_y, shoulder_y, body_angle, speed, stability)

            # 判断是否为合格俯卧撑
            if score < 60:
                cv2.putText(image, 'Push-up: Not Qualified', (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            else:
                cv2.putText(image, f'Push-up: {score} points', (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            # 检测俯卧撑完成
            if elbow_y < shoulder_y - 0.1 and not is_up:
                push_up_count += 1
                is_up = True
            if elbow_y > shoulder_y:
                is_up = False

            # 显示计数
            cv2.putText(image, f'Push-ups: {push_up_count}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # 将图像转换为JPEG流
        ret, jpeg = cv2.imencode('.jpg', image)
        if not ret:
            break
        frame = jpeg.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

    cap.release()


# 处理视频流和跳绳检测
def generate_jumprope_stream():
    global jump_rope_count, last_jump_time, last_ankle_y
    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(image)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        if results.pose_landmarks:
            mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

            landmarks = results.pose_landmarks.landmark
            ankle1 = landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value]
            ankle2 = landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value]
            knee1 = landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value]
            knee2 = landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value]
            wrist1 = landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value]
            wrist2 = landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value]
            hip = landmarks[mp_pose.PoseLandmark.LEFT_HIP.value]

            # 计算速度：脚踝y轴变化的速度
            if last_ankle_y is not None:
                speed = abs(ankle1.y - last_ankle_y) / (time.time() - last_jump_time) if last_jump_time else 0
            else:
                speed = 0
            last_ankle_y = ankle1.y
            last_jump_time = time.time()

            # 身体上下移动幅度
            body_amplitude = abs(hip.y - ankle1.y)
            # 评分
            score = calculate_jump_rope_score(ankle1, ankle2, knee1, knee2, wrist1, wrist2, speed, body_amplitude)

            # 判断是否为合格跳绳
            if score < 80:
                cv2.putText(image, 'Jump Rope: Not Qualified', (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            else:
                cv2.putText(image, f'Jump Rope: {score} points', (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            # 检测跳绳完成
            if speed > 1.0:  # 速度阈值可调整
                jump_rope_count += 1

            # 显示计数
            cv2.putText(image, f'Jump Rope: {jump_rope_count}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # 将图像转换为JPEG流
        ret, jpeg = cv2.imencode('.jpg', image)
        if not ret:
            break
        frame = jpeg.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

    cap.release()


# 处理视频流和蹲起检测
def generate_squat_stream():
    global squat_count, last_squat_time, last_hip_y, is_squat_down
    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(image)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        if results.pose_landmarks:
            mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

            landmarks = results.pose_landmarks.landmark
            hip1 = landmarks[mp_pose.PoseLandmark.LEFT_HIP.value]
            hip2 = landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value]
            knee1 = landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value]
            knee2 = landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value]
            ankle1 = landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value]
            ankle2 = landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value]
            shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value]

            # 深度（臀部低于膝盖）
            # 计算速度：臀部y轴变化的速度
            if last_hip_y is not None:
                speed = abs(hip1.y - last_hip_y) / (time.time() - last_squat_time) if last_squat_time else 0
            else:
                speed = 0
            last_hip_y = hip1.y
            last_squat_time = time.time()

            # 稳定性（臀部、膝盖和脚踝的稳定性）
            squat_stability = (euclidean_distance(hip1, knee1) + euclidean_distance(hip2, knee2) +
                             euclidean_distance(knee1, ankle1) + euclidean_distance(knee2, ankle2)) / 4
            # 身体平衡
            body_balance = abs(shoulder.x - ((hip1.x + hip2.x) / 2))

            # 评分
            score = calculate_squat_score(hip1, hip2, knee1, knee2, ankle1, ankle2, speed, squat_stability, body_balance)

            # 判断是否为合格蹲起
            if score < 60:
                cv2.putText(image, 'Squat: Not Qualified', (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            else:
                cv2.putText(image, f'Squat: {score} points', (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            # 检测蹲起完成
            if hip1.y > knee1.y and not is_squat_down:
                squat_count += 1
                is_squat_down = True
            if hip1.y < knee1.y:
                is_squat_down = False

            # 显示计数
            cv2.putText(image, f'Squats: {squat_count}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # 将图像转换为JPEG流
        ret, jpeg = cv2.imencode('.jpg', image)
        if not ret:
            break
        frame = jpeg.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

    cap.release()


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/pushup')
def pushup():
    return render_template('pushup.html')


@app.route('/jumprope')
def jumprope():
    return render_template('jump_rope.html')


@app.route('/squat')
def squat():
    return render_template('squat.html')

@app.route('/situp')
def situp():
    return render_template('situp.html')

@app.route('/jumpingjack')
def jumpingjack():
    return render_template('jumpingjack.html')


@app.route('/video_feed/<detection_type>')
def video_feed(detection_type):
    if detection_type == 'pushup':
        return Response(generate_video_stream(detection_type),
                       mimetype='multipart/x-mixed-replace; boundary=frame')
    elif detection_type == 'jumprope':
        return Response(generate_jumprope_stream(),
                       mimetype='multipart/x-mixed-replace; boundary=frame')
    elif detection_type == 'squat':
        return Response(generate_squat_stream(),
                       mimetype='multipart/x-mixed-replace; boundary=frame')
    elif detection_type == 'situp':
        return Response(generate_situp_stream(),
                        mimetype='multipart/x-mixed-replace; boundary=frame')
    elif detection_type == 'jumpingjack':
        return Response(generate_jumpingjack_stream(),
                        mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == '__main__':
    app.run(debug=True)