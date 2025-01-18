import math
import mediapipe as mp

mp_pose = mp.solutions.pose

def calculate_jumprope_score(results):
    """
    计算跳绳得分，基于双脚的跳跃动作和全身协调性。
    """
    # 关键点
    left_ankle = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_ANKLE]
    right_ankle = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_ANKLE]
    left_knee = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_KNEE]
    right_knee = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_KNEE]
    left_hip = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_HIP]
    right_hip = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_HIP]
    left_shoulder = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER]
    right_shoulder = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER]

    # 计算双脚的相对高度（通过脚踝的y坐标）
    left_ankle_height = left_ankle.y
    right_ankle_height = right_ankle.y
    feet_height = (left_ankle_height + right_ankle_height) / 2

    # 计算膝盖角度（通过膝盖与臀部和脚踝的角度来判断蹬地力度）
    def calculate_angle(a, b, c):
        """计算三个点形成的角度"""
        a_x, a_y = a.x, a.y
        b_x, b_y = b.x, b.y
        c_x, c_y = c.x, c.y
        ab_x, ab_y = a_x - b_x, a_y - b_y
        bc_x, bc_y = c_x - b_x, c_y - b_y
        dot = ab_x * bc_x + ab_y * bc_y
        ab_len = math.sqrt(ab_x ** 2 + ab_y ** 2)
        bc_len = math.sqrt(bc_x ** 2 + bc_y ** 2)
        cos_theta = dot / (ab_len * bc_len)
        angle = math.acos(cos_theta)
        return math.degrees(angle)

    left_knee_angle = calculate_angle(left_hip, left_knee, left_ankle)
    right_knee_angle = calculate_angle(right_hip, right_knee, right_ankle)

    # 评分逻辑
    score = 0

    # 1. 根据脚踝高度评分：脚踝上升的幅度较大时得分更高
    if feet_height < 0.4:
        score += 20  # 理想的跳跃高度
    elif feet_height < 0.5:
        score += 15  # 略低的跳跃
    else:
        score += 10  # 跳跃高度较低

    # 2. 根据膝盖角度评分：膝盖角度接近标准值（约80°-100°）时分数更高
    if 80 <= left_knee_angle <= 100 and 80 <= right_knee_angle <= 100:
        score += 25  # 理想的膝盖角度
    elif 70 <= left_knee_angle <= 110 and 70 <= right_knee_angle <= 110:
        score += 20  # 略微偏差
    else:
        score += 10  # 角度偏差较大时得分较低

    # 3. 根据动作的频率和协调性评分
    # 假设跳绳的动作协调性较好时，身体的上下运动较为一致
    torso_movement = (left_hip.y + right_hip.y) / 2  # 计算臀部高度的变化
    if abs(torso_movement - 0.4) < 0.05:  # 假设0.4是标准的高度
        score += 20  # 动作协调，稳定
    else:
        score += 15  # 略有偏差

    # 4. 根据运动幅度判断动作完整性
    if abs(feet_height - 0.3) < 0.05:
        score += 20  # 完整的跳跃动作
    else:
        score += 10  # 动作不够完整时分数较低

    # 最终评分（范围 0 至 100）
    final_score = min(100, score)  # 确保分数不超过 100
    return final_score
