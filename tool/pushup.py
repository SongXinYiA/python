import math
import mediapipe as mp

mp_pose = mp.solutions.pose

def calculate_pushup_score(results):
    """
    计算俯卧撑的得分，基于躯干上下移动、肘部角度和动作完整性。
    """
    # 关键点
    left_shoulder = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER]
    right_shoulder = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER]
    left_elbow = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_ELBOW]
    right_elbow = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_ELBOW]
    left_wrist = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_WRIST]
    right_wrist = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_WRIST]
    nose = results.pose_landmarks.landmark[mp_pose.PoseLandmark.NOSE]
    left_hip = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_HIP]
    right_hip = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_HIP]

    # 计算躯干上下移动（通过鼻子和臀部）
    torso_movement = (nose.y + left_hip.y + right_hip.y) / 3

    # 计算肘部角度（计算方法：acos公式）
    def calculate_angle(a, b, c):
        """计算三个点形成的角度"""
        # 点 a, b, c 的坐标
        a_x, a_y = a.x, a.y
        b_x, b_y = b.x, b.y
        c_x, c_y = c.x, c.y
        # 计算向量
        ab_x, ab_y = a_x - b_x, a_y - b_y
        bc_x, bc_y = c_x - b_x, c_y - b_y
        # 计算向量点积
        dot = ab_x * bc_x + ab_y * bc_y
        # 计算向量的模
        ab_len = math.sqrt(ab_x ** 2 + ab_y ** 2)
        bc_len = math.sqrt(bc_x ** 2 + bc_y ** 2)
        # 计算角度
        cos_theta = dot / (ab_len * bc_len)
        angle = math.acos(cos_theta)
        return math.degrees(angle)

    # 计算肘部角度
    left_elbow_angle = calculate_angle(left_shoulder, left_elbow, left_wrist)
    right_elbow_angle = calculate_angle(right_shoulder, right_elbow, right_wrist)

    # 评分逻辑：
    score = 0

    # 1. 根据肘部角度评分（接近90°时分数更高）
    if left_elbow_angle >= 90 and right_elbow_angle >= 90:
        score += 25  # 理想的俯卧撑肘部角度
    elif left_elbow_angle >= 80 and right_elbow_angle >= 80:
        score += 20  # 略微弯曲的肘部角度
    elif left_elbow_angle >= 70 and right_elbow_angle >= 70:
        score += 15  # 有些弯曲
    else:
        score += 10  # 肘部角度较大时得分较低

    # 2. 根据躯干上下移动评分
    # 假设躯干的上下移动在一个合理范围内，如 y 位置差在 0.2 至 0.4 之间为合适的动作范围
    torso_movement_score = max(0, min(10, (0.4 - abs(torso_movement - 0.3)) * 25))  # 通过调整计算，确保评分合理
    score += torso_movement_score

    # 3. 根据动作完整性评分
    # 判断躯干上升和下降的完整性，如果上下移动幅度较大且相对标准，则得分较高
    if abs(torso_movement - 0.3) < 0.1:
        score += 20  # 表示运动幅度合理
    else:
        score += 10  # 不符合标准则得分较低

    # 最终评分（范围 0 至 100）
    final_score = min(100, score)  # 确保分数不超过 100
    return final_score
