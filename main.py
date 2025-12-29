def _load_model():
    from img2pose import img2poseModel
    from model_loader import load_model

    DEPTH = 18
    MAX_SIZE = 1400
    MIN_SIZE = 300

    POSE_MEAN = "./models/WIDER_train_pose_mean_v1.npy"
    POSE_STDDEV = "./models/WIDER_train_pose_stddev_v1.npy"
    MODEL_PATH = "./models/img2pose_v1.pth"

    pose_mean = np.load(POSE_MEAN)
    pose_stddev = np.load(POSE_STDDEV)

    threed_points = np.load('./img2pose/pose_references/reference_3d_68_points_trans.npy')

    img2pose_model = img2poseModel(
        DEPTH, MIN_SIZE, MAX_SIZE, 
        pose_mean=pose_mean, pose_stddev=pose_stddev,
        threed_68_points=threed_points,
    )

    print("CPU mode:", str(img2pose_model.device) == "cpu")

    load_model(img2pose_model.fpn_model, MODEL_PATH, cpu_mode=str(img2pose_model.device) == "cpu", model_only=True)
    img2pose_model.evaluate()
    return img2pose_model

def _img2pose_rvec_to_pyr_deg(rvec_xyz) -> "tuple[float, float, float]":
        """
        rvec_xyz: (3,) Rodrigues/rotvec from img2pose
        returns: (pitch, yaw, roll) in degrees (AFLW-style)
        """
        Rm = R.from_rotvec(rvec_xyz).as_matrix()
        e = R.from_matrix(Rm.T).as_euler("xyz", degrees=True)  # x,y,z
        pitch = e[0]
        yaw   = -e[1]
        roll  = -e[2]
        return pitch, yaw, roll

def retrieve_face_angles(model, image: "np.ndarray", threshold: "float" = 0.9) -> "list[tuple[float, float, float]]":
    """Retrieve face angles and bounding boxes from an image using img2pose model."""

    res = model.predict([transform(image)])[0]

    all_bboxes = res["boxes"].cpu().numpy().astype("float")

    angles = []
    bboxes = []

    dofs = res["dofs"].cpu().numpy()

    for i in range(len(all_bboxes)):
        if res["scores"][i] > threshold:
            pose_pred = dofs[i].astype("float")
            pose_pred = pose_pred.squeeze()
            angles.append(_img2pose_rvec_to_pyr_deg(pose_pred[:3]))
            bboxes.append(all_bboxes[i])

    return angles, bboxes

def print_angles(image: "np.ndarray", angles: "list[tuple[float, float, float]]", bboxes: "list[np.ndarray]"):
    for i, bbox in enumerate(bboxes):
        x1, y1, x2, y2 = bbox.astype("int")
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 0, 255), 2)

        #draw text for angles pitch, yaw, roll
        cv2.putText(image, f"Pitch: {angles[i][0]:.1f}", (x1, y1 - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        cv2.putText(image, f"Yaw: {angles[i][1]:.1f}", (x1, y1 - 20),   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        cv2.putText(image, f"Roll: {angles[i][2]:.1f}", (x1, y1),       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
