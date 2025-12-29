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

