import mujoco
import mujoco.viewer
import numpy as np
import time

SCOOPING_TRAJ = np.load("mod_traj_1.npz")["arr_0"]


# Integration timestep in seconds. This corresponds to the amount of time the joint
# velocities will be integrated for to obtain the desired joint positions.
integration_dt: float = 0.1

# Damping term for the pseudoinverse. This is used to prevent joint velocities from
# becoming too large when the Jacobian is close to singular.
damping: float = 1e-4

# Gains for the twist computation. These should be between 0 and 1. 0 means no
# movement, 1 means move the end-effector to the target in one integration step.
Kpos: float = 0.95
Kori: float = 0.95

# Whether to enable gravity compensation.
gravity_compensation: bool = True

# Simulation timestep in seconds.
dt: float = 0.002

# Nullspace P gain.
Kn = np.asarray([10.0, 10.0, 10.0, 5.0, 5.0, 5.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

# Maximum allowable joint velocity in rad/s.
max_angvel = 0.785

eef_pose = []

def main() -> None:
    assert mujoco.__version__ >= "3.1.0", "Please upgrade to mujoco 3.1.0 or later."

    # Load the model and data.
    model = mujoco.MjModel.from_xml_path("ufactory_xarm6/scene.xml")
    data = mujoco.MjData(model)

    # Enable gravity compensation. Set to 0.0 to disable.
    model.body_gravcomp[:] = float(gravity_compensation)
    model.opt.timestep = dt

    # End-effector site we wish to control.
    site_name = "link_tcp"
    site_id = model.site(site_name).id

    # Get the dof and actuator ids for the joints we wish to control. These are copied
    # from the XML file. Feel free to comment out some joints to see the effect on
    # the controller.
    joint_names = [
        "joint1",
        "joint2",
        "joint3",
        "joint4",
        "joint5",
        "joint6",
        # "joint7",
        # "left_driver_joint",
        "drive_joint",
        "left_finger_joint",
        "left_inner_knuckle_joint",
        # "right_driver_joint",
        "right_outer_knuckle_joint",
        "right_finger_joint",
        "right_inner_knuckle_joint",
    ]

    act_names = [
        "act1",
        "act2",
        "act3",
        "act4",
        "act5",
        "act6",
        # "act7",
    ]

    dof_ids = np.array([model.joint(name).id for name in joint_names])
    actuator_ids = np.array([model.actuator(name).id for name in act_names])

    # Initial joint configuration saved as a keyframe in the XML file.
    key_name = "home"
    key_id = model.key(key_name).id
    q0 = model.key(key_name).qpos

    # Mocap body we will control with our mouse.
    mocap_name = "target"
    mocap_id = model.body(mocap_name).mocapid[0]

    # Pre-allocate numpy arrays.
    jac = np.zeros((6, model.nv))
    diag = damping * np.eye(6)
    eye = np.eye(model.nv)
    twist = np.zeros(6)
    site_quat = np.zeros(4)
    site_quat_conj = np.zeros(4)
    error_quat = np.zeros(4)

    def scooping_traj(t):
        idx = int(t / 0.01)
        if idx > len(SCOOPING_TRAJ) - 1:
            return SCOOPING_TRAJ[-1][:3], SCOOPING_TRAJ[-1][3:]
        return SCOOPING_TRAJ[idx][:3], SCOOPING_TRAJ[idx][3:]

    with mujoco.viewer.launch_passive(
        model=model,
        data=data,
        show_left_ui=False,
        show_right_ui=False,
    ) as viewer:
        # Reset the simulation.
        mujoco.mj_resetDataKeyframe(model, data, key_id)

        # Reset the free camera.
        mujoco.mjv_defaultFreeCamera(model, viewer.cam)

        # Enable site frame visualization.
        viewer.opt.frame = mujoco.mjtFrame.mjFRAME_SITE

        # Set to initial pose
        data.qpos[:6] = [ 1.18982599, -0.04728632, -0.97072353,  2.1839953,  -1.35812967, -1.34759178]

        mujoco.mj_step(model, data)
        viewer.sync()

        while viewer.is_running():
            step_start = time.time()

            # Set the target position of the end-effector site.
            # data.mocap_pos[mocap_id, 0:2] = circle(data.time, 0.1, 0.5, 0.0, 0.5)
            pos, orn = scooping_traj(data.time)
            data.mocap_pos[mocap_id] = pos
            data.mocap_quat[mocap_id] = orn
            curr_eef_pos = data.site(site_id).xpos
            curr_eef_quat = np.zeros(4)
            mujoco.mju_mat2Quat(curr_eef_quat, data.site(site_id).xmat)
            eef_pose.append(np.concatenate([curr_eef_pos, curr_eef_quat]))

            # Spatial velocity (aka twist).
            dx = data.mocap_pos[mocap_id] - data.site(site_id).xpos
            twist[:3] = Kpos * dx / integration_dt
            mujoco.mju_mat2Quat(site_quat, data.site(site_id).xmat)
            mujoco.mju_negQuat(site_quat_conj, site_quat)
            mujoco.mju_mulQuat(error_quat, data.mocap_quat[mocap_id], site_quat_conj)
            mujoco.mju_quat2Vel(twist[3:], error_quat, 1.0)
            twist[3:] *= Kori / integration_dt

            # Jacobian.
            mujoco.mj_jacSite(model, data, jac[:3], jac[3:], site_id)

            # Damped least squares.
            dq = jac.T @ np.linalg.solve(jac @ jac.T + diag, twist)

            # Nullspace control biasing joint velocities towards the home configuration.
            # print("q0:", q0.shape)
            # print("data.qpos[dof_ids]:", data.qpos[dof_ids].shape)
            # print("Kn:", Kn.shape)
            # print("jac:", jac.shape)
            # print("eye:", eye.shape)
            # print("data.qvel[dof_ids]:", data.qvel[dof_ids].shape)
            dq += (eye - np.linalg.pinv(jac) @ jac) @ (Kn * (q0 - data.qpos[dof_ids]))

            # Clamp maximum joint velocity.
            dq_abs_max = np.abs(dq).max()
            if dq_abs_max > max_angvel:
                dq *= max_angvel / dq_abs_max

            # Integrate joint velocities to obtain joint positions.
            q = data.qpos.copy()  # Note the copy here is important.
            mujoco.mj_integratePos(model, q, dq, integration_dt)
            np.clip(q, *model.jnt_range.T, out=q)

            # Set the control signal and step the simulation.
            data.ctrl[actuator_ids] = q[actuator_ids]
            mujoco.mj_step(model, data)

            viewer.sync()
            time_until_next_step = dt - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)


if __name__ == "__main__":
    main()

    # Plot the end-effector trajectory.
    eef_pose = np.array(eef_pose)
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D

    fig, ax = plt.subplots(1, 1)
    ax = fig.add_subplot(111, projection='3d')

    ax.plot(SCOOPING_TRAJ[:, 0], SCOOPING_TRAJ[:, 1], SCOOPING_TRAJ[:, 2], label="Desired", color="red")
    ax.plot(eef_pose[:, 0], eef_pose[:, 1], eef_pose[:, 2], label="Actual", alpha=0.8)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.legend()
    plt.show()
