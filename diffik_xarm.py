import mujoco
import mujoco.viewer
import numpy as np
import time

SCOOPING_TRAJ = np.load("mod_traj_1.npz")["arr_0"]

# Integration timestep in seconds. This corresponds to the amount of time the joint
# velocities will be integrated for to obtain the desired joint positions.
integration_dt: float = 1.0

# Damping term for the pseudoinverse. This is used to prevent joint velocities from
# becoming too large when the Jacobian is close to singular.
damping: float = 1e-4

# Whether to enable gravity compensation.
gravity_compensation: bool = True

# Simulation timestep in seconds.
dt: float = 0.002

# Maximum allowable joint velocity in rad/s. Set to 0 to disable.
max_angvel = 0.0

eef_pose = []


def main() -> None:
    assert mujoco.__version__ >= "3.1.0", "Please upgrade to mujoco 3.1.0 or later."

    # Load the model and data.
    model = mujoco.MjModel.from_xml_path("ufactory_xarm6/scene.xml")
    data = mujoco.MjData(model)

    # Override the simulation timestep.
    model.opt.timestep = dt

    # End-effector site we wish to control, in this case a site attached to the last
    # link (wrist_3_link) of the robot.
    site_id = model.site("link_tcp").id

    # Name of bodies we wish to apply gravity compensation to.
    body_names = [
        "link1",
        "link2",
        "link3",
        "link4",
        "link5",
        "link6",
        # "link7",
    ]
    body_ids = [model.body(name).id for name in body_names]
    if gravity_compensation:
        model.body_gravcomp[body_ids] = 1.0

    # Get the dof and actuator ids for the joints we wish to control.
    joint_names = [
        "joint1",
        "joint2",
        "joint3",
        "joint4",
        "joint5",
        "joint6",
        # "joint7",
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
    # Note that actuator names are the same as joint names in this case.
    actuator_ids = np.array([model.actuator(name).id for name in act_names])

    # Initial joint configuration saved as a keyframe in the XML file.
    key_id = model.key("home").id

    # Mocap body we will control with our mouse.
    mocap_id = model.body("target").mocapid[0]

    # Pre-allocate numpy arrays.
    jac = np.zeros((6, model.nv))
    diag = damping * np.eye(6)
    error = np.zeros(6)
    error_pos = error[:3]
    error_ori = error[3:]
    site_quat = np.zeros(4)
    site_quat_conj = np.zeros(4)
    error_quat = np.zeros(4)

    # Define a trajectory for the end-effector site to follow.
    def circle(t: float, r: float, h: float, k: float, f: float) -> np.ndarray:
        """Return the (x, y) coordinates of a circle with radius r centered at (h, k)
        as a function of time t and frequency f."""
        x = r * np.cos(2 * np.pi * f * t) + h
        y = r * np.sin(2 * np.pi * f * t) + k
        return np.array([x, y])
    
    def scooping_traj(t):
        idx = int(t / 0.01)
        if idx > len(SCOOPING_TRAJ) - 1:
            return SCOOPING_TRAJ[-1][:3], SCOOPING_TRAJ[-1][3:]
        return SCOOPING_TRAJ[idx][:3], SCOOPING_TRAJ[idx][3:]


    with mujoco.viewer.launch_passive(
        model=model, data=data, show_left_ui=False, show_right_ui=False
    ) as viewer:
        # Reset the simulation to the initial keyframe.
        mujoco.mj_resetDataKeyframe(model, data, key_id)

        # Initialize the camera view to that of the free camera.
        mujoco.mjv_defaultFreeCamera(model, viewer.cam)

        # Toggle site frame visualization.
        viewer.opt.frame = mujoco.mjtFrame.mjFRAME_SITE

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
            

            # Position error.
            error_pos[:] = data.mocap_pos[mocap_id] - data.site(site_id).xpos

            # Orientation error.
            mujoco.mju_mat2Quat(site_quat, data.site(site_id).xmat)
            mujoco.mju_negQuat(site_quat_conj, site_quat)
            mujoco.mju_mulQuat(error_quat, data.mocap_quat[mocap_id], site_quat_conj)
            mujoco.mju_quat2Vel(error_ori, error_quat, 1.0)
            print("des pos:", data.site(site_id).xpos)
            print("des quat:", data.site(site_id).xmat)
            print("error_quat:", error_quat)
            print("error_ori:", error_ori)
            print("diag:", diag)


            # Get the Jacobian with respect to the end-effector site.
            mujoco.mj_jacSite(model, data, jac[:3], jac[3:], site_id)

            # Solve system of equations: J @ dq = error.
            dq = jac.T @ np.linalg.solve(jac @ jac.T + diag, error)

            # Scale down joint velocities if they exceed maximum.
            if max_angvel > 0:
                dq_abs_max = np.abs(dq).max()
                if dq_abs_max > max_angvel:
                    dq *= max_angvel / dq_abs_max

            # Integrate joint velocities to obtain joint positions.
            q = data.qpos.copy()
            mujoco.mj_integratePos(model, q, dq, integration_dt)

            # Set the control signal.
            np.clip(q, *model.jnt_range.T, out=q)
            print("torques:", q[dof_ids])
            data.ctrl[actuator_ids] = q[dof_ids]

            # Step the simulation.
            mujoco.mj_step(model, data)
            # print("qpos:", data.qpos[dof_ids])

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