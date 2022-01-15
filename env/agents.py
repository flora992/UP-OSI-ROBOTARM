import gym
import numpy as np
import pybullet
import pybullet_data


class IIWA:
    def __init__(self, bc):
        self._p = bc

        self.load_robot_model()
        self.action_dim = len(self.joint_ids)

        high = np.ones(self.action_dim, dtype=np.float32)
        self.action_space = gym.spaces.Box(-high, high, dtype=np.float32)

        self.base_joint_angles = np.zeros(self.action_dim, np.float32)
        self.base_joint_speeds = np.zeros(self.action_dim, np.float32)

        self.joint_angles, self.joint_speeds = np.zeros(
            (2, self.action_dim), dtype=np.float32
        )

    def apply_action(self, action):

        forces = self.joint_gains * action

        pybullet.setJointTorqueArray(
            bodyUniqueId=self.id,
            jointIndices=self.joint_uindices,
            forces=forces,
            physicsClientId=self._p._client,
        )

    def load_robot_model(self):
        bc = self._p

        self._p.setAdditionalSearchPath(pybullet_data.getDataPath())
        self.id = self._p.loadURDF(
            "kuka_iiwa/model.urdf",
            (0, 0, 0),
            (0, 0, 0, 1),
            useFixedBase=True,
        )

        num_joints = bc.getNumJoints(self.id)
        self.joint_ids = list(range(num_joints))
        self.end_effector_id = self.joint_ids[-1]

        uindices = [bc.getJointInfo(self.id, pid)[4] for pid in range(num_joints)]
        self.joint_uindices = np.fromiter(uindices, dtype=np.int32)

        gains = [bc.getJointInfo(self.id, pid)[10] for pid in range(num_joints)]
        self.joint_gains = np.fromiter(gains, dtype=np.float32)

        limits = [bc.getJointInfo(self.id, pid)[8:10] for pid in range(num_joints)]
        self.joint_limits = np.array(limits, dtype=np.float64)

        self._zeros = [0 for _ in self.joint_ids]
        self._gains = [0.1 for _ in self.joint_ids]

    def reset(self, random_pose=True):

        joint_angles = self.base_joint_angles.copy()

        if random_pose:
            low, high = self.joint_limits.T
            joint_angles = self.np_random.uniform(low, high).astype("f4")

        self.reset_joint_states(joint_angles, self.base_joint_speeds)
        self.calc_state()

    def calc_state(self):
        pybullet.getJointStates2(
            self.id,
            self.joint_ids,
            self.joint_angles,
            self.joint_speeds,
            physicsClientId=self._p._client,
        )

        self.end_effector_xyz = pybullet.getLinkState(
            self.id,
            self.end_effector_id,
            computeLinkVelocity=0,
            computeForwardKinematics=0,
            physicsClientId=self._p._client,
        )[0]

    def reset_joint_states(self, positions, velocities):
        pybullet.resetJointStates(
            self.id,
            self.joint_ids,
            targetValues=positions,
            targetVelocities=velocities,
            physicsClientId=self._p._client,
        )

        pybullet.setJointMotorControlArray(
            bodyIndex=self.id,
            jointIndices=self.joint_ids,
            controlMode=pybullet.POSITION_CONTROL,
            targetPositions=self._zeros,
            targetVelocities=self._zeros,
            positionGains=self._gains,
            velocityGains=self._gains,
            forces=self._zeros,
            physicsClientId=self._p._client,
        )

    def initialize(self):
        pass
