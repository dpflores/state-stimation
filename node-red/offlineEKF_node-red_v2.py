import sys
#import pickle
import numpy as np
import matplotlib.pyplot as plt
#from mpl_toolkits.mplot3d import Axes3D
#from rotations import angle_normalize, rpy_jacobian_axis_angle, skew_symmetric, Quaternion

# msg = sys.stdin.readline()
# data = msg.split(",")
# for i in range(len(data)):
#     data[i] = float(data[i])
#print(data[0:2])

#### 1. Data ###################################################################################

################################################################################################
# This is where you will load the data from the pickle files. For parts 1 and 2, you will use
# p1_data.pkl. For Part 3, you will use pt3_data.pkl.
################################################################################################
data = np.genfromtxt('/home/del/Del/PERSONAL/AV/state-stimation/data/data_ekf02.txt', delimiter=',')

# Crear los objetos gt, imu_f y gnss
hall = {'vel': data[:,1],  '_t': data[:,0]}
imu_yaw = {'data': data[:,2], 't': data[:,0]}
gnss = {'data': data[:,3:5], 't': data[:,0]}

print(gnss['data'][:,0])
################################################################################################
# Each element of the data dictionary is stored as an item from the data dictionary, which we
# will store in local variables, described by the following:
#   gt: Data object containing ground truth. with the following fields:
#     a: Acceleration of the vehicle, in the inertial frame
#     v: Velocity of the vehicle, in the inertial frame
#     p: Position of the vehicle, in the inertial frame
#     alpha: Rotational acceleration of the vehicle, in the inertial frame
#     w: Rotational velocity of the vehicle, in the inertial frame
#     r: Rotational position of the vehicle, in Euler (XYZ) angles in the inertial frame
#     _t: Timestamp in ms.
#   imu_f: StampedData object with the imu specific force data (given in vehicle frame).
#     data: The actual data
#     t: Timestamps in ms.
#   imu_w: StampedData object with the imu rotational velocity (given in the vehicle frame).
#     data: The actual data
#     t: Timestamps in ms.
#   gnss: StampedData object with the GNSS data.
#     data: The actual data
#     t: Timestamps in ms.
#   lidar: StampedData object with the LIDAR data (positions only).
#     data: The actual data
#     t: Timestamps in ms.
################################################################################################
# n is the number of samples
#print("GNSS")
#print(gnss["data"])

#print("IMU forces")
#print(imu_f["data"])

#### 2. Constants ##############################################################################

################################################################################################
# Now that our data is set up, we can start getting things ready for our solver. One of the
# most important aspects of a filter is setting the estimated sensor variances correctly.
# We set the values here.
################################################################################################
var_speed = 0.14
var_yaw = 0.1
var_gnss  = 0.2

#### 3. Initial Values #########################################################################

################################################################################################
# Let's set up some initial values for our ES-EKF solver.
################################################################################################
p_est = np.zeros([imu_yaw["data"].shape[0], 2])  # position estimates
p_cov = np.zeros([imu_yaw["data"].shape[0], 2, 2])  # covariance matrices at each timestep

hall_pos = np.zeros([imu_yaw["data"].shape[0], 2])
# Set initial values.
p_est[0] = np.zeros(2)
p_cov[0] = np.zeros(2)  # covariance of estimate
gnss_i  = 0

#### 4. Measurement Update #####################################################################

################################################################################################
# Since we'll need a measurement update for both the GNSS and the LIDAR data, let's make
# a function for it.
################################################################################################
def measurement_update(sensor_var, p_cov_check, y_k, p_check):
    # 3.1 Compute Kalman Gain
    H = np.eye(2)
    I = np.identity(2)
    R = I * sensor_var
    K = p_cov_check @ H.T @ np.linalg.inv(H @ p_cov_check @ H.T + R)

    # 3.2 Compute error state
    error = K @ (y_k - p_check)
    p_delta = error[:2]

    # 3.3 Correct predicted state
    p_hat = p_check + p_delta

    # 3.4 Compute corrected covariance
    p_cov_hat = (np.eye(2) - K @ H) @ p_cov_check

    return p_hat, p_cov_hat

def rotate_vector(vector, theta):
    # Convertir el ángulo a radianes
    theta_rad = np.radians(theta)
    
    # Matriz de rotación en sentido antihorario
    rotation_matrix = np.array([[np.cos(theta_rad), -np.sin(theta_rad)],
                                [np.sin(theta_rad), np.cos(theta_rad)]])
    
    # Aplicar la rotación al vector
    rotated_vector = np.dot(rotation_matrix, vector)
    return rotated_vector

def normalize_angle(angle):
        """
        Normalize an angle to [-pi, pi].
        :param angle: (float)
        :return: (float) Angle in radian in [-pi, pi]
        """
        while angle > np.pi:
            angle -= 2.0 * np.pi

        while angle < -np.pi:
            angle += 2.0 * np.pi

        return angle
    
    
#### 5. Main Filter Loop #######################################################################

################################################################################################
# Now that everything is set up, we can start taking in the sensor data and creating estimates
# for our state in a loop.
################################################################################################
count = 0
for k in range(1, imu_yaw["data"].shape[0]):  # start at 1 b/c we have initial prediction from gt
    delta_t = imu_yaw["t"][k] - imu_yaw["t"][k - 1]
    delta_t = 0.2


    yaw = -imu_yaw["data"][k]
    # yaw = normalize_angle(yaw)
    vel = hall["vel"][k]

    f_v = np.array([np.cos(yaw)*delta_t,
                    np.sin(yaw)*delta_t])
    # print(hall["vel"][k-1])
    p_est[k] = p_est[k-1] + f_v*vel
    hall_pos[k] = hall_pos[k-1] + f_v*vel
    #print(p_est[k])
    # 1.1 Linearize the motion model and compute Jacobians
    F = np.eye(2)
    L = np.array([[np.cos(yaw)*delta_t, np.sin(yaw)*delta_t],
                    [-vel*delta_t*np.sin(yaw), vel*delta_t*np.cos(yaw)]])
    # L = np.eye(2)
    Q = np.diag([var_speed, var_yaw])


    # 2. Propagate uncertainty
    p_cov[k] = F @ p_cov[k-1] @ F.T + L @ Q @ L.T
    
    # 3. Check availability of GNSS measurements
    if not np.isnan(gnss["data"][k][0]): 
       
        gps_data = gnss["data"][k]
        # theta = -17
        # gps_data = rotate_vector(gps_data, theta)
        p_est[k], p_cov[k] = measurement_update(var_gnss, p_cov[k], gps_data, p_est[k])


last = imu_yaw["data"].shape[0]-1

data2sent = str(p_est[last][0]) + "," + str(p_est[last][1]) #+ "," + str(gt["p"][last,0]) + "," + str(gt["p"][last,1]) 
print(data2sent)

# Crear la figura y los ejes
est_traj_fig = plt.figure()
ax = est_traj_fig.add_subplot(111)

# Graficar las variables en el plano XY
ax.plot(p_est[:, 0], p_est[:, 1], label='EKF')
a = gnss["data"][:,0]
b = gnss["data"][:,1]
gps_x =a[~np.isnan(a)]
gps_y = b[~np.isnan(b)]
ax.plot(gps_x, gps_y, label='GPS')
ax.plot(hall_pos[:,0], hall_pos[:,1], label='Odom')
# Configurar etiquetas y título del gráfico
ax.set_xlabel('Easting [m]')
ax.set_ylabel('Northing [m]')
ax.set_title('Ground Truth and Estimated Trajectory')

# Establecer límites y marcas de los ejes
#ax.set_xlim(-400, 400)
##ax.set_ylim(-800, 200)
#ax.set_xticks([-600, -400, -200, 0, 200, 400, 600])
#ax.set_yticks([-600, -400, -200, 0, 200, 400, 600])

# Agregar leyenda y mostrar el gráfico
ax.legend()
plt.show()