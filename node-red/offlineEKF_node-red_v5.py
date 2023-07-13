import sys
#import pickle
import numpy as np
import matplotlib.pyplot as plt
import math

def make2Plot(map, BBox, x, y, x1, y1, l1, l2, lw=2):
  fig, ax = plt.subplots(figsize = (10,7))
  ax.plot(x, y, linewidth=lw, c='b')
  ax.plot(x1, y1, linewidth=lw, c='g')
  #ax.set_title('Localización: Entorno de pruebas')
  ax.set_xlim(BBox[0],BBox[1])
  ax.set_ylim(BBox[2],BBox[3])
  plt.axis('off')
  ax.imshow(map, zorder=0, extent = BBox, aspect= 'equal')
  plt.legend([l1, l2])
  plt.show()
  save =1 

def convertir_xy_a_gps(latitud_origen, longitud_origen, x, y):
    # Constantes necesarias para la conversión
    a = 6378137.0  # Semieje mayor de la elipsoide (WGS84)
    f = 1 / 298.257223563  # Achatamiento de la elipsoide (WGS84)

    # Cálculo de las coordenadas de latitud y longitud
    latitud_origen_rad = math.radians(latitud_origen)
    longitud_origen_rad = math.radians(longitud_origen)
    rho_origen = a * (1 - f) / math.pow(1 - f * math.sin(latitud_origen_rad) ** 2, 1.5)
    latitud = latitud_origen_rad + y / rho_origen
    n = a / math.sqrt(1 - f * math.sin(latitud) ** 2)
    longitud = longitud_origen_rad + x / (n * math.cos(latitud))

    # Convertir a grados decimales
    latitud_deg = math.degrees(latitud)
    longitud_deg = math.degrees(longitud)
    
    return latitud_deg, longitud_deg

def convertir_gps_a_xy(latitud, longitud, latitud_origen, longitud_origen):
    # Constantes necesarias para la conversión
    a = 6378137.0  # Semieje mayor de la elipsoide (WGS84)
    f = 1 / 298.257223563  # Achatamiento de la elipsoide (WGS84)
    latitud_rad = math.radians(latitud)
    longitud_rad = math.radians(longitud)

    # Cálculo de las coordenadas X-Y
    e2 = f * (2 - f)
    n = a / math.sqrt(1 - e2 * math.sin(latitud_rad) ** 2)
    rho = a * (1 - e2) / math.pow(1 - e2 * math.sin(latitud_rad) ** 2, 1.5)
    rho_origen = a * (1 - e2) / math.pow(1 - e2 * math.sin(math.radians(latitud_origen)) ** 2, 1.5)
    x = (n * math.cos(latitud_rad) * (longitud_rad - math.radians(longitud_origen)))
    y = rho * (latitud_rad - math.radians(latitud_origen))
    
    return x, y

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
data = np.genfromtxt('./../data/data_ekf07.txt', delimiter=',')

# Crear los objetos gt, imu_f y gnss
hall = {'vel': data[:,1],  '_t': data[:,0]}
imu_yaw = {'data': data[:,2], 't': data[:,0]}
pos_xy = {'data': data[:,3:5], 't': data[:,0]}
gnss = {'data': data[:,5:7], 't': data[:,0]}
utm = {'data': np.zeros((len(gnss['data']),2)), 't': data[:,0]}

for i in range(len(gnss['data'])):
    # if(np.isnan(pos_xy['data'][i,0])):
    #     utm['data'][i] = np.nan
    # else:
    #     
    utm['data'][i] = convertir_gps_a_xy(gnss['data'][i,0], gnss['data'][i,1],gnss['data'][0,0], gnss['data'][0,1])

#print(gnss['data'])
# print(utm['data'])

# plt.plot(utm['data'][:,0], utm['data'][:,1])
# plt.show()
#print(gnss['data'][:,0])
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
# var_speed = 0.1
# var_gnss  = 0.01
var_speed = 10.1**2#0.14
var_yaw = 0.017**2
var_gnss  = 2**2

var_speed = 0.1#0.14
var_yaw = 0.017
var_gnss  = 2
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

#### 5. Main Filter Loop #######################################################################

################################################################################################
# Now that everything is set up, we can start taking in the sensor data and creating estimates
# for our state in a loop.
################################################################################################
count = 0
for k in range(1, imu_yaw["data"].shape[0]):  # start at 1 b/c we have initial prediction from gt
    delta_t = imu_yaw["t"][k] - imu_yaw["t"][k - 1]
    #delta_t = 0.2


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
                  [-vel*delta_t*np.sin(yaw), vel*delta_t*np.cos(yaw)]]).T
    # L = np.eye(2)
    Q = np.diag([var_speed, var_yaw])


    # 2. Propagate uncertainty
    p_cov[k] = F @ p_cov[k-1] @ F.T + L @ Q @ L.T

    # 2. Propagate uncertainty
    p_cov[k] = F @ p_cov[k-1] @ F.T + L @ Q @ L.T
    
    # 3. Check availability of GNSS measurements
    if not np.isnan(utm["data"][k][0]): 
        noise_gps = np.random.normal(0,2, (2,))
        #noise_gps = np.append(noise_gps,0)
        utm_data = utm["data"][k]#+noise_gps
        p_est[k], p_cov[k] = measurement_update(var_gnss, p_cov[k], utm_data, p_est[k])
        count = 0
    
    count += 1

ekf_latlon = np.zeros(p_est.shape)
for i in range(len(p_est)):
    ekf_latlon[i,:] = convertir_xy_a_gps(gnss['data'][0,0], gnss['data'][0,1], p_est[i,0], p_est[i,1])
    
#np.savetxt('D:\davpr\DocumentsD\estimation\state-stimation\data\ekf_latlon09.txt', ekf_latlon)

last = imu_yaw["data"].shape[0]-1

data2sent = str(p_est[last][0]) + "," + str(p_est[last][1]) #+ "," + str(gt["p"][last,0]) + "," + str(gt["p"][last,1]) 
print(data2sent)

# Crear la figura y los ejes
est_traj_fig = plt.figure()
ax = est_traj_fig.add_subplot(111)

# Graficar las variables en el plano XY
ax.plot(p_est[:, 0], p_est[:, 1], label='EKF')
a = utm["data"][:,0]
b = utm["data"][:,1]
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

ruh_m = plt.imread('./../data/map.png')
BBox = (-77.02268, -77.01940, -12.13693, -12.13490)
make2Plot(ruh_m, BBox, ekf_latlon[:,1], ekf_latlon[:,0], gnss['data'][:,1], gnss['data'][:,0], l1='EKF', l2 = 'GPS')
# make2Plot(ruh_m, BBox, ekf_latlon[:,1], ekf_latlon[:,0], l1='EKF')

plt.figure(figsize=(12, 8))
plt.imshow(ruh_m, zorder=0, extent=BBox, aspect='equal')

# Plot EKF data
plt.plot(ekf_latlon[:,1], ekf_latlon[:,0], 'r-', label='EKF')


# # Set axis labels and title
# plt.xlabel('Longitude')
# plt.ylabel('Latitude')
# plt.title('EKF Localization')

# Add legend
plt.legend()
plt.axis('off')

# Show the plot
plt.show()