import sys
import numpy as np
import matplotlib.pyplot as plt
import math

RADIO_TIERRA_EN_M = 6371 * 1000

def getCartesian(lat, lon):
    # Convertir todas las coordenadas a radianes
    lat = lat*np.pi/180
    lon = lon*np.pi/180
    x = RADIO_TIERRA_EN_M * np.cos(lat) * np.cos(lon)
    y = RADIO_TIERRA_EN_M * np.cos(lat) * np.sin(lon)
    z = RADIO_TIERRA_EN_M * np.sin(lat);   

    return x, y

def cartesian2GPS(x, y):
    lat0 = -12.13548603	#-12.135551
    lon0 = -77.02197697 #-77.021916
    x0, y0 = getCartesian(lat0, lon0)

    x = x + x0
    y = y + y0

    lat = -np.arccos(np.sqrt(x**2+y**2)/RADIO_TIERRA_EN_M)*180/np.pi
    lon = np.arctan(y/x)*180/np.pi

    return lat, lon

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

data = np.genfromtxt('D:\davpr\DocumentsD\estimation\state-stimation\data\data_ekf02_filt.txt', delimiter=',')

# Crear los objetos gt, imu_f y gnss
hall = {'vel': data[:,1],  '_t': data[:,0]}
imu_yaw = {'data': data[:,2], 't': data[:,0]}
pos_xy = {'data': data[:,3:5], 't': data[:,0]}
gnss = {'data': np.zeros((len(pos_xy['data']),2)), 't': data[:,0]}
utm = {'data': np.zeros((len(pos_xy['data']),2)), 't': data[:,0]}

for i in range(len(gnss['data'])):
    if(not np.isnan(pos_xy['data'][i,0])):
        gnss['data'][i] = cartesian2GPS(pos_xy['data'][i,0], pos_xy['data'][i,1])
        utm['data'][i] = convertir_gps_a_xy(gnss['data'][i,0], gnss['data'][i,1],gnss['data'][0,0], gnss['data'][0,1])
        save_last = i
    else:
        gnss['data'][i] = gnss['data'][save_last]
        utm['data'][i] = convertir_gps_a_xy(gnss['data'][i,0], gnss['data'][i,1],gnss['data'][0,0], gnss['data'][0,1])
        
# for i in range(len(gnss['data'])):
#     utm['data'][i] = convertir_gps_a_xy(gnss['data'][i,0], gnss['data'][i,1],gnss['data'][0,0], gnss['data'][0,1])

print(gnss['data'])
# print(utm['data'])

#### 2. Constants ##############################################################################

################################################################################################
# Now that our data is set up, we can start getting things ready for our solver. One of the
# most important aspects of a filter is setting the estimated sensor variances correctly.
# We set the values here.
################################################################################################
# var_speed = 0.1
# var_gnss  = 0.01
var_speed = 0.0566 #0.14
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