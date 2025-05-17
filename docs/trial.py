import numpy as np 

lift = np.array([0.0, 0.81, 1.5, 2.1, 2.9, 3.7, 4.6, 5.6, 6.6, 7.7])
cf = np.array([0.0, 0.08, 0.15, 0.21, 0.29, 0.35, 0.40, 0.43, 0.44, 0.44])  
# temp = np.ones(lift.shape)
# print(temp)
# lift = temp/lift
# print(lift)
theta = np.sqrt(np.arcsin(lift / 7.7))
print(theta)


integral = np.trapz(cf,theta) 

average_cf = integral / (theta[-1] - theta[0])

print(average_cf)

# Provided data
x_values = [378, 398, 406, 411, 418, 424, 432, 440, 451, 473, 495, 506, 514, 522, 528, 535, 540, 548, 568]
y_values = [0, 0.08, 0.15, 0.21, 0.29, 0.35, 0.4, 0.43, 0.44, 0.44, 0.44, 0.43, 0.4, 0.35, 0.29, 0.21, 0.15, 0.08, 0]

x_values = np.array(x_values)*(2*np.pi)/360
# Using the trapezoidal rule to calculate the integral
integral = np.trapz(y_values, x_values)

# Calculate the average
average_y = integral / (x_values[-1] - x_values[0])

print(average_y)