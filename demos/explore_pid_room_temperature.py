import vip_ivp as vip

kp, ki, kd = 1, 0.5, 0

temperature_target = 20
temperature_ext = 10
k = 0.005  # Heat transfer coefficient


def compute_power(error):

    return kp * error + ki * vip.integrate(error, 0) + kd * error.derivative()


d_temperature = vip.loop_node()
temperature = vip.integrate(d_temperature, 15)

error = temperature_target - temperature
heater_power=compute_power(error)
d_temperature.loop_into(k * (heater_power - (temperature - temperature_ext)))

temperature.to_plot()
error.to_plot()

vip.solve(3600, time_step=0.1)
