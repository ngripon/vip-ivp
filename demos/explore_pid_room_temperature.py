import vip_ivp as vip


def regulated_room_temperature(kp=0.1, ki=0.0, kd=0.0):
    temperature_target = 20
    d_temperature = vip.loop_node()
    temperature = vip.integrate(d_temperature, 15)
    temperature_ext = 10
    k = 0.005  # Heat transfer coefficient
    error = temperature_target - temperature
    heater_power = kp * error + ki * vip.integrate(error, 0) + kd * vip.differentiate(error)
    d_temperature.loop_into(k * (heater_power - (temperature - temperature_ext)))
    return temperature


vip.explore(regulated_room_temperature, 3600, time_step=0.1)
