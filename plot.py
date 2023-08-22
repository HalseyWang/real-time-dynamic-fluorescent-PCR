import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import find_peaks

#temperature, fluorescence = [], []
#with open("C:/Users/wangy/Documents/PycharmProjects/Melt_data/6C.txt", "r") as file:
#    for line in file:
#        t, f = line.strip().split()
#        temperature.append(float(t))
#        fluorescence.append(float(f))

data = np.loadtxt("C:/Users/wangy/PycharmProjects/PycharmProjects/autodelta-15h/125nM-A.txt")
temperature = data[:, 0]
fluorescence = data[:, 1]
f = fluorescence


max_value = np.max(np.abs(f))
f_normalized = f / max_value


fig, ax = plt.subplots()
fig.subplots_adjust(left=0.2, right=0.95, top=0.95, bottom=0.15)
fig.set_alpha(0)


ax.spines['bottom'].set_linewidth(1.5)
ax.spines['left'].set_linewidth(1.5)
ax.spines['right'].set_linewidth(1.5)
ax.spines['top'].set_linewidth(1.5)

ax.xaxis.set_tick_params(width=1.5)
ax.yaxis.set_tick_params(width=1.5)


ax.xaxis.set_tick_params(labelsize=14)
ax.yaxis.set_tick_params(labelsize=14)


ax.set_xlabel("Temperature (°C)", fontsize=16)
ax.set_ylabel("Normalized -dF/dT", fontsize=16)



#peaks, _ = find_peaks(f_normalized, height=0)
#highest_peak_index = np.argmax(f_normalized[peaks])
#highest_peak_temperature = temperature[peaks][highest_peak_index]
#highest_peak_f_normalized = f_normalized[peaks][highest_peak_index]


#min_index = np.argmin(f_normalized)
#min_temperature = temperature[min_index]
#min_f_normalized = f_normalized[min_index]


#ax.annotate(f"Tm: {highest_peak_temperature:.1f} °C", xy=(highest_peak_temperature, highest_peak_f_normalized-0.1), xytext=(highest_peak_temperature-20, highest_peak_f_normalized-0.1), fontsize=14, color='blue')
#ax.annotate(f"Trans: {min_temperature:.1f} °C", xy=(min_temperature, min_f_normalized), xytext=(min_temperature+4, min_f_normalized), fontsize=14, color='blue')


np.savetxt('C:/Users/wangy/PycharmProjects/PycharmProjects/Anneal and Melt/125nM-A.txt', np.column_stack((temperature, f_normalized)), delimiter='\t')


#plt.plot(temperature, f_normalized, '-', label='Melt Polt')
#plt.plot(temperature, f_normalized, '-', label='Melt Derivative')
plt.plot(temperature, f_normalized, '-', label='Anneal')
plt.legend(loc="upper right", framealpha=0.0)
plt.xlabel("Temperature (°C)")
#plt.ylabel("Normalized -dF/dT")
plt.ylabel("Normalized Flu")
plt.savefig("C:/Users/wangy/PycharmProjects/PycharmProjects/Anneal and Melt/Normalized_125nM-A.png", transparent=True)
plt.show()