import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
from scipy.signal import find_peaks
from sklearn.preprocessing import MinMaxScaler


data = np.loadtxt("C:/Users/wangy/PycharmProjects/PycharmProjects/125nM-A.txt")
temperature = data[:, 0]
fluorescence = data[:, 1]


fluorescence_smooth = savgol_filter(fluorescence, 11, 3)


df = np.gradient(-fluorescence_smooth, temperature)


max_value = np.max(np.abs(df))
df_normalized = df / max_value

#scaler = MinMaxScaler(feature_range=(0, 1))
#df_normalized = scaler.fit_transform(df.reshape(-1, 1)).flatten()

#df_normalized = 2 * df_normalized - 1

fig, ax = plt.subplots()
fig.subplots_adjust(left=0.2, right=0.95, top=0.95, bottom=0.15)

ax.spines['bottom'].set_linewidth(1.5)
ax.spines['left'].set_linewidth(1.5)
ax.spines['right'].set_linewidth(1.5)
ax.spines['top'].set_linewidth(1.5)

ax.xaxis.set_tick_params(width=1.5)
ax.yaxis.set_tick_params(width=1.5)

ax.xaxis.set_tick_params(labelsize=14)
ax.yaxis.set_tick_params(labelsize=14)

ax.set_xlabel("Temperature (°C)", fontsize=16)
ax.set_ylabel("Fluorescence Intensity", fontsize=16)

peaks, _ = find_peaks(df_normalized, height=0)
highest_peak_index = np.argmax(df_normalized[peaks])
highest_peak_temperature = temperature[peaks][highest_peak_index]
highest_peak_df_normalized = df_normalized[peaks][highest_peak_index]
peaks_without_highest = np.delete(peaks, highest_peak_index)
second_highest_peak_index = np.argmax(df_normalized[peaks_without_highest])
second_highest_peak_temperature = temperature[peaks_without_highest][second_highest_peak_index]
second_highest_peak_df_normalized = df_normalized[peaks][second_highest_peak_index]

min_index = np.argmin(df_normalized)
min_temperature = temperature[min_index]
min_df_normalized = df_normalized[min_index]

header = f"Temperature (°C)\tNormalized Fluorescence\nHighest peak temperature:\t{highest_peak_temperature} °C\tSecond highest peak temperature:\t{second_highest_peak_temperature} °C\tMin peak Temperature:\t{min_temperature} °C"
np.savetxt('C:/Users/wangy/PycharmProjects/PycharmProjects/125nM-A.txt', np.column_stack((temperature, df_normalized)), delimiter='\t', header=header)

ax.annotate(f"Tm: {highest_peak_temperature:.1f} °C", xy=(highest_peak_temperature, highest_peak_df_normalized-0.1), xytext=(highest_peak_temperature-20, highest_peak_df_normalized-0.1), fontsize=14, color='blue')
#ax.annotate(f"Tm: {second_highest_peak_temperature:.1f} °C", xy=(second_highest_peak_temperature, second_highest_peak_df_normalized), xytext=(second_highest_peak_temperature+5, second_highest_peak_df_normalized-0.1), fontsize=14, color='blue')
#ax.annotate(f"Trans: {min_temperature:.1f} °C", xy=(min_temperature, min_df_normalized), xytext=(min_temperature+3, min_df_normalized), fontsize=14, color='blue')

#plt.plot(temperature, fluorescence_smooth, '-', label='Anneal Derivative', color=('green'))
#plt.plot(temperature, df, '-', label='Anneal Derivative', color=('green'))
plt.plot(temperature, df_normalized, '-', label='Anneal Derivative', color=('green'))
plt.legend(loc="upper right", framealpha=0.0)
plt.xlabel("Temperature (°C)")
plt.ylabel("Normalized -dF/dT")
plt.savefig('C:/Users/wangy/PycharmProjects/PycharmProjects/Normalized-125nM-A.png', transparent=True)
plt.show()