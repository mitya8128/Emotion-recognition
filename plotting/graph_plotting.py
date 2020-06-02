import pandas as pd
import pylab as plt

# создаём табличку со всеми значениями из log-файла
file_name = "_emotion_training_50_1_VGG_16_GRU.log"  # имя log-файла
df = pd.read_csv(file_name)

# табличка со значениями метрики точности
df_accuracy = df.drop(columns=['loss', 'val_loss'])

# табличка со значениями функции потери
df_loss = df.drop(columns=['accuracy', 'val_accuracy'])

# изменяем все шрифты на Times New Roman
plt.rcParams["font.family"] = "Times New Roman"

# построение графика по метрике точности
ax = plt.gca()

df_accuracy.plot(kind='line', x='epoch', y='accuracy', ax=ax)
df_accuracy.plot(kind='line', x='epoch', y='val_accuracy', color='red', ax=ax)
plt.ylabel('Аккуратность', fontsize=12)
plt.xlabel('Эпоха', fontsize=12)
plt.legend(['обучающая', 'тестовая'], loc='upper left')
plt.title('График метрики точности', fontsize=12)

plt.show()
# plt.savefig('output.png')   # автоматическое сохранение файла

# построение графика функции потери
ax = plt.gca()

df_loss.plot(kind='line', x='epoch', y='loss', ax=ax)
df_loss.plot(kind='line', x='epoch', y='val_loss', color='red', ax=ax)
plt.ylabel('Функция потери', fontsize=12)
plt.xlabel('Эпоха', fontsize=12)
plt.legend(['обучающая', 'тестовая'], loc='upper left')
plt.title('График функции потери', fontsize=12)

plt.show()
